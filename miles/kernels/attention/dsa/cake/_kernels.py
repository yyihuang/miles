"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0

Host side of the generated DSA training kernels: geometry rules and named launches.

The flat contract every kernel sees: query rows ``[num_rows, heads, d]``, global key rows
``[num_kv, d]`` and per-row key indices ``[num_rows, topk]`` (``-1`` pads).  The kernel variant is a
pure function of the shape (heads per CTA, MLA tail, padded indexer heads), mirroring the generator's
host code one to one so the exported kernels are launched exactly as they were validated.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch

from ._jit import MODULES, device_arch, kernel, prebuild

LOG2E = 1.4426950408889634
INDEXER_HEAD_DIM = 128
INDEXER_HEADS = (8, 16, 32, 64)
ATTENTION_SHAPES = ((512, 64), (512, 0))  # (d_v, d_tail)
TOPK_STAGE_PREFIX = "dsa_topk_fp32_"

# (arch, family) pairs whose stages were prebuilt (or deliberately left lazy) in this process.
_PREBUILT: set[tuple[str, str]] = set()


def _is_topk_stage(name: str) -> bool:
    return name.startswith(TOPK_STAGE_PREFIX)


def _family_stages(family: str) -> tuple[str, ...]:
    """The registry stages one family prebuilds: ``"topk"`` = the ``dsa_topk_fp32_*`` capacity variants,
    ``"attention"`` = everything else (sparse attention, indexer logits / backward, clean, segmented reduce)."""
    return tuple(stage for stage in MODULES if _is_topk_stage(stage) == (family == "topk"))


def _kernel(name: str, device):
    """The native kernel for stage ``name`` on ``device``'s architecture.

    The first use of a stage family on an architecture prebuilds that family's stages in parallel (about a
    minute per stage), so the attention / indexer operators never compile the top-k variants and the top-k
    backend never compiles the attention kernels; ``MILES_DSA_TRAIN_PREBUILD=0`` builds each stage lazily.
    """
    arch = device_arch(device)
    family = "topk" if _is_topk_stage(name) else "attention"
    if (arch, family) not in _PREBUILT:
        _PREBUILT.add((arch, family))
        if os.environ.get("MILES_DSA_TRAIN_PREBUILD", "1") != "0":
            prebuild(arch, stages=_family_stages(family))
    return kernel(name, arch)


def _attention_name(kind: str, block_h: int, d_v: int, d_tail: int) -> str:
    return f"dsa_attention_{kind}_h{block_h}_d{d_v}t{d_tail}"


# --------------------------------------------------------------------------- #
# sparse attention
# --------------------------------------------------------------------------- #


def attention_block_h(heads: int) -> int:
    """Heads per CTA: 16 for up to 16 heads, otherwise 32 (larger head counts use several CTAs)."""
    return 16 if heads <= 16 else 32


def attention_head_blocks(heads: int) -> int:
    block = attention_block_h(heads)
    return (heads + block - 1) // block


@dataclass(frozen=True)
class AttentionGeometry:
    heads: int
    d_v: int
    d_tail: int
    block_h: int
    head_blocks: int
    topk: int

    @property
    def d_qk(self) -> int:
        return self.d_v + self.d_tail

    @property
    def key_block(self) -> int:
        return 32 if self.block_h == 32 else 64


def attention_geometry(q, kv, indices, d_v: int) -> AttentionGeometry:
    num_rows, heads, d_qk = q.shape
    if kv.dim() != 2 or kv.shape[1] != d_qk:
        raise ValueError(f"kv must be [num_kv, {d_qk}], got {tuple(kv.shape)}")
    if (d_v, d_qk - d_v) not in ATTENTION_SHAPES:
        raise ValueError(f"unsupported (d_v, d_tail) = {(d_v, d_qk - d_v)}; supported: {ATTENTION_SHAPES}")
    if indices.dim() != 2 or indices.shape[0] != num_rows:
        raise ValueError(f"indices must be [{num_rows}, topk], got {tuple(indices.shape)}")
    block_h = attention_block_h(heads)
    topk = int(indices.shape[1])
    if topk % 64 != 0:
        raise ValueError(f"topk must be a multiple of 64 (pad with -1), got {topk}")
    return AttentionGeometry(heads, d_v, d_qk - d_v, block_h, attention_head_blocks(heads), topk)


def dsa_attention_forward(q, kv, indices, sink, sm_scale: float, *, d_v: int = 512, out=None):
    """Flat-contract forward: ``(out[num_rows, heads, d_v] BF16, lse[num_rows, heads] FP32 base-2)``.

    ``out`` may be a caller-owned contiguous ``[num_rows, heads, d_v]`` buffer (for example a flat view of a
    layout-shaped tensor); it is allocated here otherwise.
    """
    geo = attention_geometry(q, kv, indices, d_v)
    num_rows = q.shape[0]
    if out is None:
        out = torch.empty(num_rows, geo.heads, geo.d_v, dtype=q.dtype, device=q.device)
    elif out.shape != (num_rows, geo.heads, geo.d_v) or out.dtype != q.dtype or not out.is_contiguous():
        raise ValueError(f"out must be a contiguous [{num_rows}, {geo.heads}, {geo.d_v}] {q.dtype} tensor")
    lse = torch.empty(num_rows, geo.heads, dtype=torch.float32, device=q.device)
    if num_rows == 0:
        return out, lse
    _kernel(_attention_name("fwd", geo.block_h, geo.d_v, geo.d_tail), q.device).launch(
        grid=(num_rows, geo.head_blocks, 1),
        q=q,
        kv=kv,
        sink=sink,
        indices=indices,
        out=out,
        lse=lse,
        heads=geo.heads,
        topk=geo.topk,
        scale_log2=float(sm_scale) * LOG2E,
    )
    return out, lse


def dsa_attention_backward_rows(q, kv, do, indices, lse, delta, dq, partial, sm_scale: float, geo: AttentionGeometry):
    """Launch the backward kernel on a contiguous slice of query rows (fills ``dq`` rows and ``partial``)."""
    num_rows = q.shape[0]
    if num_rows == 0:
        return
    _kernel(_attention_name("bwd", geo.block_h, geo.d_v, geo.d_tail), q.device).launch(
        grid=(num_rows, geo.head_blocks, 1),
        q=q,
        kv=kv,
        do=do,
        indices=indices,
        lse=lse,
        delta=delta,
        dq=dq,
        partial=partial,
        heads=geo.heads,
        topk=geo.topk,
        num_head_blocks=geo.head_blocks,
        sm_scale=float(sm_scale),
        scale_log2=float(sm_scale) * LOG2E,
    )


def dsa_segmented_reduce_launch(partial, order, seg_start, seg_count, out) -> None:
    """``out[key, :] += sum partial[order[p], :]`` over each key's sorted segment, in order."""
    num_out = int(out.shape[0])
    dim = int(partial.shape[1])
    if num_out == 0:
        return
    chunks = dim // 4
    keys_per_cta = 1
    for candidate in (4, 2):
        if 128 // candidate >= chunks:
            keys_per_cta = candidate
            break
    _kernel("dsa_segmented_reduce", out.device).launch(
        grid=((num_out + keys_per_cta - 1) // keys_per_cta, 1, 1),
        partial=partial,
        order=order,
        seg_start=seg_start,
        seg_count=seg_count,
        out=out,
        dim=dim,
        keys_per_cta=keys_per_cta,
        num_out=num_out,
    )


# --------------------------------------------------------------------------- #
# lightning indexer
# --------------------------------------------------------------------------- #


def indexer_padded_heads(heads: int) -> int:
    if heads <= 0 or heads > 64:
        raise ValueError(f"indexer heads must be in 1..64, got {heads}")
    return 16 if heads <= 16 else (32 if heads <= 32 else 64)


def indexer_key_block(heads: int) -> int:
    return 64 if indexer_padded_heads(heads) == 16 else 32


def dsa_indexer_logits(index_q, index_k, weights, cu_ks, cu_ke, *, batch: int, seq_len: int, seq_len_kv: int):
    """Batched indexer logits ``[batch, seq_len, seq_len_kv]`` (FP32; ``-inf`` outside each query's key range).

    ``index_q`` is ``[batch * seq_len, heads, 128]`` BF16, ``index_k`` ``[batch * seq_len_kv, 128]``
    BF16, ``weights`` ``[batch * seq_len, heads]`` FP32, ``cu_ks``/``cu_ke`` int32 ``[seq_len]`` key
    bounds shared by the batch.
    """
    heads, dim = int(index_q.shape[1]), int(index_q.shape[2])
    if dim != INDEXER_HEAD_DIM:
        raise ValueError(f"indexer head dim must be {INDEXER_HEAD_DIM}, got {dim}")
    if heads not in INDEXER_HEADS:
        raise ValueError(f"indexer heads must be one of {INDEXER_HEADS}, got {heads}")
    logits = torch.empty(batch, seq_len, seq_len_kv, dtype=torch.float32, device=index_q.device)
    if batch == 0 or seq_len == 0:
        return logits
    bq = 64 // heads
    if seq_len_kv > 0:
        _kernel(f"dsa_indexer_logits_h{heads}", index_q.device).launch(
            grid=((seq_len + bq - 1) // bq, batch, 1),
            index_q=index_q,
            index_k=index_k,
            weights=weights,
            cu_ks=cu_ks,
            cu_ke=cu_ke,
            logits=logits,
            seq_len=seq_len,
            seq_len_kv=seq_len_kv,
        )
        _kernel("dsa_indexer_clean", index_q.device).launch(
            grid=(seq_len, batch, 1),
            logits=logits,
            cu_ks=cu_ks,
            cu_ke=cu_ke,
            seq_len=seq_len,
            seq_len_kv=seq_len_kv,
        )
    return logits


def dsa_indexer_backward_rows(index_q, index_k, weights, topk_indices, grad_scores, d_index_q, d_weights, partial):
    """Launch the indexer backward on a contiguous slice of query rows.

    ``partial`` receives one FP32 ``[128]`` key-side partial per ``(row, slot)`` at
    ``row * topk + slot``; padded slots (``-1``) hold exact zeros.
    """
    num_rows, heads, dim = index_q.shape
    if num_rows == 0:
        return
    if dim != INDEXER_HEAD_DIM:
        raise ValueError(f"indexer head dim must be {INDEXER_HEAD_DIM}, got {dim}")
    topk = int(topk_indices.shape[1])
    hp = indexer_padded_heads(int(heads))
    if topk % indexer_key_block(int(heads)) != 0:
        raise ValueError(
            f"indexer topk must be a multiple of {indexer_key_block(int(heads))} for {heads} heads, got {topk}"
        )
    _kernel(f"dsa_indexer_bwd_h{hp}", index_q.device).launch(
        grid=(num_rows, 1, 1),
        index_q=index_q,
        index_k=index_k,
        weights=weights,
        topk_indices=topk_indices,
        grad_scores=grad_scores,
        d_index_q=d_index_q,
        d_weights=d_weights,
        partial=partial,
        heads=int(heads),
        topk=topk,
    )


# --------------------------------------------------------------------------- #
# indexer top-k
# --------------------------------------------------------------------------- #

# One exported variant per shared-memory row capacity (``num_keys <= capacity``), all traced for
# ``K = TOPK_K`` slots: ``dsa_topk_fp32_c{capacity}_k512``.  ``TOPK_CAPACITIES[-1]`` is the kernel's
# maximum row length; 4096 is the DeepSeek-V4.1 compressed row length.
TOPK_CAPACITIES = (4096, 8192, 16384, 32768)
TOPK_K = 512

_TOPK_NO_ROW_END: dict = {}


def _topk_name(capacity: int) -> str:
    return f"{TOPK_STAGE_PREFIX}c{capacity}_k{TOPK_K}"


def topk_capacity(num_keys: int) -> int:
    """The exported top-k variant for a row length: the smallest ``TOPK_CAPACITIES`` entry with ``num_keys <= capacity``.

    A variant stages at most ``capacity`` positions per row and has no device-side ``num_keys`` check,
    so the host must select it here.
    """
    num_keys = int(num_keys)
    if num_keys < 1 or num_keys > TOPK_CAPACITIES[-1]:
        raise ValueError(f"top-k row length must be in 1..{TOPK_CAPACITIES[-1]}, got {num_keys}")
    return next(capacity for capacity in TOPK_CAPACITIES if num_keys <= capacity)


def _topk_no_row_end(device):
    """The ignored ``row_end`` operand of a launch with ``has_row_end == 0`` (one int32 per device)."""
    tensor = _TOPK_NO_ROW_END.get(device)
    if tensor is None:
        tensor = _TOPK_NO_ROW_END[device] = torch.zeros(1, dtype=torch.int32, device=device)
    return tensor


def dsa_topk(logits, topk: int):
    """Deterministic FP32 row top-k: int32 ``[rows, topk]`` key indices of ``logits[rows, num_keys]``.

    Slot ``j`` of a row holds the key with the ``j``-th highest score (ties: the smaller index first);
    ``-inf`` and NaN positions are never selected and every unfilled slot is ``-1`` -- a tail after the
    valid picks, also when a row has fewer than ``topk`` finite scores (the kernel writes it; nothing is
    post-processed here).  Identical inputs give bit-identical outputs.  ``num_keys`` must be at most
    ``TOPK_CAPACITIES[-1]`` and ``topk`` at most ``TOPK_K``: the exported variants are traced for
    ``K = 512`` and a smaller ``topk`` keeps the first ``topk`` slots.  When ``topk > num_keys`` the
    row is padded with ``-inf`` columns up to ``topk`` and the kernel pads the missing slots with ``-1``.
    """
    if logits.dim() != 2 or logits.dtype != torch.float32:
        raise TypeError(f"top-k logits must be a 2-D float32 tensor, got {tuple(logits.shape)} {logits.dtype}")
    rows, num_keys = (int(v) for v in logits.shape)
    topk = int(topk)
    if topk < 1 or topk > TOPK_K:
        raise ValueError(f"the cake top-k backend supports 1 <= topk <= {TOPK_K}, got topk={topk}")
    if num_keys > TOPK_CAPACITIES[-1]:
        raise ValueError(
            f"the cake top-k backend supports at most {TOPK_CAPACITIES[-1]} keys per row, got num_keys={num_keys}"
        )
    if num_keys < topk:
        logits = torch.nn.functional.pad(logits, (0, topk - num_keys), value=float("-inf"))
        num_keys = topk
    if not logits.is_cuda:
        raise ValueError("the cake top-k backend needs CUDA logits")
    name = _topk_name(topk_capacity(num_keys))
    if name not in MODULES:
        raise NotImplementedError(f"{name}: the generated top-k kernels are not in this checkout's registry.json")
    out = torch.empty(rows, TOPK_K, dtype=torch.int32, device=logits.device)
    if rows == 0:
        return out[:, :topk]
    _kernel(name, logits.device).launch(
        grid=(rows, 1, 1),
        logits=logits.contiguous(),
        out=out,
        row_end=_topk_no_row_end(logits.device),
        n_kv=num_keys,
        has_row_end=0,
    )
    return out if topk == TOPK_K else out[:, :topk].contiguous()


__all__ = [
    "ATTENTION_SHAPES",
    "INDEXER_HEADS",
    "INDEXER_HEAD_DIM",
    "MODULES",
    "TOPK_CAPACITIES",
    "TOPK_K",
    "AttentionGeometry",
    "attention_block_h",
    "attention_geometry",
    "attention_head_blocks",
    "dsa_attention_backward_rows",
    "dsa_attention_forward",
    "dsa_indexer_backward_rows",
    "dsa_indexer_logits",
    "dsa_segmented_reduce_launch",
    "dsa_topk",
    "indexer_key_block",
    "indexer_padded_heads",
    "topk_capacity",
]

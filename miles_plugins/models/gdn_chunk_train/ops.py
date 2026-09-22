"""Deterministic chunked gated delta rule for the Qwen GDN layers (forward + backward).

Drop-in replacement for flash-linear-attention v0.5.2's ``chunk_gated_delta_rule`` as the
Qwen3.5 / Qwen3-Next wrappers call it: ``[B, T, H, K]`` bf16 q/k, ``[B, T, HV, V]`` bf16 v with
``K = V = 128`` and ``HV % H == 0``, fp32 log-decay ``g``, bf16 or fp32 ``beta``, packed varlen
through ``cu_seqlens`` (``B = 1``), in-kernel q/k L2 normalisation, optional fp32
initial / final state, and context-parallel state passing through an ``FLACPContext``.

Every launch is one of the generated Cake kernels under ``csrc/`` (``mma.sync`` register-MMA,
chunk 64): the launch geometry is a pure function of the shape and the device, there are no
atomics and every reduction has a fixed order, so repeated forward and backward passes on
identical inputs are bit-identical. Context parallelism reuses FLA's ``fla.ops.cp``
pre-processing exactly (``chunk_gated_delta_rule_fwd_h_pre_process``, ``compress_h0``,
``expand_h0``, ``chunk_gated_delta_rule_bwd_dhu_pre_process``), so ``build_gdn_cp_context``
semantics are unchanged; those helpers are imported only when a ``cp_context`` is supplied.

Calls the kernels do not cover (a GPU other than SM100a / SM103a, head dims other than 128,
non-bf16 q/k/v, FLA options the kernels do not implement) fall back to FLA's operator for that
call and warn once, so training never stops on an unsupported configuration.
"""

from __future__ import annotations

import math
import os
import warnings
from dataclasses import dataclass
from functools import cache
from typing import Any

import torch

from ._jit import MODULES, SUPPORTED_CAPABILITIES, device_arch, kernel

__all__ = [
    "CHUNK",
    "HEAD_DIM",
    "ChunkGatedDeltaRuleFunction",
    "ChunkMeta",
    "chunk_gated_delta_rule",
    "deterministic_applies",
    "gdn_chunk_backward",
    "gdn_chunk_forward",
]

CHUNK = 64
HEAD_DIM = 128
# Register-resident scan configurations (value rows per CTA, K split), in order of preference.
REG_SCAN_CONFIGS = ((32, 2), (32, 1), (64, 1))
# Shared-memory V-block scan variants (the small-SMEM fallback), widest first.
SCAN_V_BLOCKS = (64, 32, 16, 8)
# FLA options the kernels do not implement; a truthy value routes the call to FLA.
_UNSUPPORTED_FLA_OPTIONS = ("use_gate_in_kernel", "head_first", "state_v_first", "safe_gate")


# --------------------------------------------------------------------------------------------
# Chunk table
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ChunkMeta:
    """Chunk table for packed varlen sequences (all int32 CUDA tensors).

    ``num_chunks`` is the launch extent; when the table is built on the device from
    ``cu_seqlens`` it is an upper bound (``T // 64 + num_seqs``) and the trailing entries are
    padding chunks with ``chunk_len == 0`` that every chunk kernel skips.
    """

    chunk_start: torch.Tensor
    chunk_len: torch.Tensor
    seq_chunk_start: torch.Tensor
    num_chunks: int
    num_seqs: int


def build_chunk_meta(seq_lens, device) -> ChunkMeta:
    """Exact chunk table from host-side sequence lengths (equal-length batches, host cu_seqlens)."""
    starts: list[int] = []
    lens: list[int] = []
    seq_chunk_start = [0]
    token = 0
    for seq_len in seq_lens:
        seq_len = int(seq_len)
        for chunk_start in range(0, seq_len, CHUNK):
            starts.append(token + chunk_start)
            lens.append(min(CHUNK, seq_len - chunk_start))
        seq_chunk_start.append(len(starts))
        token += seq_len

    def to_dev(values):
        return torch.tensor(values, dtype=torch.int32, device=device)

    return ChunkMeta(
        chunk_start=to_dev(starts),
        chunk_len=to_dev(lens),
        seq_chunk_start=to_dev(seq_chunk_start),
        num_chunks=len(starts),
        num_seqs=len(seq_lens),
    )


def build_chunk_meta_from_cu_seqlens(arch: str, cu_seqlens: torch.Tensor, total_tokens: int) -> ChunkMeta:
    """Chunk table from a device ``cu_seqlens`` without synchronising the host (one tiny kernel)."""
    if cu_seqlens.dim() != 1 or cu_seqlens.numel() < 2:
        raise ValueError("cu_seqlens must be a 1-D tensor with at least two entries")
    num_seqs = int(cu_seqlens.numel()) - 1
    num_chunks_max = total_tokens // CHUNK + num_seqs
    cu32 = cu_seqlens if cu_seqlens.dtype == torch.int32 else cu_seqlens.to(torch.int32)
    cu32 = cu32.contiguous()
    device = cu_seqlens.device
    chunk_start = torch.empty(num_chunks_max, dtype=torch.int32, device=device)
    chunk_len = torch.empty(num_chunks_max, dtype=torch.int32, device=device)
    seq_chunk_start = torch.empty(num_seqs + 1, dtype=torch.int32, device=device)
    kernel("meta", arch).launch(
        grid=(1, 1, 1),
        cu_seqlens=cu32,
        chunk_start=chunk_start,
        chunk_len=chunk_len,
        seq_chunk_start=seq_chunk_start,
        num_seqs=num_seqs,
        num_chunks_max=num_chunks_max,
        total_tokens=int(total_tokens),
    )
    return ChunkMeta(
        chunk_start=chunk_start,
        chunk_len=chunk_len,
        seq_chunk_start=seq_chunk_start,
        num_chunks=num_chunks_max,
        num_seqs=num_seqs,
    )


def _chunk_meta(arch, cu_seqlens, cu_seqlens_cpu, batch: int, seq_len: int, device) -> ChunkMeta:
    """Chunk table without a device-to-host synchronisation.

    Equal-length batches and a caller-supplied host copy of ``cu_seqlens`` build the exact table
    from host integers; a device-only ``cu_seqlens`` builds it on the GPU (padded launch extent).
    """
    if cu_seqlens is None:
        return build_chunk_meta([seq_len] * batch, device)
    if batch != 1:
        raise ValueError("cu_seqlens requires a leading batch dimension of 1")
    if cu_seqlens_cpu is not None:
        values = [int(x) for x in cu_seqlens_cpu.tolist()]
        return build_chunk_meta([b - a for a, b in zip(values[:-1], values[1:], strict=True)], device)
    return build_chunk_meta_from_cu_seqlens(arch, cu_seqlens, batch * seq_len)


# --------------------------------------------------------------------------------------------
# Launch geometry (pure functions of shape and device)
# --------------------------------------------------------------------------------------------


@cache
def _device_sm_count(device_index: int) -> int:
    return int(torch.cuda.get_device_properties(device_index).multi_processor_count)


@cache
def _device_smem_optin(device_index: int) -> int:
    return int(torch.cuda.get_device_properties(device_index).shared_memory_per_block_optin)


def _device_index(device: torch.device) -> int:
    return device.index if device.index is not None else torch.cuda.current_device()


def _reg_scan_smem_bytes(arch: str, kind: str, vb: int, ksplit: int) -> int:
    return int(MODULES[f"{kind}_r{vb}k{ksplit}"][arch]["dynamic_smem_bytes"])


def choose_register_scan_config(arch: str, kind: str, num_seqs: int, num_v_heads: int, device_index: int):
    """Register-scan (value rows per CTA, K split) whose grid is fully co-resident on the device.

    Every warp owns 16 value rows and K / ksplit keys, so the preferred config (32 rows, K split in
    two -> four warps) has the shortest per-chunk chain and fills all four SM sub-partitions; wider
    blocks are used only when the preferred grid would need more than one wave.
    ``MILES_GDN_SCAN_CFG=vb,ksplit`` pins a config for experiments. The choice depends only on the
    shape and the device, never on data.
    """
    forced = os.environ.get("MILES_GDN_SCAN_CFG")
    if forced:
        vb, ks = (int(x) for x in forced.split(","))
        return vb, ks
    sm_count = _device_sm_count(device_index)
    smem_optin = _device_smem_optin(device_index)
    for vb, ks in REG_SCAN_CONFIGS:
        ctas = num_seqs * num_v_heads * (HEAD_DIM // vb)
        threads = 32 * (vb // 16) * ks
        smem = _reg_scan_smem_bytes(arch, kind, vb, ks)
        resident = min(2048 // threads, 65536 // (256 * threads), smem_optin // smem, 4)
        if resident >= 1 and ctas <= sm_count * resident:
            return vb, ks
    return 64, 1


def use_register_scan(arch: str, kind: str, vb: int, ksplit: int, device_index: int) -> bool:
    """Register-resident scans are the default when their SMEM fits; ``MILES_GDN_SCAN=vblock`` forces the
    shared-memory V-block variants."""
    if os.environ.get("MILES_GDN_SCAN", "register").lower() == "vblock":
        return False
    return _reg_scan_smem_bytes(arch, kind, vb, ksplit) <= _device_smem_optin(device_index)


def choose_scan_v_block(num_seqs: int, num_v_heads: int, device_index: int) -> int:
    """Largest V block whose (sequence, head, V-block) grid covers the device's SMs."""
    sms = _device_sm_count(device_index)
    for vb in SCAN_V_BLOCKS:
        if num_seqs * num_v_heads * (HEAD_DIM // vb) >= sms:
            return vb
    return SCAN_V_BLOCKS[-1]


def _scan_kernel(arch: str, kind: str, meta: ChunkMeta, num_v_heads: int, device: torch.device):
    device_index = _device_index(device)
    rvb, rks = choose_register_scan_config(arch, kind, meta.num_seqs, num_v_heads, device_index)
    if use_register_scan(arch, kind, rvb, rks, device_index):
        return f"{kind}_r{rvb}k{rks}", (meta.num_seqs, num_v_heads, HEAD_DIM // rvb)
    vb = choose_scan_v_block(meta.num_seqs, num_v_heads, device_index)
    return f"{kind}_v{vb}", (meta.num_seqs, num_v_heads, HEAD_DIM // vb)


# --------------------------------------------------------------------------------------------
# Forward
# --------------------------------------------------------------------------------------------


def _launch_wy(arch, kn, v, g_cs, beta32, A, meta: ChunkMeta, num_heads: int, num_v_heads: int, *, recompute: bool):
    w = torch.empty(kn.shape[0], num_v_heads, HEAD_DIM, dtype=kn.dtype, device=kn.device)
    u = torch.empty_like(v)
    kernel("wy", arch).launch(
        grid=(meta.num_chunks, num_v_heads, 1),
        kn=kn,
        v=v,
        g_cs=g_cs,
        beta=beta32,
        A_out=A,
        w_out=w,
        u_out=u,
        chunk_start=meta.chunk_start,
        chunk_len=meta.chunk_len,
        num_heads=num_heads,
        num_v_heads=num_v_heads,
        recompute=1 if recompute else 0,
    )
    return w, u


def _launch_fwd_h(arch, kn, w, u, g_cs, initial_state, meta: ChunkMeta, num_heads: int, num_v_heads: int, *, store_final_state: bool):
    device = kn.device
    h = torch.empty(meta.num_chunks, num_v_heads, HEAD_DIM, HEAD_DIM, dtype=kn.dtype, device=device)
    v_new = torch.empty_like(u)
    # The kernel writes every element of ``final_state`` when requested; no fill kernel is needed.
    final_state = torch.empty(meta.num_seqs if store_final_state else 0, num_v_heads, HEAD_DIM, HEAD_DIM, dtype=torch.float32, device=device)
    h0 = initial_state.contiguous().float() if initial_state is not None else final_state
    name, grid = _scan_kernel(arch, "fwd_h", meta, num_v_heads, device)
    kernel(name, arch).launch(
        grid=grid,
        kn=kn,
        w=w,
        u=u,
        g_cs=g_cs,
        h0=h0,
        h_out=h,
        v_new=v_new,
        final_state=final_state,
        chunk_start=meta.chunk_start,
        chunk_len=meta.chunk_len,
        seq_chunk_start=meta.seq_chunk_start,
        num_heads=num_heads,
        num_v_heads=num_v_heads,
        use_initial_state=1 if initial_state is not None else 0,
        store_final_state=1 if store_final_state else 0,
    )
    return h, v_new, final_state


def gdn_chunk_forward_stage1(arch, q, k, v, g, beta, *, meta: ChunkMeta, scale: float | None = None, normalize_qk: bool = True) -> dict[str, Any]:
    """Prep (l2norm, gate cumsum) and WY (A, w, u); no state is touched yet. Flat ``[T, H, K]`` inputs."""
    total_tokens, num_heads, key_dim = q.shape
    num_v_heads, value_dim = v.shape[1], v.shape[2]
    if key_dim != HEAD_DIM or value_dim != HEAD_DIM:
        raise ValueError("the deterministic GDN kernels require K = V = 128")
    if num_v_heads % num_heads != 0:
        raise ValueError("num_v_heads must be a multiple of num_heads")
    if scale is None:
        scale = 1.0 / math.sqrt(key_dim)
    device = q.device
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    g32 = g.contiguous().float()
    beta = beta.contiguous()
    # BF16 beta is converted inside the prep kernel (no separate cast kernel); FP32 passes through.
    beta_bf16 = beta.dtype == torch.bfloat16
    beta32 = torch.empty(total_tokens, num_v_heads, dtype=torch.float32, device=device) if beta_bf16 else beta.float()

    qn = torch.empty_like(q)
    kn = torch.empty_like(k)
    rstd_q = torch.empty(total_tokens, num_heads, dtype=torch.float32, device=device)
    rstd_k = torch.empty_like(rstd_q)
    g_cs = torch.empty(total_tokens, num_v_heads, dtype=torch.float32, device=device)
    kernel("prep_bf16beta" if beta_bf16 else "prep", arch).launch(
        grid=(meta.num_chunks, num_v_heads, 1),
        q=q,
        k=k,
        g=g32,
        beta=beta,
        beta32_out=beta32,
        qn=qn,
        kn=kn,
        rstd_q=rstd_q,
        rstd_k=rstd_k,
        g_cs=g_cs,
        chunk_start=meta.chunk_start,
        chunk_len=meta.chunk_len,
        num_heads=num_heads,
        num_v_heads=num_v_heads,
        normalize_qk=1 if normalize_qk else 0,
    )
    A = torch.empty(total_tokens, num_v_heads, CHUNK, dtype=q.dtype, device=device)  # fully written by the WY kernel
    w, u = _launch_wy(arch, kn, v, g_cs, beta32, A, meta, num_heads, num_v_heads, recompute=False)
    return {
        "arch": arch,
        "normalize_qk": normalize_qk,
        "num_heads": num_heads,
        "num_v_heads": num_v_heads,
        "qn": qn,
        "kn": kn,
        "rstd_q": rstd_q,
        "rstd_k": rstd_k,
        "g_cs": g_cs,
        "A": A,
        "w": w,
        "u": u,
        "v": v,
        "beta32": beta32,
        "meta": meta,
        "scale": float(scale),
    }


def gdn_chunk_forward_stage2(stage1: dict[str, Any], *, initial_state=None, output_final_state: bool = False) -> dict[str, Any]:
    """State scan and output; extends the stage-1 dict in place."""
    arch = stage1["arch"]
    meta: ChunkMeta = stage1["meta"]
    kn, qn, w, u, g_cs, v = (stage1[name] for name in ("kn", "qn", "w", "u", "g_cs", "v"))
    num_heads, num_v_heads = stage1["num_heads"], stage1["num_v_heads"]
    h, v_new, final_state = _launch_fwd_h(arch, kn, w, u, g_cs, initial_state, meta, num_heads, num_v_heads, store_final_state=output_final_state)
    o = torch.empty_like(v)
    kernel("fwd_o", arch).launch(
        grid=(meta.num_chunks, num_v_heads, 1),
        qn=qn,
        kn=kn,
        v_new=v_new,
        h=h,
        g_cs=g_cs,
        o_out=o,
        chunk_start=meta.chunk_start,
        chunk_len=meta.chunk_len,
        num_heads=num_heads,
        num_v_heads=num_v_heads,
        scale=float(stage1["scale"]),
    )
    stage1.update(
        {
            "output": o,
            "final_state": final_state if output_final_state else None,
            "initial_state": initial_state,
            "h": h,
            "v_new": v_new,
        }
    )
    return stage1


def gdn_chunk_forward(
    arch,
    q,
    k,
    v,
    g,
    beta,
    *,
    meta: ChunkMeta,
    scale: float | None = None,
    initial_state=None,
    output_final_state: bool = False,
    normalize_qk: bool = True,
) -> dict[str, Any]:
    """Run the deterministic chunked forward on flat ``[T, H, K]`` inputs; returns outputs and the tape."""
    stage1 = gdn_chunk_forward_stage1(arch, q, k, v, g, beta, meta=meta, scale=scale, normalize_qk=normalize_qk)
    return gdn_chunk_forward_stage2(stage1, initial_state=initial_state, output_final_state=output_final_state)


# --------------------------------------------------------------------------------------------
# Backward
# --------------------------------------------------------------------------------------------


def gdn_chunk_backward(
    forward: dict[str, Any],
    do,
    dht=None,
    *,
    initial_state=None,
    dht_from_dv_local=None,
    dbeta_dtype=None,
) -> dict[str, Any]:
    """Run the deterministic chunked backward from the (possibly pruned) forward tape.

    ``forward`` must carry ``arch``, ``qn``, ``kn``, ``rstd_q``, ``rstd_k``, ``v``, ``g_cs``,
    ``beta32``, ``A``, ``meta``, ``scale`` and ``normalize_qk``. ``w``/``u``/``h``/``v_new`` are
    recomputed exactly (same kernels, same order) when the tape does not carry them.
    ``initial_state`` is the fp32 state the forward started from (``None`` for a zero state); its
    gradient is returned only when it is given. ``dht_from_dv_local(w, dv_local)`` optionally
    supplies the final-state gradient once ``dv_local`` exists (context-parallel state passing).
    """
    arch = forward["arch"]
    meta: ChunkMeta = forward["meta"]
    qn, kn = forward["qn"], forward["kn"]
    A = forward["A"]
    g_cs = forward["g_cs"]
    scale = float(forward["scale"])
    v = forward["v"]
    beta32 = forward["beta32"]
    total_tokens, num_heads, _ = qn.shape
    num_v_heads = v.shape[1]
    device = qn.device
    do = do.contiguous()
    if initial_state is None:
        initial_state = forward.get("initial_state")
    has_initial_state = initial_state is not None
    if forward.get("w") is None or forward.get("u") is None:
        w, u = _launch_wy(arch, kn, v, g_cs, beta32, A, meta, num_heads, num_v_heads, recompute=True)
    else:
        w, u = forward["w"], forward["u"]
    if forward.get("h") is None or forward.get("v_new") is None:
        h, v_new, _ = _launch_fwd_h(arch, kn, w, u, g_cs, initial_state, meta, num_heads, num_v_heads, store_final_state=False)
    else:
        h, v_new = forward["h"], forward["v_new"]

    dv_local = torch.empty_like(v)
    kernel("dv_local", arch).launch(
        grid=(meta.num_chunks, num_v_heads, 1),
        qn=qn,
        kn=kn,
        do=do,
        g_cs=g_cs,
        dv_local=dv_local,
        chunk_start=meta.chunk_start,
        chunk_len=meta.chunk_len,
        num_heads=num_heads,
        num_v_heads=num_v_heads,
        scale=scale,
    )

    if dht_from_dv_local is not None:
        dht = dht_from_dv_local(w, dv_local)
    dh = torch.empty_like(h)
    dv2 = torch.empty_like(v)
    # Written in full by the scan when requested; no fill kernel.
    dh0 = torch.empty(meta.num_seqs if has_initial_state else 0, num_v_heads, HEAD_DIM, HEAD_DIM, dtype=torch.float32, device=device)
    dht32 = dht.contiguous().float() if dht is not None else dh0
    scan_name, scan_grid = _scan_kernel(arch, "dhu", meta, num_v_heads, device)
    kernel(scan_name, arch).launch(
        grid=scan_grid,
        qn=qn,
        kn=kn,
        w=w,
        do=do,
        dv_local=dv_local,
        g_cs=g_cs,
        dht=dht32,
        dh_out=dh,
        dv2=dv2,
        dh0=dh0,
        chunk_start=meta.chunk_start,
        chunk_len=meta.chunk_len,
        seq_chunk_start=meta.seq_chunk_start,
        num_heads=num_heads,
        num_v_heads=num_v_heads,
        use_final_state_grad=1 if dht is not None else 0,
        store_initial_state_grad=1 if has_initial_state else 0,
        scale=scale,
    )

    dq_hv = torch.empty(total_tokens, num_v_heads, HEAD_DIM, dtype=qn.dtype, device=device)
    dk_hv = torch.empty_like(dq_hv)
    dw = torch.empty_like(w)
    dg1 = torch.empty(total_tokens, num_v_heads, dtype=torch.float32, device=device)
    kernel("dqkwg", arch).launch(
        grid=(meta.num_chunks, num_v_heads, 1),
        qn=qn,
        kn=kn,
        v_new=v_new,
        do=do,
        dv2=dv2,
        h=h,
        dh=dh,
        g_cs=g_cs,
        dq_out=dq_hv,
        dk_out=dk_hv,
        dw_out=dw,
        dg_out=dg1,
        chunk_start=meta.chunk_start,
        chunk_len=meta.chunk_len,
        num_heads=num_heads,
        num_v_heads=num_v_heads,
        scale=scale,
    )

    dk2 = torch.empty_like(dq_hv)
    dv = torch.empty_like(v)
    # ``dbeta`` is produced directly in the caller's dtype (bf16 or fp32); no cast kernel follows.
    dbeta_bf16 = dbeta_dtype == torch.bfloat16
    dbeta = torch.empty(total_tokens, num_v_heads, dtype=torch.bfloat16 if dbeta_bf16 else torch.float32, device=device)
    dg2 = torch.empty(total_tokens, num_v_heads, dtype=torch.float32, device=device)
    kernel("wy_bwd_bf16" if dbeta_bf16 else "wy_bwd", arch).launch(
        grid=(meta.num_chunks, num_v_heads, 1),
        kn=kn,
        v=v,
        A=A,
        dw=dw,
        du=dv2,
        g_cs=g_cs,
        beta=beta32,
        dk2_out=dk2,
        dv_out=dv,
        dbeta_out=dbeta,
        dg2_out=dg2,
        chunk_start=meta.chunk_start,
        chunk_len=meta.chunk_len,
        num_heads=num_heads,
        num_v_heads=num_v_heads,
    )

    dq = torch.empty_like(qn)
    dk = torch.empty_like(kn)
    dg = torch.empty_like(dg2)
    kernel("finalize", arch).launch(
        grid=(meta.num_chunks, num_v_heads, 1),
        qn=qn,
        kn=kn,
        rstd_q=forward["rstd_q"],
        rstd_k=forward["rstd_k"],
        dq_hv=dq_hv,
        dk_hv=dk_hv,
        dk2=dk2,
        dg1=dg1,
        dg2=dg2,
        dq_out=dq,
        dk_out=dk,
        dg_out=dg,
        chunk_start=meta.chunk_start,
        chunk_len=meta.chunk_len,
        num_heads=num_heads,
        num_v_heads=num_v_heads,
        normalize_qk=1 if forward.get("normalize_qk", True) else 0,
    )
    return {
        "dq": dq,
        "dk": dk,
        "dv": dv,
        "dg": dg,
        "dbeta": dbeta,
        "dinitial_state": dh0 if has_initial_state else None,
    }


# --------------------------------------------------------------------------------------------
# FLA-signature autograd operator
# --------------------------------------------------------------------------------------------


def _cp_modules():
    from fla.ops.cp.chunk_delta_h import (
        chunk_gated_delta_rule_bwd_dhu_pre_process,
        chunk_gated_delta_rule_fwd_h_pre_process,
        compress_h0,
        expand_h0,
    )

    return (
        chunk_gated_delta_rule_fwd_h_pre_process,
        chunk_gated_delta_rule_bwd_dhu_pre_process,
        compress_h0,
        expand_h0,
    )


class ChunkGatedDeltaRuleFunction(torch.autograd.Function):
    """Deterministic chunked gated delta rule with FLA's autograd contract."""

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor | None,
        output_final_state: bool,
        cu_seqlens: torch.Tensor | None,
        cu_seqlens_cpu: torch.Tensor | None,
        use_qk_l2norm_in_kernel: bool,
        cp_context: Any,
    ):
        arch = device_arch(q.device)
        batch, seq_len, num_heads, key_dim = q.shape
        num_v_heads, value_dim = v.shape[2], v.shape[3]

        def flat(t, heads, dim):
            return t.reshape(batch * seq_len, heads, dim)

        q_flat, k_flat = flat(q, num_heads, key_dim), flat(k, num_heads, key_dim)
        v_flat = flat(v, num_v_heads, value_dim)
        g_flat = g.reshape(batch * seq_len, num_v_heads)
        beta_flat = beta.reshape(batch * seq_len, num_v_heads)
        meta = _chunk_meta(arch, cu_seqlens, cu_seqlens_cpu, batch, seq_len, q.device)

        use_cp = cp_context is not None and getattr(cp_context, "group", None) is not None
        if use_cp and initial_state is not None:
            raise ValueError("chunk_gated_delta_rule: initial_state must be None under context parallelism")

        # prep + WY first; CP derives this rank's incoming state from w/u
        if use_cp:
            fwd_h_pre, _bwd_pre, compress_h0, _expand_h0 = _cp_modules()
            fwd = gdn_chunk_forward_stage1(
                arch,
                q_flat,
                k_flat,
                v_flat,
                g_flat,
                beta_flat,
                meta=meta,
                scale=scale,
                normalize_qk=use_qk_l2norm_in_kernel,
            )
            initial_state = fwd_h_pre(
                k=fwd["kn"].unsqueeze(0),
                w=fwd["w"].unsqueeze(0),
                u=fwd["u"].unsqueeze(0),
                g=fwd["g_cs"].unsqueeze(0),
                cu_seqlens=cu_seqlens,
                initial_state=None,
                context=cp_context,
                state_v_first=False,
                chunk_size=CHUNK,
            )
            fwd = gdn_chunk_forward_stage2(fwd, initial_state=initial_state, output_final_state=output_final_state)
            saved_state = compress_h0(initial_state, context=cp_context)
        else:
            fwd = gdn_chunk_forward(
                arch,
                q_flat,
                k_flat,
                v_flat,
                g_flat,
                beta_flat,
                meta=meta,
                scale=scale,
                initial_state=initial_state,
                output_final_state=output_final_state,
                normalize_qk=use_qk_l2norm_in_kernel,
            )
            saved_state = initial_state

        ctx.save_for_backward(
            fwd["qn"],
            fwd["kn"],
            fwd["rstd_q"],
            fwd["rstd_k"],
            fwd["v"],
            fwd["g_cs"],
            fwd["beta32"],
            fwd["A"],
            saved_state,
            cu_seqlens,
        )
        ctx.arch = arch
        ctx.meta = meta
        ctx.scale = float(scale)
        ctx.shapes = (batch, seq_len, num_heads, num_v_heads, key_dim, value_dim)
        ctx.normalize_qk = bool(use_qk_l2norm_in_kernel)
        ctx.cp_context = cp_context if use_cp else None
        ctx.has_initial_state_input = initial_state is not None and not use_cp
        ctx.dtypes = (g.dtype, beta.dtype)
        output = fwd["output"].reshape(batch, seq_len, num_v_heads, value_dim)
        final_state = fwd["final_state"] if output_final_state else None
        return output.to(q.dtype), final_state

    @staticmethod
    def backward(ctx, do: torch.Tensor, dht: torch.Tensor | None):
        qn, kn, rstd_q, rstd_k, v_flat, g_cs, beta32, A, saved_state, cu_seqlens = ctx.saved_tensors
        batch, seq_len, num_heads, num_v_heads, key_dim, value_dim = ctx.shapes
        g_dtype, beta_dtype = ctx.dtypes
        do_flat = do.reshape(batch * seq_len, num_v_heads, value_dim).contiguous()
        tape = {
            "arch": ctx.arch,
            "qn": qn,
            "kn": kn,
            "rstd_q": rstd_q,
            "rstd_k": rstd_k,
            "v": v_flat,
            "g_cs": g_cs,
            "beta32": beta32,
            "A": A,
            "meta": ctx.meta,
            "scale": ctx.scale,
            "normalize_qk": ctx.normalize_qk,
        }
        cp_context = ctx.cp_context
        initial_state = saved_state
        dht_hook = None
        if cp_context is not None:
            _fwd_pre, bwd_dhu_pre, _compress_h0, expand_h0 = _cp_modules()
            initial_state = expand_h0(saved_state, context=cp_context)
            if dht is not None:
                raise ValueError("chunk_gated_delta_rule: final-state gradients are not part of the CP contract")

            def dht_hook(w, dv_local):
                dht_local, _ = bwd_dhu_pre(
                    q=qn.unsqueeze(0),
                    k=kn.unsqueeze(0),
                    w=w.unsqueeze(0),
                    do=do_flat.unsqueeze(0),
                    dv=dv_local.unsqueeze(0),
                    g=g_cs.unsqueeze(0),
                    scale=ctx.scale,
                    cu_seqlens=cu_seqlens,
                    dht=None,
                    initial_state=initial_state,
                    context=cp_context,
                    state_v_first=False,
                    chunk_size=CHUNK,
                )
                return dht_local

        grads = gdn_chunk_backward(tape, do_flat, dht, initial_state=initial_state, dht_from_dv_local=dht_hook, dbeta_dtype=beta_dtype)
        dq = grads["dq"].reshape(batch, seq_len, num_heads, key_dim)
        dk = grads["dk"].reshape(batch, seq_len, num_heads, key_dim)
        dv = grads["dv"].reshape(batch, seq_len, num_v_heads, value_dim)
        dg = grads["dg"].reshape(batch, seq_len, num_v_heads).to(g_dtype)  # no-op for the fp32 gate
        dbeta = grads["dbeta"].reshape(batch, seq_len, num_v_heads).to(beta_dtype)  # no-op: produced in beta's dtype
        dh0 = grads["dinitial_state"] if ctx.has_initial_state_input else None
        return dq, dk, dv, dg, dbeta, None, dh0, None, None, None, None, None


# --------------------------------------------------------------------------------------------
# Public entry: FLA signature, FLA fallback outside the kernels' domain
# --------------------------------------------------------------------------------------------


def deterministic_applies(
    *,
    capability: tuple[int, int],
    key_dim: int,
    value_dim: int,
    num_heads: int,
    num_v_heads: int,
    dtype: torch.dtype,
    chunk_size: int = 64,
    options: dict[str, Any] | None = None,
) -> bool:
    """Pure domain check for the deterministic kernels (SM100a / SM103a, K = V = 128, bf16, chunk 64).

    Sequence lengths, ``cu_seqlens`` packings, ``initial_state`` / ``output_final_state`` and
    ``cp_context`` do not restrict the domain.
    """
    if any(bool(value) for name, value in (options or {}).items() if name in _UNSUPPORTED_FLA_OPTIONS):
        return False
    return capability in SUPPORTED_CAPABILITIES and key_dim == HEAD_DIM and value_dim == HEAD_DIM and num_v_heads % num_heads == 0 and dtype == torch.bfloat16 and chunk_size == CHUNK


_fallback_warned = False


def _warn_fallback_once(reason: str) -> None:
    global _fallback_warned
    if not _fallback_warned:
        _fallback_warned = True
        warnings.warn(
            f"chunk_gated_delta_rule: this call is outside the deterministic GDN kernels' domain ({reason}); falling back to flash-linear-attention for it (this warning is shown once)",
            RuntimeWarning,
            stacklevel=3,
        )


def _fla_chunk_gated_delta_rule():
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule as fla_op

    return fla_op


def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    cp_context: Any = None,
    chunk_size: int = 64,
    **options: Any,
):
    """flash-linear-attention v0.5.2 ``chunk_gated_delta_rule`` signature, deterministic kernels.

    Returns ``(output, final_state)`` like FLA. Calls outside :func:`deterministic_applies`
    (another GPU generation, head dims other than 128, non-bf16 q/k/v, ``use_gate_in_kernel`` and
    the other FLA options the kernels do not implement) run FLA's operator instead and warn once.
    """
    capability = torch.cuda.get_device_capability(q.device) if q.is_cuda else (0, 0)
    if not deterministic_applies(
        capability=capability,
        key_dim=q.shape[-1],
        value_dim=v.shape[-1],
        num_heads=q.shape[-2],
        num_v_heads=v.shape[-2],
        dtype=q.dtype if q.dtype == k.dtype == v.dtype else torch.float32,
        chunk_size=chunk_size,
        options=options,
    ):
        unsupported = sorted(name for name, value in options.items() if name in _UNSUPPORTED_FLA_OPTIONS and value)
        reason = f"capability={capability}, K={q.shape[-1]}, V={v.shape[-1]}, H={q.shape[-2]}, HV={v.shape[-2]}, dtype={q.dtype}, chunk_size={chunk_size}, options={unsupported}"
        _warn_fallback_once(reason)
        return _fla_chunk_gated_delta_rule()(
            q,
            k,
            v,
            g,
            beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            cp_context=cp_context,
            chunk_size=chunk_size,
            **options,
        )
    if g.dtype != torch.float32:
        g = g.float()
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    if cu_seqlens is not None and cu_seqlens_cpu is None and cp_context is not None:
        cu_seqlens_cpu = getattr(cp_context, "cu_seqlens_cpu", None)
    return ChunkGatedDeltaRuleFunction.apply(
        q,
        k,
        v,
        g,
        beta,
        float(scale),
        initial_state,
        output_final_state,
        cu_seqlens,
        cu_seqlens_cpu,
        use_qk_l2norm_in_kernel,
        cp_context,
    )

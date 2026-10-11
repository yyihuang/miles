"""Backend selection for the DSA (DeepSeek Sparse Attention) indexer and sparse-attention kernels.

``--dsa-attention-backend`` picks one kernel family for both DSA model plugins (GLM-5 / DeepSeek-V3.2
``thd`` MLA and DeepSeek-V4 ``bshd`` MQA):

* ``tilelang`` (default): the fused TileLang kernels in ``miles/kernels/attention/dsa``.
* ``loom``: the generated deterministic kernels in ``miles_plugins/models/dsa_train`` (SM100a /
  SM103a): one launch for the batched DeepSeek-V4 indexer, bit-deterministic backward for both
  operators, FP32 attention sink and shared-latent gradients reduced without atomics.
* ``megatron``: Megatron-core's portable DSA path (bridge / LoRA only; not a plugin-spec choice).

``--miles-dsa-topk-backend loom`` independently selects the generated deterministic indexer top-k
(``dsa_train.loom_topk``); :func:`check_loom_topk_backend` rejects it at startup for models whose
top-k exceeds the kernel's ``LOOM_TOPK_MAX`` slots (GLM-5 / DeepSeek-V3.2 select 2048) or on
non-Blackwell devices.
"""

DSA_ATTENTION_BACKENDS = ("megatron", "tilelang", "loom")

# Limits of the generated top-k kernel (``dsa_train._kernels.TOPK_K`` / ``TOPK_CAPACITIES[-1]``).
LOOM_TOPK_MAX = 512
LOOM_TOPK_MAX_KEYS = 32768


def resolve_dsa_attention_backend(backend: str | None) -> str:
    """Validate the plugin-side kernel backend; ``None`` and ``megatron`` fall back to ``tilelang``."""
    if backend is None or backend == "megatron":
        # The plugin specs always run the fused miles kernels; ``megatron`` only selects Megatron-core's
        # own attention on the bridge path, where this module is not involved.
        return "tilelang"
    if backend not in ("tilelang", "loom"):
        raise ValueError(f"Unsupported DSA attention backend: {backend!r} (expected one of {DSA_ATTENTION_BACKENDS})")
    return backend


def use_loom_dsa(backend: str | None) -> bool:
    return resolve_dsa_attention_backend(backend) == "loom"


def check_loom_topk_backend(topk_backend: str | None, index_topk: int | None, *, model: str = "DSA") -> None:
    """Fail at startup when ``--miles-dsa-topk-backend loom`` cannot serve ``model``; other backends pass.

    The generated top-k kernel fills at most ``LOOM_TOPK_MAX`` slots per query row and runs on SM100a /
    SM103a only.  ``index_topk`` is the model's per-query selection width (``None`` when it is not known
    yet; the per-call check in ``dsa_train.loom_topk`` still applies).
    """
    if topk_backend != "loom":
        return
    if index_topk is not None and int(index_topk) > LOOM_TOPK_MAX:
        raise ValueError(
            f"--miles-dsa-topk-backend loom supports index_topk <= {LOOM_TOPK_MAX}, but {model} selects "
            f"{int(index_topk)} keys per query; use --miles-dsa-topk-backend torch or flashinfer."
        )
    import torch

    if torch.cuda.is_available():
        from miles_plugins.models.dsa_train._jit import SUPPORTED_CAPABILITIES

        capability = torch.cuda.get_device_capability()
        if capability not in SUPPORTED_CAPABILITIES:
            raise ValueError(
                "--miles-dsa-topk-backend loom requires a Blackwell SM100a / SM103a device (compute capability "
                f"10.0 / 10.3), got {capability[0]}.{capability[1]}; use --miles-dsa-topk-backend torch or flashinfer."
            )


def loom_dsa_ops():
    """Import the generated deterministic DSA operators (raises a clear error when they are missing)."""
    try:
        from miles_plugins.models import dsa_train
    except ImportError as exc:
        raise ImportError(
            "DSA attention backend 'loom' requires the generated kernels in miles_plugins/models/dsa_train."
        ) from exc
    return dsa_train

"""The DSA attention backend switch: one value selects the kernel family for both DSA model plugins.

Also the ``loom`` top-k backend (``--miles-dsa-topk-backend loom``) on CPU: resolver, registry, input validation
ahead of any launch, argument parsing and the GLM-5 (top-k 2048 > 512) rejection.  Nothing here needs CUDA.
"""

import ast
import importlib.util
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
TOPK_STAGES = {f"dsa_topk_fp32_c{capacity}_k512" for capacity in (4096, 8192, 16384, 32768)}


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, REPO / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_plugin_backend_resolution():
    backend = _load("test_dsa_backend_module", "miles_plugins/models/dsa_backend.py")
    assert backend.resolve_dsa_attention_backend(None) == "tilelang"
    assert backend.resolve_dsa_attention_backend("tilelang") == "tilelang"
    assert backend.resolve_dsa_attention_backend("loom") == "loom"
    # ``megatron`` is a bridge-path choice; the plugin specs keep their fused kernels.
    assert backend.resolve_dsa_attention_backend("megatron") == "tilelang"
    assert backend.use_loom_dsa("loom") and not backend.use_loom_dsa("tilelang")
    with pytest.raises(ValueError, match="Unsupported DSA attention backend"):
        backend.resolve_dsa_attention_backend("nope")


def test_bridge_path_rejects_loom():
    bridge = _load("test_megatron_bridge_utils_module", "miles/utils/megatron_bridge_utils.py")
    provider = Namespace(dsa_attention_backend="megatron")
    bridge.apply_dsa_backend_args(provider, Namespace(dsa_attention_backend="tilelang"))
    assert provider.dsa_attention_backend == "tilelang"
    with pytest.raises(ValueError, match="only available with the miles_plugins model specs"):
        bridge.apply_dsa_backend_args(provider, Namespace(dsa_attention_backend="loom"))


def test_generated_package_registry_covers_both_architectures():
    """Every exported stage is exported for SM100a and SM103a from one source set that exists on disk."""
    jit = _load("test_dsa_train_jit_module", "miles_plugins/models/dsa_train/_jit.py")
    assert set(jit.MODULES), "registry.json missing"
    expected = {
        *(f"dsa_attention_{kind}_h{h}_d512t{t}" for kind in ("fwd", "bwd") for h in (16, 32) for t in (64, 0)),
        *(f"dsa_indexer_logits_h{h}" for h in (8, 16, 32, 64)),
        *(f"dsa_indexer_bwd_h{h}" for h in (16, 32, 64)),
        "dsa_indexer_clean",
        "dsa_segmented_reduce",
        *TOPK_STAGES,
    }
    assert set(jit.MODULES) == expected
    for stage, record in jit.MODULES.items():
        assert set(record["arches"]) == {"sm_100a", "sm_103a"}, stage
        assert record["kernel_symbol"] == f"kernel_{record['name']}", stage
        for relative in record["sources"]:
            assert (REPO / "miles_plugins/models/dsa_train/csrc" / relative).is_file(), (stage, relative)
    for stage in TOPK_STAGES:
        record = jit.MODULES[stage]
        # The host wrapper binds these names and leaves the launch geometry to the record.
        assert [name for kind, name in record["arg_plan"] if kind != "grid"] == [
            "logits",
            "out",
            "row_end",
            "n_kv",
            "has_row_end",
        ], stage
        assert record["block"] == [256, 1, 1] and record["tma_workspace_bytes"] == 0, stage


# --------------------------------------------------------------------------- #
# loom top-k backend
# --------------------------------------------------------------------------- #


def test_core_topk_resolver_defers_the_plugin_import():
    """``get_dsa_topk_fn("loom")`` is a plain callable; the plugin (and its CUDA extensions) load on the first call."""
    topk = _load("test_dsa_topk_module", "miles/kernels/attention/dsa/topk.py")
    before = {name for name in sys.modules if name.startswith("miles_plugins")}
    fn = topk.get_dsa_topk_fn("loom")
    assert callable(fn) and fn is topk.loom_dsa_topk
    assert {name for name in sys.modules if name.startswith("miles_plugins")} == before
    assert topk.get_dsa_topk_fn("torch") is topk.torch_dsa_topk
    assert topk.get_dsa_topk_fn("flashinfer") is topk.flashinfer_dsa_topk
    with pytest.raises(ValueError, match="Unsupported miles DSA topk backend"):
        topk.get_dsa_topk_fn("nope")
    # The plugin validates before touching CUDA: GLM-5's 2048 exceeds the kernel's 512 slots.
    with pytest.raises(ValueError, match="512"):
        fn(torch.zeros(2, 8192), 2048)


def test_plugin_topk_backends_include_loom():
    from miles_plugins.models.dsa_train import ops

    assert set(ops._TOPK_BACKENDS) >= {"torch", "flashinfer", "loom"}
    assert ops.get_topk_fn("loom") is ops.loom_topk

    def replay_wrapper(logits, topk):  # a replay-manager style callable passes through untouched
        return logits

    assert ops.get_topk_fn(replay_wrapper) is replay_wrapper
    with pytest.raises(ValueError, match="unknown DSA top-k backend"):
        ops.get_topk_fn("nope")


def test_dsa_topk_validates_before_any_launch(monkeypatch):
    from miles_plugins.models.dsa_train import _kernels

    monkeypatch.setattr(_kernels, "_kernel", lambda name, device: pytest.fail(f"{name} was launched"))
    logits = torch.zeros(3, 1024)
    with pytest.raises(TypeError, match="float32"):
        _kernels.dsa_topk(logits.half(), 64)
    with pytest.raises(TypeError, match="2-D"):
        _kernels.dsa_topk(logits.unsqueeze(0), 64)
    with pytest.raises(ValueError, match="32768"):
        _kernels.dsa_topk(torch.zeros(1, 32769), 64)
    with pytest.raises(ValueError, match="512"):
        _kernels.dsa_topk(logits, 513)
    with pytest.raises(ValueError, match="1 <= topk"):
        _kernels.dsa_topk(logits, 0)
    # Valid requests on CPU tensors stop at the device check, still ahead of the kernel loader.
    with pytest.raises(ValueError, match="CUDA"):
        _kernels.dsa_topk(logits, 64)
    with pytest.raises(ValueError, match="CUDA"):
        _kernels.dsa_topk(torch.zeros(3, 100), 512)  # topk > num_keys: -inf columns are padded first
    assert [_kernels.topk_capacity(n) for n in (1, 4096, 4097, 8192, 8193, 16384, 16385, 32768)] == [
        4096,
        4096,
        8192,
        8192,
        16384,
        16384,
        32768,
        32768,
    ]
    for bad in (0, 32769):
        with pytest.raises(ValueError, match="1..32768"):
            _kernels.topk_capacity(bad)


def test_parser_accepts_the_loom_topk_backend():
    from miles_plugins.models.glm5.arguments import add_dsa_arguments, normalize_dsa_args

    parser = add_dsa_arguments(ArgumentParser())
    assert parser.parse_args([]).miles_dsa_topk_backend == "torch"
    assert parser.parse_args(["--miles-dsa-topk-backend", "loom"]).miles_dsa_topk_backend == "loom"
    with pytest.raises(SystemExit):
        parser.parse_args(["--miles-dsa-topk-backend", "nope"])
    # Native Megatron DSA cannot host the generated kernel.
    with pytest.raises(ValueError, match="only available with --dsa-impl miles"):
        normalize_dsa_args(Namespace(dsa_impl="megatron", miles_dsa_topk_backend="loom"), None)
    args = Namespace(dsa_impl="miles", miles_dsa_topk_backend="loom")
    normalize_dsa_args(args, None)
    assert args.miles_dsa_topk_backend == "loom"


def test_glm5_rejects_the_loom_topk_backend(monkeypatch):
    backend = _load("test_dsa_backend_topk_module", "miles_plugins/models/dsa_backend.py")
    with pytest.raises(ValueError, match=r"index_topk <= 512.*GLM-5.*2048 keys per query"):
        backend.check_loom_topk_backend("loom", 2048, model="GLM-5")
    backend.check_loom_topk_backend("torch", 2048, model="GLM-5")
    backend.check_loom_topk_backend("flashinfer", 2048, model="GLM-5")
    # Within the limit: the device check only runs on CUDA hosts (the kernel loader enforces it again).
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    backend.check_loom_topk_backend("loom", 512, model="DeepSeek-V4.1")
    backend.check_loom_topk_backend("loom", None, model="DeepSeek-V4.1")
    # The plugin specs call the guard (with the static key-row bound) before the model is built.
    guard, bound = "check_loom_topk_backend", "loom_topk_key_bound"
    for relative, scopes in (
        ("miles_plugins/models/glm5/glm5.py", {"get_glm5_spec": {guard, bound}, "__init__": {guard}}),
        ("miles_plugins/models/deepseek_v4/deepseek_v4.py", {"get_dsv4_spec": {guard, bound}}),
        ("miles_plugins/models/deepseek_v4_1/deepseek_v4_1.py", {"get_dsv41_spec": {guard, bound}}),
    ):
        called: dict[str, set] = {}
        for node in ast.walk(ast.parse((REPO / relative).read_text())):
            if isinstance(node, ast.FunctionDef):
                called.setdefault(node.name, set()).update(
                    getattr(call.func, "id", None) for call in ast.walk(node) if isinstance(call, ast.Call)
                )
        for scope, names in scopes.items():
            assert names <= called.get(scope, set()), (relative, scope, called.get(scope))


def test_loom_topk_backend_requires_blackwell(monkeypatch):
    backend = _load("test_dsa_backend_capability_module", "miles_plugins/models/dsa_backend.py")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (9, 0))
    with pytest.raises(ValueError, match=r"SM100a / SM103a.*10\.0 / 10\.3.*got 9\.0"):
        backend.check_loom_topk_backend("loom", 512, model="DeepSeek-V4.1")
    backend.check_loom_topk_backend("torch", 512, model="DeepSeek-V4.1")  # other backends never probe the device
    for capability in ((10, 0), (10, 3)):
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None, cap=capability: cap)
        backend.check_loom_topk_backend("loom", 512, model="DeepSeek-V4.1")


def test_loom_topk_backend_rejects_long_key_rows(monkeypatch):
    backend = _load("test_dsa_backend_rows_module", "miles_plugins/models/dsa_backend.py")
    assert backend.loom_topk_key_bound(None) is None
    assert backend.loom_topk_key_bound(4096) == 4096  # thd: the packed stream itself
    assert backend.loom_topk_key_bound(65536, (0, 2, 1)) == 65536  # V4.1: a ratio-1 layer keeps every key
    assert backend.loom_topk_key_bound(65536, (0, 2)) == 32768
    assert backend.loom_topk_key_bound(131073, (4,)) == 32769  # V4: ceil(seq_length / 4)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    backend.check_loom_topk_backend("loom", 512, model="DeepSeek-V4.1", max_keys=32768)
    backend.check_loom_topk_backend("loom", 512, model="DeepSeek-V4.1", max_keys=None)
    backend.check_loom_topk_backend("torch", 512, model="DeepSeek-V4.1", max_keys=10**6)
    with pytest.raises(ValueError, match=r"at most 32768 compressed keys.*DeepSeek-V4\.1.*32769 keys"):
        backend.check_loom_topk_backend("loom", 512, model="DeepSeek-V4.1", max_keys=32769)


def test_prebuild_is_scoped_to_the_stage_family(monkeypatch):
    """The attention / indexer operators never compile the top-k variants and vice versa; PREBUILD=0 stays lazy."""
    from miles_plugins.models.dsa_train import _jit, _kernels

    calls = []
    monkeypatch.setattr(_kernels, "device_arch", lambda device: "sm_100a")
    monkeypatch.setattr(_kernels, "prebuild", lambda arch, stages=None, max_workers=None: calls.append((arch, stages)))
    monkeypatch.setattr(_kernels, "kernel", lambda name, arch: (name, arch))
    monkeypatch.setattr(_kernels, "_PREBUILT", set())
    monkeypatch.delenv("MILES_DSA_TRAIN_PREBUILD", raising=False)
    assert _kernels._kernel("dsa_topk_fp32_c4096_k512", "cuda") == ("dsa_topk_fp32_c4096_k512", "sm_100a")
    assert len(calls) == 1 and calls[0][0] == "sm_100a" and set(calls[0][1]) == TOPK_STAGES
    _kernels._kernel("dsa_topk_fp32_c32768_k512", "cuda")
    assert len(calls) == 1, "a family is prebuilt once per architecture"
    _kernels._kernel("dsa_indexer_clean", "cuda")
    assert len(calls) == 2 and set(calls[1][1]) == set(_jit.MODULES) - TOPK_STAGES and len(calls[1][1]) == 17
    monkeypatch.setenv("MILES_DSA_TRAIN_PREBUILD", "0")
    monkeypatch.setattr(_kernels, "_PREBUILT", set())
    _kernels._kernel("dsa_topk_fp32_c4096_k512", "cuda")
    assert len(calls) == 2, "MILES_DSA_TRAIN_PREBUILD=0 builds lazily"
    # The real prebuild honours the stage filter (its loader is replaced, so nothing compiles here).
    loaded = []
    monkeypatch.setattr(_jit, "load", lambda stage, arch: loaded.append(stage))
    _jit.prebuild.__wrapped__("sm_100a", stages=("dsa_topk_fp32_c4096_k512", "dsa_indexer_clean"), max_workers=1)
    assert set(loaded) == {"dsa_topk_fp32_c4096_k512", "dsa_indexer_clean"}

import logging
from argparse import Namespace
from pathlib import Path

import pytest

from miles.utils.audit_utils.config_snapshot import check_config_snapshot, configure_config_snapshots
from miles.utils.audit_utils.process_identity import ProcessIdentity, SimpleProcessIdentity, TrainProcessIdentity
from miles.utils.test_utils.snapshot import SNAPSHOT_DIR, SNAPSHOT_RECORD_DIR_ENV_VAR, SNAPSHOT_UPDATE_ENV_VAR

_SNAPSHOT_DIR = SNAPSHOT_DIR / "_never_exists"


def _configure(source: ProcessIdentity) -> None:
    args = Namespace(
        ci_disable_config_snapshot=False,
        ci_test=True,
        config_snapshot_dir=str(_SNAPSHOT_DIR),
        config_snapshot_name="unit/run-0000",
        config_snapshot_normalize=None,
        deploy_component="all",
        deploy_instance_id=None,
        run_uuid="run-uuid",
    )
    configure_config_snapshots(args=args, source=source)


@pytest.fixture(autouse=True)
def _clean_snapshot_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(SNAPSHOT_UPDATE_ENV_VAR, raising=False)
    monkeypatch.delenv(SNAPSHOT_RECORD_DIR_ENV_VAR, raising=False)


@pytest.fixture
def record_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv(SNAPSHOT_RECORD_DIR_ENV_VAR, str(tmp_path))
    return tmp_path / "tests" / "snapshots" / "_never_exists" / "unit" / "run-0000" / "all" / "default"


def test_mismatch_without_record_dir_raises() -> None:
    """A mismatching boundary aborts the process when nobody collects records."""
    _configure(SimpleProcessIdentity(component="main"))

    with pytest.raises(AssertionError, match="runtime configuration main/process_config-0000"):
        check_config_snapshot(boundary="process_config", config={"args": {"a": 1}})


def test_mismatch_with_record_dir_logs_and_continues(record_dir: Path, caplog: pytest.LogCaptureFixture) -> None:
    """A mismatching boundary records the actual content and lets the run proceed."""
    _configure(SimpleProcessIdentity(component="main"))

    with caplog.at_level(logging.ERROR):
        check_config_snapshot(boundary="process_config", config={"args": {"a": 1}})
        check_config_snapshot(boundary="checkpoint_load", config={"args": {"a": 1}, "load": "ckpt"})

    assert sorted(path.name for path in (record_dir / "main").iterdir()) == [
        "checkpoint_load-0000.diff",
        "process_config-0000.yaml",
    ]
    assert "Recorded the actual content at" in caplog.text
    assert caplog.text.count("does not match its snapshot") == 2


def test_later_boundaries_are_stored_as_diffs_against_process_config(record_dir: Path) -> None:
    """Only the lines that changed since the process configuration reach the diff snapshot."""
    _configure(SimpleProcessIdentity(component="main"))

    check_config_snapshot(boundary="process_config", config={"args": {"a": 1, "b": 2}})
    check_config_snapshot(boundary="train_first_step", config={"args": {"a": 1, "b": 3}, "role": "actor"})

    diff = (record_dir / "main" / "train_first_step-0000.diff").read_text()
    assert diff.startswith("--- main/process_config-0000\n+++ main/train_first_step-0000\n")
    assert "-      b: 2\n+      b: 3\n" in diff
    assert "+    role: actor\n" in diff
    assert "-  boundary: process_config\n+  boundary: train_first_step\n" in diff
    assert "a: 1" not in diff


def test_the_first_boundary_must_be_the_process_configuration(record_dir: Path) -> None:
    """A diff needs its base, so a run cannot start snapshotting at a later boundary."""
    _configure(SimpleProcessIdentity(component="main"))

    with pytest.raises(ValueError, match="must be 'process_config'"):
        check_config_snapshot(boundary="checkpoint_load", config={"args": {"a": 1}})


def test_train_ranks_share_one_source_and_normalize_their_rank(record_dir: Path) -> None:
    """Ranks of one cell write the same file with every rank value replaced by a placeholder."""
    _configure(TrainProcessIdentity(component="actor", model_id="policy", cell_index=1, rank_within_cell=3))

    check_config_snapshot(boundary="process_config", config={"args": {"rank": 0}})
    check_config_snapshot(boundary="train_first_step", config={"args": {"rank": 11}, "rank": 3, "role": "actor"})

    base = (record_dir / "policy_actor_cell00001" / "process_config-0000.yaml").read_text()
    diff = (record_dir / "policy_actor_cell00001" / "train_first_step-0000.diff").read_text()
    assert "rank: $RANK\n" in base
    assert "rank: 0" not in base
    assert "rank: 11" not in diff and "rank: 3" not in diff
    assert "+    rank: $RANK\n" in diff
    assert "+    role: actor\n" in diff


def test_a_rank_that_disagrees_with_the_process_identity_is_rejected(record_dir: Path) -> None:
    """A boundary reporting a foreign rank is a bug, not a value to normalize away."""
    _configure(TrainProcessIdentity(component="actor", model_id=None, cell_index=0, rank_within_cell=3))
    check_config_snapshot(boundary="process_config", config={"args": {"rank": 0}})

    with pytest.raises(AssertionError, match="snapshot rank 4 differs from 3"):
        check_config_snapshot(boundary="train_first_step", config={"args": {"rank": 4}, "rank": 4})

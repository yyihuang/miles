import logging
from argparse import Namespace
from pathlib import Path

import pytest

from miles.utils.audit_utils.config_snapshot import check_config_snapshot, configure_config_snapshots
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity
from miles.utils.test_utils.snapshot import SNAPSHOT_DIR, SNAPSHOT_RECORD_DIR_ENV_VAR, SNAPSHOT_UPDATE_ENV_VAR

_SNAPSHOT_DIR = SNAPSHOT_DIR / "_never_exists"


@pytest.fixture(autouse=True)
def _configured_snapshots(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(SNAPSHOT_UPDATE_ENV_VAR, raising=False)
    monkeypatch.delenv(SNAPSHOT_RECORD_DIR_ENV_VAR, raising=False)
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
    configure_config_snapshots(args=args, source=SimpleProcessIdentity(component="main"))


def test_mismatch_without_record_dir_raises() -> None:
    """A mismatching boundary aborts the process when nobody collects records."""
    with pytest.raises(AssertionError, match="runtime configuration main/process_config-0000.yaml"):
        check_config_snapshot(boundary="process_config", config={"a": 1})


def test_mismatch_with_record_dir_logs_and_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A mismatching boundary records the actual content and lets the run proceed."""
    monkeypatch.setenv(SNAPSHOT_RECORD_DIR_ENV_VAR, str(tmp_path))

    with caplog.at_level(logging.ERROR):
        check_config_snapshot(boundary="process_config", config={"a": 1})
        check_config_snapshot(boundary="checkpoint_load", config={"a": 2})

    recorded_dir = tmp_path / "tests" / "snapshots" / "_never_exists" / "unit" / "run-0000" / "all" / "default"
    assert sorted(path.name for path in recorded_dir.rglob("*.yaml")) == [
        "checkpoint_load-0000.yaml",
        "process_config-0000.yaml",
    ]
    assert "Recorded the actual content at" in caplog.text
    assert caplog.text.count("does not match its snapshot") == 2

import logging
from pathlib import Path

import pytest

from tests.ci.ci_register import register_cpu_ci
from tests.ci.ci_utils import SNAPSHOT_RECORD_DIR_ENV
from tests.ci.ci_utils import TestFile as CITestFile
from tests.ci.ci_utils import run_unittest_files

register_cpu_ci(est_time=1, suite="stage-a-cpu", labels=[])

_RECORDING_TEST = (
    "import os, pathlib\n"
    "record = pathlib.Path(os.environ['MILES_SNAPSHOT_RECORD_DIR']) / 'tests' / 'snapshots' / 'x.yaml'\n"
    "record.parent.mkdir(parents=True)\n"
    "record.write_text('a: 1\\n')\n"
    "print('AssertionError: value not equal to expected')\n"
)


@pytest.fixture
def suite_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(SNAPSHOT_RECORD_DIR_ENV, str(tmp_path / "records"))
    return tmp_path


def test_a_clean_run_with_no_records_passes(suite_dir: Path) -> None:
    """A test that records nothing keeps passing under a record dir."""
    (suite_dir / "test_clean.py").write_text("print('ok')\n")

    assert run_unittest_files([CITestFile(name="test_clean.py")], timeout_per_file=60) == 0


def test_a_finished_run_with_records_fails_without_retry(suite_dir: Path, caplog: pytest.LogCaptureFixture) -> None:
    """A test that exits cleanly but recorded mismatches is reported as failed and never retried."""
    (suite_dir / "test_mismatch.py").write_text(_RECORDING_TEST)

    with caplog.at_level(logging.INFO):
        result = run_unittest_files(
            [CITestFile(name="test_mismatch.py")], timeout_per_file=60, enable_retry=True, max_attempts=2
        )

    assert result == -1
    assert "SNAPSHOT MISMATCH: test_mismatch.py recorded 1 file(s)" in caplog.text
    assert "test_mismatch.py (1 snapshot mismatch(es))" in caplog.text
    assert "tests/snapshots/x.yaml" in caplog.text
    assert "[CI Retry]" not in caplog.text


def test_records_are_kept_per_test_and_attempt(suite_dir: Path) -> None:
    """Each test's records land in their own attempt directory under the job record dir."""
    (suite_dir / "test_mismatch.py").write_text(_RECORDING_TEST)

    run_unittest_files([CITestFile(name="test_mismatch.py")], timeout_per_file=60)

    recorded = list((suite_dir / "records").rglob("x.yaml"))
    assert len(recorded) == 1
    assert recorded[0].parent.parent.parent.name == "attempt-1"


def test_mismatches_are_written_to_the_step_summary(suite_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The GitHub step summary lists every recorded snapshot per test."""
    summary = suite_dir / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    (suite_dir / "test_mismatch.py").write_text(_RECORDING_TEST)

    run_unittest_files([CITestFile(name="test_mismatch.py")], timeout_per_file=60)

    assert "**Snapshot mismatches in 1 test(s):**" in summary.read_text()
    assert "`tests/snapshots/x.yaml`" in summary.read_text()

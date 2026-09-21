from pathlib import Path

import pytest

from miles.utils.test_utils.snapshot import (
    SNAPSHOT_DIR,
    SNAPSHOT_RECORD_DIR_ENV_VAR,
    SNAPSHOT_UPDATE_ENV_VAR,
    _record_snapshot_mismatch,
    assert_matches_snapshot,
    compare_snapshot,
)

_MISSING_SNAPSHOT = SNAPSHOT_DIR / "_never_exists" / "case.yaml"


@pytest.fixture(autouse=True)
def _clean_snapshot_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(SNAPSHOT_UPDATE_ENV_VAR, raising=False)
    monkeypatch.delenv(SNAPSHOT_RECORD_DIR_ENV_VAR, raising=False)


def test_missing_baseline_without_record_dir_reports_actual_content_inline() -> None:
    """Without a record dir the mismatch message carries the actual content and nothing is written."""
    mismatch = compare_snapshot(snapshot=_MISSING_SNAPSHOT, actual="a: 1\n", subject="case")

    assert mismatch is not None
    assert mismatch.recorded is None
    assert "--- BEGIN ACTUAL case.yaml ---\na: 1\n--- END ACTUAL ---" in mismatch.message
    assert not _MISSING_SNAPSHOT.exists()


def test_record_dir_stores_actual_at_repo_relative_path_and_names_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With a record dir the actual content lands under the snapshot's repo-relative path."""
    monkeypatch.setenv(SNAPSHOT_RECORD_DIR_ENV_VAR, str(tmp_path))

    mismatch = compare_snapshot(snapshot=_MISSING_SNAPSHOT, actual="a: 1\n", subject="case")

    recorded = tmp_path / "tests" / "snapshots" / "_never_exists" / "case.yaml"
    assert mismatch is not None
    assert mismatch.recorded == recorded
    assert recorded.read_text() == "a: 1\n"
    assert f"Recorded the actual content at {recorded}" in mismatch.message
    assert "BEGIN ACTUAL" not in mismatch.message


def test_assert_matches_snapshot_still_raises_when_recording(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Recording keeps the assertion failing so in-process tests report the mismatch."""
    monkeypatch.setenv(SNAPSHOT_RECORD_DIR_ENV_VAR, str(tmp_path))

    with pytest.raises(AssertionError, match="Recorded the actual content"):
        assert_matches_snapshot(_MISSING_SNAPSHOT, "a: 1\n", "case")


def test_matching_snapshot_records_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An up-to-date baseline produces no mismatch and no record."""
    monkeypatch.setenv(SNAPSHOT_RECORD_DIR_ENV_VAR, str(tmp_path / "records"))
    snapshot = tmp_path / "case.yaml"
    snapshot.write_text("a: 1\n")

    assert compare_snapshot(snapshot=snapshot, actual="a: 1\n", subject="case") is None
    assert not (tmp_path / "records").exists()


def test_update_mode_writes_baseline_instead_of_recording(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Update mode rewrites the baseline file even when a record dir is configured."""
    monkeypatch.setenv(SNAPSHOT_UPDATE_ENV_VAR, "1")
    monkeypatch.setenv(SNAPSHOT_RECORD_DIR_ENV_VAR, str(tmp_path / "records"))
    snapshot = tmp_path / "baseline" / "case.yaml"

    assert compare_snapshot(snapshot=snapshot, actual="a: 1\n", subject="case") is None
    assert snapshot.read_text() == "a: 1\n"
    assert not (tmp_path / "records").exists()


def test_identical_records_share_one_file_and_differing_records_become_conflicts(tmp_path: Path) -> None:
    """Concurrent writers with equal content converge; a different content is kept as a conflict file."""
    first = _record_snapshot_mismatch(record_dir=tmp_path, snapshot=_MISSING_SNAPSHOT, actual="a: 1\n")
    second = _record_snapshot_mismatch(record_dir=tmp_path, snapshot=_MISSING_SNAPSHOT, actual="a: 1\n")
    conflict = _record_snapshot_mismatch(record_dir=tmp_path, snapshot=_MISSING_SNAPSHOT, actual="a: 2\n")

    assert first == second
    assert first.read_text() == "a: 1\n"
    assert conflict.parent == first.parent
    assert conflict.name.startswith("case.yaml.conflict-")
    assert conflict.read_text() == "a: 2\n"
    assert sorted(path.name for path in first.parent.iterdir()) == sorted([first.name, conflict.name])


def test_snapshot_outside_the_repo_cannot_be_recorded(tmp_path: Path) -> None:
    """Recording refuses snapshots that the sync back into the repo could not place."""
    with pytest.raises(ValueError, match="must live under"):
        _record_snapshot_mismatch(record_dir=tmp_path, snapshot=tmp_path / "case.yaml", actual="a: 1\n")

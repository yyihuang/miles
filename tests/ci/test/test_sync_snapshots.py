from pathlib import Path

import pytest

from tests.ci.ci_register import register_cpu_ci
from tests.ci.sync_snapshots import apply, collect_records

register_cpu_ci(est_time=1, suite="stage-a-cpu", labels=[])


def _record(directory: Path, artifact: str, relative: str, content: str) -> Path:
    path = directory / artifact / "tests_e2e_test_x.py-0123456789" / "attempt-1" / "tests" / "snapshots" / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def test_records_are_keyed_by_their_repo_relative_snapshot_path(tmp_path: Path) -> None:
    """Each recorded file maps to the tests/snapshots path it came from."""
    _record(tmp_path, "snapshot-records-stage-a", "runtime_config/x/main/process_config-0000.yaml", "a: 1\n")
    (tmp_path / "snapshot-records-stage-a" / "unrelated.txt").write_text("ignored")

    records = collect_records(tmp_path)

    assert [record.target for record in records] == [
        Path("tests/snapshots/runtime_config/x/main/process_config-0000.yaml")
    ]
    assert records[0].content == "a: 1\n"


def test_identical_records_from_two_jobs_merge_and_differing_ones_are_rejected(tmp_path: Path) -> None:
    """The same snapshot recorded by two jobs must agree byte for byte."""
    _record(tmp_path, "snapshot-records-stage-a", "runtime_config/y.yaml", "a: 1\n")
    _record(tmp_path, "snapshot-records-stage-b", "runtime_config/y.yaml", "a: 1\n")
    assert len(collect_records(tmp_path)) == 1

    _record(tmp_path, "snapshot-records-stage-c", "runtime_config/y.yaml", "a: 2\n")
    with pytest.raises(ValueError, match="recorded with different content"):
        collect_records(tmp_path)


def test_process_conflict_files_abort_the_sync(tmp_path: Path) -> None:
    """A conflict file left by disagreeing ranks is surfaced instead of silently picked."""
    _record(tmp_path, "snapshot-records-stage-a", "runtime_config/z.yaml", "a: 1\n")
    _record(tmp_path, "snapshot-records-stage-a", "runtime_config/z.yaml.conflict-abcd", "a: 2\n")

    with pytest.raises(ValueError, match="disagreed on these snapshots"):
        collect_records(tmp_path)


def test_temporary_files_are_skipped(tmp_path: Path) -> None:
    """Leftover temp files from an interrupted writer are not snapshots."""
    _record(tmp_path, "snapshot-records-stage-a", ".z.yaml.abcd.tmp", "partial")

    assert collect_records(tmp_path) == []


def test_apply_writes_records_into_the_repository_unless_dry_run(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """Apply materializes each record at its snapshot path, and dry-run only prints the plan."""
    downloads = tmp_path / "downloads"
    repo = tmp_path / "repo"
    _record(downloads, "snapshot-records-stage-a", "runtime_config/w.yaml", "a: 1\n")

    apply(directory=downloads, repo_root=repo, dry_run=True)
    assert not (repo / "tests" / "snapshots" / "runtime_config" / "w.yaml").exists()
    assert "PLAN:" in capsys.readouterr().out

    apply(directory=downloads, repo_root=repo, dry_run=False)
    assert (repo / "tests" / "snapshots" / "runtime_config" / "w.yaml").read_text() == "a: 1\n"

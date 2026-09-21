import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import typer

from miles.utils.external_utils.command_utils.common import repo_base_dir, run_process
from miles.utils.test_utils.snapshot import SNAPSHOT_CONFLICT_MARKER, SNAPSHOT_DIR, list_recorded_snapshots

app = typer.Typer(add_completion=False)

_SNAPSHOT_ROOT_PARTS = SNAPSHOT_DIR.relative_to(repo_base_dir).parts
_ARTIFACT_PATTERN = "snapshot-records-*"


@dataclass(frozen=True)
class SnapshotRecord:
    target: Path
    content: str
    source: Path


@app.command(help="Download every snapshot record artifact of one workflow run")
def download(
    run_id: Annotated[str, typer.Option(help="GitHub Actions run id")],
    directory: Annotated[Path, typer.Option(help="Empty directory that receives one subdirectory per artifact")],
    repo: Annotated[str, typer.Option(help="GitHub repository")] = "radixark/miles",
) -> None:
    run_process(
        argv=[
            "gh",
            "run",
            "download",
            run_id,
            "--repo",
            repo,
            "--pattern",
            _ARTIFACT_PATTERN,
            "--dir",
            str(directory),
        ],
        capture_output=False,
        check=True,
    )


@app.command(help="Write the downloaded records into tests/snapshots")
def apply(
    directory: Annotated[Path, typer.Option(help="Directory filled by the download command")],
    repo_root: Annotated[Path, typer.Option(help="Repository that receives the snapshots")] = repo_base_dir,
    dry_run: Annotated[bool, typer.Option(help="Only print what would be written")] = False,
) -> None:
    records = collect_records(directory)
    for record in records:
        destination = repo_root / record.target
        print(f"{'PLAN' if dry_run else 'WRITE'}: {record.source} -> {destination}", flush=True)
        if dry_run:
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(record.content)
    print(f"{len(records)} snapshot file(s) {'planned' if dry_run else 'written'}", flush=True)


def collect_records(directory: Path) -> list[SnapshotRecord]:
    records: dict[Path, SnapshotRecord] = {}
    conflicts: list[Path] = []
    for path in list_recorded_snapshots(directory):
        parts = path.relative_to(directory).parts
        if (index := _snapshot_root_index(parts)) is None:
            continue
        if SNAPSHOT_CONFLICT_MARKER in path.name:
            conflicts.append(path)
            continue

        record = SnapshotRecord(target=Path(*parts[index:]), content=path.read_text(), source=path)
        if (existing := records.get(record.target)) is not None and existing.content != record.content:
            raise ValueError(f"{record.target} was recorded with different content by {existing.source} and {path}")
        records.setdefault(record.target, record)

    if conflicts:
        raise ValueError("Processes disagreed on these snapshots:\n" + "\n".join(str(path) for path in conflicts))
    return list(records.values())


def _snapshot_root_index(parts: tuple[str, ...]) -> int | None:
    return next(
        (index for index in range(len(parts) - 1) if parts[index : index + 2] == _SNAPSHOT_ROOT_PARTS),
        None,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    app()

"""Memory-bounded access to the pinned LiveCodeBench v6 JSONL source files."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator


DATASET_REPOSITORY = "livecodebench/code_generation_lite"
RELEASE_FILES = {
    "release_v6": (
        "test.jsonl",
        "test2.jsonl",
        "test3.jsonl",
        "test4.jsonl",
        "test5.jsonl",
        "test6.jsonl",
    )
}


def resolve_snapshot_dir(cache_root: Path | None = None) -> Path:
    """Find the immutable Hugging Face snapshot downloaded by the official loader."""
    cache_root = cache_root or Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser()
    repository_root = cache_root / "hub" / "datasets--livecodebench--code_generation_lite"
    revision = (repository_root / "refs" / "main").read_text(encoding="utf-8").strip()
    snapshot = repository_root / "snapshots" / revision
    if not snapshot.is_dir():
        raise FileNotFoundError(f"LiveCodeBench snapshot not found: {snapshot}")
    return snapshot


def iter_filtered_records(
    *,
    release_version: str,
    start_date: str,
    end_date: str,
    snapshot_dir: Path | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield official raw rows without materializing private tests in memory."""
    files = RELEASE_FILES.get(release_version)
    if files is None:
        raise ValueError(f"unsupported release version: {release_version}")
    lower = datetime.strptime(start_date, "%Y-%m-%d")
    upper = datetime.strptime(end_date, "%Y-%m-%d")
    source_root = snapshot_dir or resolve_snapshot_dir()
    for name in files:
        path = source_root / name
        if not path.is_file():
            raise FileNotFoundError(f"missing official source file: {path}")
        with path.open("rb") as handle:
            for raw_line in handle:
                row = json.loads(raw_line)
                contest_date = datetime.fromisoformat(str(row["contest_date"]))
                if lower <= contest_date <= upper:
                    yield row

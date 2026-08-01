"""Shared provenance and dataset helpers for one-pass benchmark runners."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import aiohttp


ROOT = Path(__file__).resolve().parents[1]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def endpoint_root(chat_url: str) -> str:
    parsed = urlsplit(chat_url)
    path = parsed.path
    suffix = "/v1/chat/completions"
    if path.endswith(suffix):
        path = path[: -len(suffix)]
    return urlunsplit((parsed.scheme, parsed.netloc, path.rstrip("/"), "", ""))


async def get_json(session: aiohttp.ClientSession, url: str) -> dict[str, Any]:
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as response:
            raw = await response.text()
            if response.status != 200:
                return {"ok": False, "http_status": response.status, "body": raw[:500]}
            return {"ok": True, "response": json.loads(raw)}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def _run_command(*command: str) -> str:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else ""


def git_snapshot() -> dict[str, Any]:
    status = _run_command("git", "status", "--porcelain=v1")
    return {
        "commit": _run_command("git", "rev-parse", "HEAD") or None,
        "branch": _run_command("git", "branch", "--show-current") or None,
        "dirty": bool(status),
        "status": status.splitlines(),
    }


def file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_versions(names: list[str]) -> dict[str, str | None]:
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def machine_snapshot() -> dict[str, Any]:
    memory_bytes = None
    try:
        memory_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        pass
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": sys.version,
        "logical_cpu_count": os.cpu_count(),
        "memory_bytes": memory_bytes,
        "runtime_packages": _package_versions(["aiohttp", "evalplus", "numpy"]),
    }


def load_tasks(args: argparse.Namespace) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    from evalplus.data import get_human_eval_plus, get_human_eval_plus_hash
    from evalplus.data.humaneval import HUMANEVAL_PLUS_VERSION

    all_problems = get_human_eval_plus(version=HUMANEVAL_PLUS_VERSION)
    dataset_hash = get_human_eval_plus_hash(version=HUMANEVAL_PLUS_VERSION)
    if args.task_ids:
        task_ids = [item.strip() for item in args.task_ids.split(",") if item.strip()]
    else:
        task_ids = list(all_problems)[args.task_start : args.task_start + args.task_count]
    missing = [task_id for task_id in task_ids if task_id not in all_problems]
    if missing:
        raise ValueError(f"unknown HumanEval+ task IDs: {missing}")
    problems = {task_id: all_problems[task_id] for task_id in task_ids}
    return problems, {
        "name": "HumanEval+",
        "split": "test",
        "version": HUMANEVAL_PLUS_VERSION,
        "content_md5": dataset_hash,
        "source": (
            "https://github.com/evalplus/humanevalplus_release/releases/download/"
            f"{HUMANEVAL_PLUS_VERSION}/HumanEvalPlus.jsonl.gz"
        ),
        "task_ids": task_ids,
        "task_order": "dataset order",
    }


def parse_seeds(value: str) -> list[int]:
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return seeds

#!/usr/bin/env python3
"""Create a decoder-safe LiveCodeBench Code Generation task manifest.

This trusted preparation step is the only step that loads the official dataset
before scoring.  It intentionally serializes task text and prompt metadata
only: no public-test objects, private tests, evaluator samples, or oracle data
are written to the manifest consumed by the model caller.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path
from typing import Any

from lcb_runner.benchmarks.code_generation import load_code_generation_dataset
from lcb_runner.prompts.code_generation import (
    PromptConstants,
    get_generic_question_template_answer,
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def git_revision() -> str:
    completed = subprocess.run(
        ("git", "rev-parse", "HEAD"), text=True, capture_output=True, check=True
    )
    return completed.stdout.strip()


def decoder_task(problem: Any) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": PromptConstants.SYSTEM_MESSAGE_GENERIC},
        {"role": "user", "content": get_generic_question_template_answer(problem)},
    ]
    return {
        "task_id": str(problem.question_id),
        "contest_date": problem.contest_date.date().isoformat(),
        "platform": problem.platform.value,
        "difficulty": problem.difficulty.value,
        "prompt_messages": messages,
        "prompt_sha256": sha256_json(messages),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--release-version", default="release_v6")
    parser.add_argument("--start-date", default="2024-01-01")
    parser.add_argument("--end-date", default="2025-04-30")
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite manifest: {args.output}")
    problems = load_code_generation_dataset(
        args.release_version, start_date=args.start_date, end_date=args.end_date
    )
    tasks = sorted((decoder_task(problem) for problem in problems), key=lambda task: task["task_id"])
    if not tasks:
        raise RuntimeError("official LiveCodeBench filter returned no tasks")
    if len({task["task_id"] for task in tasks}) != len(tasks):
        raise RuntimeError("official LiveCodeBench filter returned duplicate task IDs")
    serialized_tasks = canonical_json(tasks)
    prohibited = ("private_test", "input_output", "public_test_cases")
    if any(marker in serialized_tasks for marker in prohibited):
        raise RuntimeError("decoder manifest unexpectedly contains evaluator data")

    manifest = {
        "schema_version": 1,
        "benchmark": "LiveCodeBench Code Generation",
        "official_source": {
            "repository": "https://github.com/LiveCodeBench/LiveCodeBench",
            "revision": git_revision(),
            "dataset": "livecodebench/code_generation_lite",
            "release_version": args.release_version,
            "datasets_version": importlib.metadata.version("datasets"),
        },
        "selection": {
            "start_date": args.start_date,
            "end_date": args.end_date,
            "reason": "strictly after the December 2023 Llama 3.1 training-data cutoff",
        },
        "decoder_input_contract": {
            "contains_hidden_tests": False,
            "contains_public_test_objects": False,
            "contains_evaluator_samples": False,
            "prompt_source": "official LiveCodeBench generic LLaMa3 prompt messages",
        },
        "task_count": len(tasks),
        "tasks_sha256": sha256_json(tasks),
        "tasks": tasks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"task_count": len(tasks), "tasks_sha256": manifest["tasks_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

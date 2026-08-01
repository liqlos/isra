#!/usr/bin/env python3
"""Score a completed LiveCodeBench paired run with the pinned official checker.

Run this only in the LiveCodeBench evaluator image.  Unlike the paired decoder
runner, this process loads official test data and executes generated programs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.harness_common import file_sha256, utc_now  # noqa: E402
from benchmarks.paired_core import read_jsonl, sha256_text, write_json_exclusive  # noqa: E402


def load_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != 1 or value.get("decoder_input_contract", {}).get("contains_hidden_tests") is not False:
        raise ValueError("not a safe LiveCodeBench decoder manifest")
    return value


def index_results(records: list[dict[str, Any]], task_ids: list[str]) -> dict[str, list[dict[str, Any]]]:
    expected = set(task_ids)
    grouped: dict[str, list[dict[str, Any]]] = {"direct_greedy": [], "spa_greedy": []}
    for variant in grouped:
        rows = sorted((row for row in records if row.get("variant") == variant), key=lambda row: str(row.get("task_id")))
        if len(rows) != len(task_ids):
            raise ValueError(f"{variant} has {len(rows)} records, expected {len(task_ids)}")
        if [str(row.get("task_id")) for row in rows] != task_ids:
            raise ValueError(f"{variant} task IDs do not exactly match the decoder manifest")
        if any(int(row.get("seed", -1)) != 0 for row in rows):
            raise ValueError("official pass@1 scorer currently requires exactly seed 0")
        grouped[variant] = rows
    unexpected = {str(row.get("task_id")) for row in records} - expected
    if unexpected:
        raise ValueError(f"results contain tasks absent from decoder manifest: {sorted(unexpected)[:3]}")
    return grouped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-manifest", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluation-metadata", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=6)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--task-limit", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists() or args.evaluation_metadata.exists():
        raise FileExistsError("refusing to overwrite an existing scored artifact")

    # Imports occur only in this evaluator process, never in the decoder runner.
    os.chdir("/opt/livecodebench")
    from lcb_runner.benchmarks.code_generation import load_code_generation_dataset
    from lcb_runner.evaluation import codegen_metrics, extract_instance_results
    from lcb_runner.lm_styles import LMStyle
    from lcb_runner.utils.extraction_utils import extract_code

    task_manifest = load_manifest(args.task_manifest)
    if args.task_limit < 0:
        raise ValueError("task-limit cannot be negative")
    selected_tasks = task_manifest["tasks"][: args.task_limit or None]
    task_ids = [str(task["task_id"]) for task in selected_tasks]
    records = read_jsonl(args.results)
    by_variant = index_results(records, task_ids)
    source = task_manifest["official_source"]
    selection = task_manifest["selection"]
    benchmark = sorted(load_code_generation_dataset(source["release_version"], start_date=selection["start_date"], end_date=selection["end_date"]), key=lambda problem: problem.question_id)
    benchmark = [problem for problem in benchmark if str(problem.question_id) in set(task_ids)]
    if [str(problem.question_id) for problem in benchmark] != task_ids:
        raise RuntimeError("official evaluator task set differs from frozen decoder manifest")
    samples = [problem.get_evaluation_sample() for problem in benchmark]

    scored: list[dict[str, Any]] = []
    score_metadata: dict[str, Any] = {"scored_at": utc_now(), "evaluator": "LiveCodeBench official codegen_metrics", "official_source": source, "selection": selection, "timeout_seconds": args.timeout, "workers": args.workers, "variants": {}}
    for variant, rows in by_variant.items():
        generations = [[extract_code(str(row.get("response", "")), LMStyle.LLaMa3) if row.get("status") == "completed" else ""] for row in rows]
        metrics, raw_grades, metadata = codegen_metrics(samples, generations, k_list=[1], num_process_evaluate=args.workers, timeout=args.timeout, debug=False)
        grades = extract_instance_results(raw_grades)
        if len(grades) != len(rows):
            raise RuntimeError("official evaluator returned an unexpected grade count")
        score_metadata["variants"][variant] = {"pass_at_1": metrics.get("pass@1"), "grader_metadata": metadata}
        for row, generated, grade in zip(rows, generations, grades):
            copy = dict(row)
            if copy.get("status") == "completed":
                copy["passed"] = bool(grade[0])
                copy["evaluator_message"] = "official LiveCodeBench codegen_metrics"
                copy["extracted_code_sha256"] = sha256_text(generated[0])
            scored.append(copy)
    scored.sort(key=lambda row: (str(row["task_id"]), str(row["variant"])))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        for row in scored:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    score_metadata["input_results_sha256"] = file_sha256(args.results)
    score_metadata["scored_results_sha256"] = file_sha256(args.output)
    write_json_exclusive(args.evaluation_metadata, score_metadata)
    print(json.dumps({variant: details["pass_at_1"] for variant, details in score_metadata["variants"].items()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

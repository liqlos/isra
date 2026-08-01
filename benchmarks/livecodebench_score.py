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
from benchmarks.livecodebench_data import iter_filtered_records  # noqa: E402
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


def batches(values: Any, size: int) -> Any:
    batch = []
    for value in values:
        batch.append(value)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-manifest", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluation-metadata", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=6)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--task-limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists() or args.evaluation_metadata.exists():
        raise FileExistsError("refusing to overwrite an existing scored artifact")

    # Imports occur only in this evaluator process, never in the decoder runner.
    os.chdir("/opt/livecodebench")
    from lcb_runner.benchmarks.code_generation import CodeGenerationProblem
    from lcb_runner.evaluation import codegen_metrics, extract_instance_results
    from lcb_runner.lm_styles import LMStyle
    from lcb_runner.utils.extraction_utils import extract_code

    task_manifest = load_manifest(args.task_manifest)
    if args.task_limit < 0 or args.batch_size <= 0:
        raise ValueError("task-limit cannot be negative and batch-size must be positive")
    selected_tasks = task_manifest["tasks"][: args.task_limit or None]
    task_ids = [str(task["task_id"]) for task in selected_tasks]
    records = read_jsonl(args.results)
    by_variant = index_results(records, task_ids)
    source = task_manifest["official_source"]
    selection = task_manifest["selection"]
    requested = set(task_ids)
    rows_by_id = {
        variant: {str(row["task_id"]): row for row in rows}
        for variant, rows in by_variant.items()
    }
    outcomes: dict[tuple[str, str], tuple[bool, str]] = {}
    seen: set[str] = set()
    passed_counts = {variant: 0 for variant in by_variant}
    evaluated_counts = {variant: 0 for variant in by_variant}
    official_rows = (
        row
        for row in iter_filtered_records(
            release_version=source["release_version"],
            start_date=selection["start_date"],
            end_date=selection["end_date"],
        )
        if str(row["question_id"]) in requested
    )
    for raw_batch in batches(official_rows, args.batch_size):
        ids = [str(row["question_id"]) for row in raw_batch]
        if len(set(ids)) != len(ids) or any(task_id in seen for task_id in ids):
            raise RuntimeError("official evaluator source has duplicate selected task IDs")
        seen.update(ids)
        problems = [CodeGenerationProblem(**row) for row in raw_batch]
        samples = [problem.get_evaluation_sample() for problem in problems]
        for variant in by_variant:
            active = [index for index, task_id in enumerate(ids) if rows_by_id[variant][task_id].get("status") == "completed"]
            if not active:
                continue
            active_samples = [samples[index] for index in active]
            generated = [
                [extract_code(str(rows_by_id[variant][ids[index]].get("response", "")), LMStyle.LLaMa3)]
                for index in active
            ]
            _, raw_grades, _ = codegen_metrics(
                active_samples,
                generated,
                k_list=[1],
                num_process_evaluate=args.workers,
                timeout=args.timeout,
                debug=False,
            )
            grades = extract_instance_results(raw_grades)
            if len(grades) != len(active):
                raise RuntimeError("official evaluator returned an unexpected grade count")
            for index, generated_code, grade in zip(active, generated, grades):
                task_id = ids[index]
                passed = bool(grade[0])
                outcomes[(variant, task_id)] = (passed, generated_code[0])
                passed_counts[variant] += int(passed)
                evaluated_counts[variant] += 1
        del samples, problems
    if seen != requested:
        missing = sorted(requested - seen)
        unexpected = sorted(seen - requested)
        raise RuntimeError(f"official evaluator task set differs from frozen decoder manifest: missing={missing[:3]}, unexpected={unexpected[:3]}")
    for variant, rows in by_variant.items():
        expected_completed = {str(row["task_id"]) for row in rows if row.get("status") == "completed"}
        actual = {task_id for current_variant, task_id in outcomes if current_variant == variant}
        if actual != expected_completed:
            raise RuntimeError(f"official evaluator did not grade every completed {variant} result")

    scored: list[dict[str, Any]] = []
    score_metadata: dict[str, Any] = {
        "scored_at": utc_now(),
        "evaluator": "LiveCodeBench official codegen_metrics",
        "official_source": source,
        "selection": selection,
        "timeout_seconds": args.timeout,
        "workers": args.workers,
        "batch_size": args.batch_size,
        "variants": {},
    }
    for variant, rows in by_variant.items():
        score_metadata["variants"][variant] = {
            "completed_records_evaluated": evaluated_counts[variant],
            "completed_pass_at_1": passed_counts[variant] / evaluated_counts[variant] if evaluated_counts[variant] else None,
            "pass_at_1_with_noncompleted_as_failures": passed_counts[variant] / len(task_ids),
        }
        for row in rows:
            copy = dict(row)
            outcome = outcomes.get((variant, str(copy["task_id"])))
            if outcome is not None:
                copy["passed"], generated_code = outcome
                copy["evaluator_message"] = "official LiveCodeBench codegen_metrics"
                copy["extracted_code_sha256"] = sha256_text(generated_code)
            scored.append(copy)
    scored.sort(key=lambda row: (str(row["task_id"]), str(row["variant"])))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        for row in scored:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    score_metadata["input_results_sha256"] = file_sha256(args.results)
    score_metadata["scored_results_sha256"] = file_sha256(args.output)
    write_json_exclusive(args.evaluation_metadata, score_metadata)
    print(json.dumps(score_metadata["variants"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

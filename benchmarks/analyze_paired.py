#!/usr/bin/env python3
"""Analyze paired benchmark JSONL without treating infrastructure failures as wrong answers."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.paired_core import (  # noqa: E402
    exact_mcnemar_p,
    paired_bootstrap_delta,
    percentile,
    read_jsonl,
)


def metric_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [record for record in records if record.get("status") == "completed"]
    latency = [float(record["latency_ms"]) for record in completed]
    return {
        "completed_records": len(completed),
        "median_latency_ms": statistics.median(latency) if latency else None,
        "p95_latency_ms": percentile(latency, 0.95),
        "mean_prompt_tokens": (
            statistics.fmean(float(record.get("prompt_tokens", 0)) for record in completed)
            if completed
            else None
        ),
        "mean_completion_tokens": (
            statistics.fmean(float(record.get("completion_tokens", 0)) for record in completed)
            if completed
            else None
        ),
        "mean_llm_calls": (
            statistics.fmean(float(record.get("llm_calls", 0)) for record in completed)
            if completed
            else None
        ),
    }


def analyze(
    records: list[dict[str, Any]],
    direct_variant: str,
    isra_variant: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    selected = [
        record for record in records if record.get("variant") in {direct_variant, isra_variant}
    ]
    by_variant = {
        variant: [record for record in selected if record.get("variant") == variant]
        for variant in (direct_variant, isra_variant)
    }
    indexes: dict[str, dict[tuple[str, int, str], dict[str, Any]]] = {}
    for variant, variant_records in by_variant.items():
        index = {}
        for record in variant_records:
            key = (str(record["task_id"]), int(record["seed"]), str(record["model"]))
            if key in index:
                raise ValueError(f"duplicate pair member for {variant}: {key}")
            index[key] = record
        indexes[variant] = index

    all_keys = sorted(set(indexes[direct_variant]) | set(indexes[isra_variant]))
    completed_pairs: list[tuple[bool, bool]] = []
    pair_rows = []
    excluded = []
    disagreements = []
    for key in all_keys:
        direct = indexes[direct_variant].get(key)
        isra = indexes[isra_variant].get(key)
        if not direct or not isra:
            excluded.append({"key": key, "reason": "missing_pair_member"})
            continue
        if direct.get("status") != "completed" or isra.get("status") != "completed":
            excluded.append(
                {
                    "key": key,
                    "reason": "non_completed_status",
                    "direct_status": direct.get("status"),
                    "isra_status": isra.get("status"),
                }
            )
            continue
        direct_pass = bool(direct.get("passed"))
        isra_pass = bool(isra.get("passed"))
        completed_pairs.append((direct_pass, isra_pass))
        row = {
            "task_id": key[0],
            "seed": key[1],
            "model": key[2],
            "direct_passed": direct_pass,
            "isra_passed": isra_pass,
        }
        pair_rows.append(row)
        if direct_pass != isra_pass:
            disagreements.append(
                {
                    **row,
                    "provisional_outcome": "fix" if isra_pass else "regression",
                    "manual_classification": None,
                    "manual_notes": None,
                    "direct_response": direct.get("response", ""),
                    "isra_response": isra.get("response", ""),
                    "direct_evaluator_message": direct.get("evaluator_message"),
                    "isra_evaluator_message": isra.get("evaluator_message"),
                }
            )

    both_pass = sum(direct and isra for direct, isra in completed_pairs)
    direct_only = sum(direct and not isra for direct, isra in completed_pairs)
    isra_only = sum(not direct and isra for direct, isra in completed_pairs)
    both_fail = sum(not direct and not isra for direct, isra in completed_pairs)
    n = len(completed_pairs)
    delta = sum(int(isra) - int(direct) for direct, isra in completed_pairs) / n if n else None
    ci = (
        paired_bootstrap_delta(
            completed_pairs, samples=bootstrap_samples, seed=bootstrap_seed
        )
        if completed_pairs
        else (None, None)
    )
    summary = {
        "direct_variant": direct_variant,
        "isra_variant": isra_variant,
        "pair_count": n,
        "outcome_table": {
            "both_pass": both_pass,
            "direct_only_regressions": direct_only,
            "isra_only_fixes": isra_only,
            "both_fail": both_fail,
        },
        "direct_accuracy": (
            sum(direct for direct, _ in completed_pairs) / n if n else None
        ),
        "isra_accuracy": sum(isra for _, isra in completed_pairs) / n if n else None,
        "paired_accuracy_delta": delta,
        "paired_bootstrap_95pct_ci": list(ci),
        "exact_mcnemar_two_sided_p": exact_mcnemar_p(isra_only, direct_only),
        "status_counts": {
            variant: dict(Counter(record.get("status") for record in variant_records))
            for variant, variant_records in by_variant.items()
        },
        "excluded_pairs": excluded,
        "cost_latency": {
            variant: metric_summary(variant_records)
            for variant, variant_records in by_variant.items()
        },
        "disagreement_count": len(disagreements),
        "interpretation_guardrail": (
            "A 20-task smoke set validates the harness and supports manual diagnosis; "
            "it is not a confirmatory efficacy result."
        ),
    }
    return summary, disagreements


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--direct-variant", default="direct_greedy")
    parser.add_argument("--isra-variant", default="isra_greedy")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--disagreements", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    records = read_jsonl(args.results)
    summary, disagreements = analyze(
        records,
        args.direct_variant,
        args.isra_variant,
        args.bootstrap_samples,
        args.bootstrap_seed,
    )
    output = args.output or args.results.with_name(
        f"analysis-{args.direct_variant}-vs-{args.isra_variant}.json"
    )
    disagreement_path = args.disagreements or args.results.with_name(
        f"disagreements-{args.direct_variant}-vs-{args.isra_variant}.jsonl"
    )
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    disagreement_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in disagreements),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"Analysis: {output}")
    print(f"Disagreements: {disagreement_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

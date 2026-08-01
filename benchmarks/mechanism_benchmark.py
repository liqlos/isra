#!/usr/bin/env python3
"""Frozen-candidate benchmark for attributing ISRA code-repair mechanisms."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import random
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiohttp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.paired_benchmark import (  # noqa: E402
    endpoint_root,
    file_sha256,
    get_json,
    git_snapshot,
    load_tasks,
    machine_snapshot,
    orchestrator_snapshot,
    parse_seeds,
    utc_now,
)
from benchmarks.paired_core import (  # noqa: E402
    RESULT_SCHEMA_VERSION,
    EvalPlusHumanEvalEvaluator,
    ResultStore,
    call_openai_endpoint,
    primary_key,
    sha256_json,
    sha256_text,
    task_messages,
    write_json_exclusive,
)


PARENT_VARIANT = "phase1_only"
BRANCH_VARIANTS = ("phase1_unguided_retry", "grounded_one_repair")
ALL_VARIANTS = (PARENT_VARIANT,) + BRANCH_VARIANTS


def branch_order(task_id: str, seed: int) -> list[str]:
    digest = sha256_text(f"{seed}\0{task_id}\0mechanism-branches")
    rng = random.Random(int(digest[:16], 16))
    variants = list(BRANCH_VARIANTS)
    rng.shuffle(variants)
    return variants


async def mechanism_preflight(args: argparse.Namespace) -> dict[str, Any]:
    async with aiohttp.ClientSession() as session:
        models, health = await asyncio.gather(
            get_json(session, endpoint_root(args.isra_url) + "/v1/models"),
            get_json(session, endpoint_root(args.isra_url) + "/health"),
        )
    snapshot = {"isra_models": models, "isra_health": health}
    if args.skip_preflight:
        return snapshot
    failures = [name for name, value in snapshot.items() if not value.get("ok")]
    if failures:
        raise RuntimeError(f"endpoint preflight failed: {', '.join(failures)}")
    config = health["response"].get("config", {})
    advertised = set(config.get("code_pipeline_variants", []))
    missing = set(ALL_VARIANTS) - advertised
    if missing:
        raise RuntimeError(f"ISRA endpoint is missing mechanism variants: {sorted(missing)}")
    if not config.get("frozen_initial_candidate"):
        raise RuntimeError("ISRA endpoint does not advertise frozen candidate injection")
    if not config.get("trusted_evidence_rollback"):
        raise RuntimeError("ISRA endpoint does not advertise trusted-evidence rollback")
    backend_model = health["response"].get("backend_model")
    if backend_model and backend_model != args.model and not args.allow_model_mismatch:
        raise RuntimeError(
            f"ISRA backend model {backend_model!r} differs from requested model {args.model!r}"
        )
    return snapshot


def make_body(
    args: argparse.Namespace,
    problem: dict[str, Any],
    variant: str,
    seed: int,
    initial_candidate: str | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": args.isra_model,
        "messages": task_messages(problem),
        "temperature": args.proposer_temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "stream": False,
        "seed": seed,
        "isra_variant": variant,
        "isra_temperature_override": args.proposer_temperature,
    }
    if initial_candidate is not None:
        body["isra_initial_candidate"] = initial_candidate
    if variant == "grounded_one_repair":
        body["isra_repair_temperature"] = args.repair_temperature
    return body


def result_from_outcome(
    *,
    args: argparse.Namespace,
    task_id: str,
    variant: str,
    seed: int,
    order_index: int,
    messages_sha256: str,
    outcome,
    evaluator: EvalPlusHumanEvalEvaluator,
    parent: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = outcome.payload or {}
    usage = payload.get("usage", {}) if isinstance(payload.get("usage"), dict) else {}
    metadata = (
        payload.get("isra_metadata", {})
        if isinstance(payload.get("isra_metadata"), dict)
        else {}
    )
    incremental_prompt = int(usage.get("prompt_tokens", 0) or 0)
    incremental_completion = int(usage.get("completion_tokens", 0) or 0)
    incremental_calls = int(metadata.get("llm_calls", 0) or 0)
    incremental_metrics = metadata.get("phase_metrics", []) or []
    parent_prompt = int(parent.get("prompt_tokens", 0) or 0) if parent else 0
    parent_completion = int(parent.get("completion_tokens", 0) or 0) if parent else 0
    parent_calls = int(parent.get("llm_calls", 0) or 0) if parent else 0
    parent_latency = float(parent.get("latency_ms", 0) or 0) if parent else 0.0
    parent_metrics = list(parent.get("phase_metrics", []) or []) if parent else []

    record: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "run_id": args.run_id,
        "task_id": task_id,
        "model": args.model,
        "variant": variant,
        "seed": seed,
        "status": outcome.status,
        "passed": None,
        "response": outcome.response,
        "latency_ms": round(parent_latency + outcome.latency_ms, 3),
        "incremental_latency_ms": outcome.latency_ms,
        "prompt_tokens": parent_prompt + incremental_prompt,
        "completion_tokens": parent_completion + incremental_completion,
        "incremental_prompt_tokens": incremental_prompt,
        "incremental_completion_tokens": incremental_completion,
        "llm_calls": parent_calls + incremental_calls,
        "incremental_llm_calls": incremental_calls,
        "phase_metrics": parent_metrics + incremental_metrics,
        "incremental_phase_metrics": incremental_metrics,
        "termination_reason": metadata.get("termination_reason", outcome.status),
        "evaluator_message": "not evaluated",
        "endpoint_attempts": outcome.attempts,
        "retry_history": outcome.retry_history,
        "finish_reason": outcome.finish_reason,
        "variant_order_index": order_index,
        "task_messages_sha256": messages_sha256,
        "candidate_trace": metadata.get("candidate_trace"),
        "parent_variant": parent.get("variant") if parent else None,
        "parent_candidate_sha256": sha256_text(parent["response"]) if parent else None,
    }
    if outcome.error:
        record["error"] = outcome.error
    if outcome.status == "completed":
        evaluator_started = time.perf_counter()
        try:
            evaluated = evaluator.evaluate(task_id, outcome.response)
        except Exception as exc:
            record["status"] = "evaluator_error"
            record["evaluator_message"] = f"{type(exc).__name__}: {exc}"[:1000]
        else:
            record["passed"] = bool(evaluated.pop("passed"))
            record["evaluator_message"] = evaluated.pop("message")
            record["evaluator_details"] = evaluated
        record["evaluator_latency_ms"] = round(
            (time.perf_counter() - evaluator_started) * 1000, 3
        )
    return record


async def execute_mechanism_run(
    args: argparse.Namespace,
    problems: dict[str, dict[str, Any]],
    store: ResultStore,
    evaluator: EvalPlusHumanEvalEvaluator,
) -> None:
    headers_base = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {args.api_key}",
    }
    total = len(problems) * len(args.seeds) * len(ALL_VARIANTS)
    completed = len(store.records)

    async with aiohttp.ClientSession() as session:
        for task_id, problem in problems.items():
            messages_sha256 = sha256_json(task_messages(problem))
            for seed in args.seeds:
                parent_key = (args.run_id, args.model, task_id, PARENT_VARIANT, seed)
                if store.has(parent_key):
                    parent = store.records[parent_key]
                else:
                    headers = dict(headers_base)
                    headers["X-Session-Id"] = sha256_text("\0".join(map(str, parent_key)))[:24]
                    headers["X-Benchmark-Variant"] = PARENT_VARIANT
                    headers["X-Benchmark-Task-Id"] = task_id
                    outcome = await call_openai_endpoint(
                        session,
                        url=args.isra_url,
                        body=make_body(args, problem, PARENT_VARIANT, seed),
                        headers=headers,
                        timeout_seconds=args.timeout,
                        retries=args.retries,
                        backoff_seconds=args.retry_backoff,
                    )
                    parent = result_from_outcome(
                        args=args,
                        task_id=task_id,
                        variant=PARENT_VARIANT,
                        seed=seed,
                        order_index=-1,
                        messages_sha256=messages_sha256,
                        outcome=outcome,
                        evaluator=evaluator,
                    )
                    store.append(parent)
                    completed += 1
                    print(
                        f"[{completed:>3}/{total}] {task_id:<14} {PARENT_VARIANT:<24} "
                        f"{parent['status']:<16}",
                        flush=True,
                    )

                if parent.get("status") != "completed" or not parent.get("response"):
                    for order_index, variant in enumerate(branch_order(task_id, seed)):
                        key = (args.run_id, args.model, task_id, variant, seed)
                        if store.has(key):
                            continue
                        skipped = {
                            "schema_version": RESULT_SCHEMA_VERSION,
                            "run_id": args.run_id,
                            "task_id": task_id,
                            "model": args.model,
                            "variant": variant,
                            "seed": seed,
                            "status": "skipped_parent_unavailable",
                            "passed": None,
                            "response": "",
                            "latency_ms": float(parent.get("latency_ms", 0) or 0),
                            "incremental_latency_ms": 0.0,
                            "prompt_tokens": int(parent.get("prompt_tokens", 0) or 0),
                            "completion_tokens": int(parent.get("completion_tokens", 0) or 0),
                            "incremental_prompt_tokens": 0,
                            "incremental_completion_tokens": 0,
                            "llm_calls": int(parent.get("llm_calls", 0) or 0),
                            "incremental_llm_calls": 0,
                            "phase_metrics": list(parent.get("phase_metrics", []) or []),
                            "incremental_phase_metrics": [],
                            "termination_reason": "parent_candidate_unavailable",
                            "evaluator_message": "not evaluated: parent candidate unavailable",
                            "endpoint_attempts": 0,
                            "retry_history": [],
                            "finish_reason": None,
                            "variant_order_index": order_index,
                            "task_messages_sha256": messages_sha256,
                            "candidate_trace": None,
                            "parent_variant": parent.get("variant"),
                            "parent_candidate_sha256": None,
                            "error": (
                                f"parent {parent.get('status', 'unknown')}: "
                                f"{parent.get('error', 'candidate unavailable')}"
                            )[:1000],
                        }
                        store.append(skipped)
                        completed += 1
                        print(
                            f"[{completed:>3}/{total}] {task_id:<14} {variant:<24} "
                            f"{skipped['status']:<16}",
                            flush=True,
                        )
                    continue

                frozen_candidate = str(parent["response"])
                for order_index, variant in enumerate(branch_order(task_id, seed)):
                    key = (args.run_id, args.model, task_id, variant, seed)
                    if store.has(key):
                        continue
                    headers = dict(headers_base)
                    headers["X-Session-Id"] = sha256_text("\0".join(map(str, key)))[:24]
                    headers["X-Benchmark-Variant"] = variant
                    headers["X-Benchmark-Task-Id"] = task_id
                    outcome = await call_openai_endpoint(
                        session,
                        url=args.isra_url,
                        body=make_body(
                            args,
                            problem,
                            variant,
                            seed,
                            initial_candidate=frozen_candidate,
                        ),
                        headers=headers,
                        timeout_seconds=args.timeout,
                        retries=args.retries,
                        backoff_seconds=args.retry_backoff,
                    )
                    record = result_from_outcome(
                        args=args,
                        task_id=task_id,
                        variant=variant,
                        seed=seed,
                        order_index=order_index,
                        messages_sha256=messages_sha256,
                        outcome=outcome,
                        evaluator=evaluator,
                        parent=parent,
                    )
                    trace = record.get("candidate_trace") or {}
                    if trace.get("candidate_a") != frozen_candidate:
                        raise RuntimeError(
                            f"endpoint violated frozen candidate parity for {task_id} {variant}"
                        )
                    store.append(record)
                    completed += 1
                    print(
                        f"[{completed:>3}/{total}] {task_id:<14} {variant:<24} "
                        f"{record['status']:<16}",
                        flush=True,
                    )


def manifest_configuration(
    args: argparse.Namespace,
    problems: dict[str, dict[str, Any]],
    dataset: dict[str, Any],
    endpoint_snapshot: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment_kind": "frozen_initial_candidate_mechanism_attribution",
        "model": {
            "identifier": args.model,
            "weights_revision": args.model_revision,
            "quantization": args.quantization,
            "weights_fingerprint": args.weights_fingerprint,
        },
        "backend": {
            "isra_url": args.isra_url,
            "name": args.backend_name,
            "versions": args.backend_versions,
            "remote_machine": args.backend_machine,
            "endpoint_snapshot": endpoint_snapshot,
        },
        "dataset": dataset,
        "evaluator": {
            "name": "EvalPlus HumanEval+ official oracle",
            "version": EvalPlusHumanEvalEvaluator.VERSION,
            "container_image": getattr(args, "evaluator_container_image", "") or None,
            "hidden_tests_exposed_to_models": False,
        },
        "variants": list(ALL_VARIANTS),
        "dependency": {
            "parent": PARENT_VARIANT,
            "branches": list(BRANCH_VARIANTS),
            "candidate_transfer": "exact response bytes",
            "branch_cost_includes_parent": True,
            "hidden_evaluator_feedback_forwarded": False,
        },
        "temperatures": {
            "proposer": args.proposer_temperature,
            "grounded_repair": args.repair_temperature,
        },
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "seeds": args.seeds,
        "branch_order": "per-task deterministic SHA-256 shuffle",
        "timeout_seconds": args.timeout,
        "retry_policy": {
            "infrastructure_only": True,
            "retries": args.retries,
            "backoff_seconds": args.retry_backoff,
        },
        "task_message_hashes": {
            task_id: sha256_json(task_messages(problem))
            for task_id, problem in problems.items()
        },
        "orchestrator": orchestrator_snapshot(),
        "harness_source_sha256": {
            "mechanism_benchmark.py": file_sha256(Path(__file__).resolve()),
            "paired_core.py": file_sha256(ROOT / "benchmarks" / "paired_core.py"),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isra-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--isra-model", default="isra-a3b")
    parser.add_argument("--model-revision", default="unspecified")
    parser.add_argument("--quantization", default="unspecified")
    parser.add_argument("--weights-fingerprint", default="unspecified")
    parser.add_argument("--backend-name", default="OpenAI-compatible")
    parser.add_argument("--backend-versions", default="unspecified")
    parser.add_argument("--backend-machine", default="unspecified")
    parser.add_argument(
        "--evaluator-container-image",
        default=os.environ.get("ISRA_EVALUATOR_CONTAINER_IMAGE", ""),
        help="exact evaluator container tag/digest recorded in the immutable manifest",
    )
    parser.add_argument("--api-key", default=os.environ.get("MLX_LOCAL_API_KEY", "test-key"))
    parser.add_argument("--task-count", type=int, default=20)
    parser.add_argument("--task-start", type=int, default=0)
    parser.add_argument("--task-ids", default="")
    parser.add_argument("--seeds", type=parse_seeds, default=[0])
    parser.add_argument("--proposer-temperature", type=float, default=0.0)
    parser.add_argument("--repair-temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-backoff", type=float, default=0.5)
    parser.add_argument("--run-root", type=Path, default=ROOT / "benchmark_runs")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--allow-model-mismatch", action="store_true")
    return parser


async def async_main(args: argparse.Namespace) -> int:
    if args.task_count <= 0 or args.max_tokens <= 0 or args.timeout <= 0:
        raise ValueError("task-count, max-tokens, and timeout must be positive")
    for value in (args.proposer_temperature, args.repair_temperature):
        if not 0.0 <= value <= 2.0:
            raise ValueError("temperatures must be between 0 and 2")
    if platform.system() == "Darwin":
        raise RuntimeError(
            "Full EvalPlus execution is disabled on macOS. Run the mechanism "
            "benchmark inside the pinned EvalPlus Docker image."
        )
    if not args.run_id:
        args.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]

    problems, dataset = load_tasks(args)
    evaluator = EvalPlusHumanEvalEvaluator(problems, dataset["content_md5"])
    evaluator_sanity = evaluator.sanity_check()
    endpoint_snapshot = await mechanism_preflight(args)
    configuration = manifest_configuration(args, problems, dataset, endpoint_snapshot)
    fingerprint = sha256_json(configuration)
    run_dir = args.run_root.resolve() / args.run_id
    partial_path = run_dir / "manifest.partial.json"
    final_path = run_dir / "manifest.json"
    results_path = run_dir / "results.jsonl"

    if final_path.exists():
        print(f"Run already finalized: {final_path}")
        return 0
    if partial_path.exists():
        partial = json.loads(partial_path.read_text(encoding="utf-8"))
        if partial.get("configuration_fingerprint") != fingerprint:
            raise RuntimeError("resume configuration does not match immutable partial manifest")
    else:
        partial = {
            "manifest_schema_version": 1,
            "run_id": args.run_id,
            "started_at": utc_now(),
            "configuration_fingerprint": fingerprint,
            "configuration": configuration,
            "evaluator_sanity": evaluator_sanity,
            "git": git_snapshot(),
            "runner_machine": machine_snapshot(),
        }
        write_json_exclusive(partial_path, partial)

    store = ResultStore(results_path)
    await execute_mechanism_run(args, problems, store, evaluator)
    expected = {
        (args.run_id, args.model, task_id, variant, seed)
        for task_id in problems
        for seed in args.seeds
        for variant in ALL_VARIANTS
    }
    missing = sorted(expected - set(store.records))
    if missing:
        print(f"Partial run preserved at {run_dir}; {len(missing)} records are missing")
        return 2

    finalized = dict(partial)
    finalized["ended_at"] = utc_now()
    finalized["result_count"] = len(store.records)
    finalized["results_sha256"] = file_sha256(results_path)
    write_json_exclusive(final_path, finalized)
    print(f"Finalized run: {final_path}")
    return 0


def main() -> int:
    args = build_parser().parse_args()
    try:
        return asyncio.run(async_main(args))
    except KeyboardInterrupt:
        print("Interrupted; partial manifest and JSONL records were preserved", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())

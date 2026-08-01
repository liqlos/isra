#!/usr/bin/env python3
"""Paired direct-vs-SPA code-generation benchmark with immutable provenance."""

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

from anchored_generation import (  # noqa: E402
    DEFAULT_ANCHOR_MODE,
    DEFAULT_MASK_TOKEN,
    DEFAULT_STRENGTH,
    PAPER_URL,
)
from benchmarks.harness_common import (  # noqa: E402
    endpoint_root,
    file_sha256,
    get_json,
    git_snapshot,
    load_tasks,
    machine_snapshot,
    parse_seeds,
    utc_now,
)
from benchmarks.paired_core import (  # noqa: E402
    RESULT_SCHEMA_VERSION,
    EvalPlusHumanEvalEvaluator,
    ResultStore,
    call_openai_endpoint,
    sha256_json,
    sha256_text,
    task_messages,
    write_json_exclusive,
)


VARIANTS = ("direct_greedy", "spa_greedy")


def variant_order(task_id: str, seed: int) -> list[str]:
    digest = sha256_text(f"{seed}\0{task_id}\0direct-spa-order")
    rng = random.Random(int(digest[:16], 16))
    variants = list(VARIANTS)
    rng.shuffle(variants)
    return variants


async def decoding_preflight(args: argparse.Namespace) -> dict[str, Any]:
    root = endpoint_root(args.endpoint_url)
    async with aiohttp.ClientSession() as session:
        models, health = await asyncio.gather(
            get_json(session, root + "/v1/models"),
            get_json(session, root + "/health"),
        )
    snapshot = {"models": models, "health": health}
    if args.skip_preflight:
        return snapshot
    failures = [name for name, value in snapshot.items() if not value.get("ok")]
    if failures:
        raise RuntimeError(f"endpoint preflight failed: {', '.join(failures)}")

    config = health["response"].get("config", {})
    modes = set(config.get("decoding_modes", []))
    if not set(("direct", "spa")).issubset(modes):
        raise RuntimeError(f"endpoint does not advertise direct and spa modes: {modes}")
    if config.get("hidden_tests_exposed") is not False:
        raise RuntimeError("endpoint does not prove hidden tests are isolated")
    if config.get("answer_candidates") != 1 or config.get("llm_calls") != 1:
        raise RuntimeError("endpoint is not a one-answer, one-call decoder")
    if not args.allow_algorithm_mismatch:
        if config.get("mask_token") != args.mask_token:
            raise RuntimeError("endpoint mask token differs from requested configuration")
        if config.get("default_anchor_mode") != args.anchor_mode:
            raise RuntimeError("endpoint anchor mode differs from requested configuration")
        if abs(float(config.get("default_strength")) - args.strength) > 1e-9:
            raise RuntimeError("endpoint anchoring strength differs from requested configuration")
    return snapshot


def make_body(
    args: argparse.Namespace,
    problem: dict[str, Any],
    variant: str,
    seed: int,
) -> dict[str, Any]:
    mode = "spa" if variant == "spa_greedy" else "direct"
    return {
        "model": f"{args.model_id}-{mode}",
        "messages": task_messages(problem),
        "decoding_mode": mode,
        "anchoring_strength": args.strength,
        "anchor_mode": args.anchor_mode,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": args.max_tokens,
        "stream": False,
        "seed": seed,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def manifest_configuration(
    args: argparse.Namespace,
    problems: dict[str, dict[str, Any]],
    dataset: dict[str, Any],
    endpoint_snapshot: dict[str, Any],
) -> dict[str, Any]:
    messages_sha256 = {
        task_id: sha256_json(task_messages(problem))
        for task_id, problem in problems.items()
    }
    return {
        "schema_version": 1,
        "objective": "one-pass quality improvement with bounded latency",
        "model": {
            "identifier": args.model_id,
            "weights_revision": args.model_revision,
            "quantization": args.quantization,
            "weights_fingerprint": args.weights_fingerprint,
        },
        "backend": {
            "url": args.endpoint_url,
            "name": args.backend_name,
            "versions": args.backend_versions,
            "remote_machine": args.backend_machine,
            "endpoint_snapshot": endpoint_snapshot,
        },
        "dataset": dataset,
        "evaluator": {
            "name": "EvalPlus HumanEval+ official oracle",
            "version": EvalPlusHumanEvalEvaluator.VERSION,
            "container_image": args.evaluator_container_image or None,
            "hidden_tests_exposed_to_decoder": False,
        },
        "messages": {
            "per_task_sha256": messages_sha256,
            "identical_between_variants": True,
        },
        "variants": {
            "direct_greedy": {
                "decoding_mode": "direct",
                "temperature": 0.0,
                "top_p": 1.0,
                "max_tokens": args.max_tokens,
                "answer_candidates": 1,
                "llm_calls": 1,
            },
            "spa_greedy": {
                "decoding_mode": "spa",
                "temperature": 0.0,
                "top_p": 1.0,
                "max_tokens": args.max_tokens,
                "answer_candidates": 1,
                "llm_calls": 1,
                "strength": args.strength,
                "anchor_mode": args.anchor_mode,
                "mask_token": args.mask_token,
                "paper": PAPER_URL,
                "activation": "all treatment tasks; no test gate",
            },
        },
        "seeds": args.seeds,
        "variant_order": "per-task deterministic SHA-256-seeded shuffle",
        "timeout_seconds": args.timeout,
        "retry_policy": {
            "infrastructure_only": True,
            "retries": args.retries,
            "backoff_seconds": args.retry_backoff,
            "retry_statuses": ["timeout", "transport_error"],
        },
        "concurrency": 1,
        "harness_source_sha256": {
            "decoding_benchmark.py": file_sha256(Path(__file__).resolve()),
            "paired_core.py": file_sha256(ROOT / "benchmarks" / "paired_core.py"),
            "anchored_generation.py": file_sha256(ROOT / "anchored_generation.py"),
        },
    }


def result_from_outcome(
    *,
    args: argparse.Namespace,
    task_id: str,
    variant: str,
    seed: int,
    order_index: int,
    messages_sha256: str,
    outcome: Any,
    evaluator: EvalPlusHumanEvalEvaluator,
) -> dict[str, Any]:
    payload = outcome.payload or {}
    usage = payload.get("usage", {}) if isinstance(payload.get("usage"), dict) else {}
    metadata = (
        payload.get("decoding_metadata", {})
        if isinstance(payload.get("decoding_metadata"), dict)
        else {}
    )
    record: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "run_id": args.run_id,
        "task_id": task_id,
        "model": args.model_id,
        "variant": variant,
        "seed": seed,
        "status": outcome.status,
        "passed": None,
        "response": outcome.response,
        "latency_ms": outcome.latency_ms,
        "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
        "llm_calls": int(metadata.get("llm_calls", 1 if payload else 0) or 0),
        "phase_metrics": [],
        "termination_reason": metadata.get("algorithm", outcome.status),
        "evaluator_message": "not evaluated",
        "endpoint_attempts": outcome.attempts,
        "retry_history": outcome.retry_history,
        "finish_reason": outcome.finish_reason,
        "variant_order_index": order_index,
        "task_messages_sha256": messages_sha256,
        "decoding_metadata": metadata,
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


async def execute_run(
    args: argparse.Namespace,
    problems: dict[str, dict[str, Any]],
    store: ResultStore,
    evaluator: EvalPlusHumanEvalEvaluator,
) -> None:
    total = len(problems) * len(args.seeds) * len(VARIANTS)
    completed = len(store.records)
    headers_base = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {args.api_key}",
    }
    async with aiohttp.ClientSession() as session:
        for task_id, problem in problems.items():
            messages_sha256 = sha256_json(task_messages(problem))
            for seed in args.seeds:
                for order_index, variant in enumerate(variant_order(task_id, seed)):
                    key = (args.run_id, args.model_id, task_id, variant, seed)
                    if store.has(key):
                        continue
                    headers = dict(headers_base)
                    headers["X-Benchmark-Variant"] = variant
                    headers["X-Benchmark-Task-Id"] = task_id
                    outcome = await call_openai_endpoint(
                        session,
                        url=args.endpoint_url,
                        body=make_body(args, problem, variant, seed),
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
                    )
                    store.append(record)
                    completed += 1
                    pass_label = (
                        "-"
                        if record["passed"] is None
                        else ("PASS" if record["passed"] else "FAIL")
                    )
                    print(
                        f"[{completed:>3}/{total}] {task_id:<14} {variant:<15} "
                        f"{record['status']:<16} {pass_label} "
                        f"{record['latency_ms']:.1f}ms",
                        flush=True,
                    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint-url", required=True)
    parser.add_argument("--model-id", default="llama31-8b3bit")
    parser.add_argument("--model-revision", default="unspecified")
    parser.add_argument("--quantization", default="unspecified")
    parser.add_argument("--weights-fingerprint", default="unspecified")
    parser.add_argument("--backend-name", default="mlx-anchored-generation")
    parser.add_argument("--backend-versions", default="unspecified")
    parser.add_argument("--backend-machine", default="unspecified")
    parser.add_argument(
        "--evaluator-container-image",
        default=os.environ.get("SPA_EVALUATOR_CONTAINER_IMAGE", ""),
    )
    parser.add_argument("--api-key", default="local-research")
    parser.add_argument("--task-count", type=int, default=20)
    parser.add_argument("--task-start", type=int, default=20)
    parser.add_argument("--task-ids", default="")
    parser.add_argument("--seeds", type=parse_seeds, default=[0])
    parser.add_argument("--max-tokens", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-backoff", type=float, default=0.5)
    parser.add_argument("--strength", type=float, default=DEFAULT_STRENGTH)
    parser.add_argument(
        "--anchor-mode",
        choices=("natural_language", "last_user"),
        default=DEFAULT_ANCHOR_MODE,
    )
    parser.add_argument("--mask-token", default=DEFAULT_MASK_TOKEN)
    parser.add_argument("--run-root", type=Path, default=ROOT / "benchmark_runs")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--allow-algorithm-mismatch", action="store_true")
    parser.add_argument("--evaluator-sanity-only", action="store_true")
    return parser


async def async_main(args: argparse.Namespace) -> int:
    if args.task_count <= 0 or args.max_tokens <= 0 or args.timeout <= 0:
        raise ValueError("task-count, max-tokens and timeout must be positive")
    if args.retries < 0 or args.retry_backoff < 0:
        raise ValueError("retry settings cannot be negative")
    if not args.run_id:
        args.run_id = (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "-"
            + uuid.uuid4().hex[:8]
        )

    if platform.system() == "Darwin":
        if args.evaluator_sanity_only:
            os.environ.setdefault("EVALPLUS_MAX_MEMORY_BYTES", "-1")
        else:
            raise RuntimeError(
                "Full EvalPlus execution is disabled on macOS; run this runner "
                "inside the pinned evaluator container"
            )

    problems, dataset = load_tasks(args)
    evaluator = EvalPlusHumanEvalEvaluator(problems, dataset["content_md5"])
    evaluator_sanity = evaluator.sanity_check()
    if args.evaluator_sanity_only:
        print(json.dumps(evaluator_sanity, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    endpoint_snapshot = await decoding_preflight(args)
    configuration = manifest_configuration(
        args, problems, dataset, endpoint_snapshot
    )
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
            raise RuntimeError(
                "resume configuration does not match immutable partial manifest"
            )
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
    await execute_run(args, problems, store, evaluator)
    expected = {
        (args.run_id, args.model_id, task_id, variant, seed)
        for task_id in problems
        for seed in args.seeds
        for variant in VARIANTS
    }
    missing = sorted(expected - set(store.records))
    if missing:
        print(f"Partial run preserved at {run_dir}; {len(missing)} records missing")
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
        print(
            "Interrupted; partial manifest and completed JSONL records are preserved",
            file=sys.stderr,
        )
        return 130


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Paired direct-vs-SPA generation over a decoder-safe LiveCodeBench manifest."""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import random
import sys
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
    machine_snapshot,
    parse_seeds,
    utc_now,
)
from benchmarks.paired_core import (  # noqa: E402
    RESULT_SCHEMA_VERSION,
    ResultStore,
    call_openai_endpoint,
    sha256_json,
    sha256_text,
    write_json_exclusive,
)


VARIANTS = ("direct_greedy", "spa_greedy")
SAFE_TASK_KEYS = {
    "task_id",
    "contest_date",
    "platform",
    "difficulty",
    "prompt_messages",
    "prompt_sha256",
}


def variant_order(task_id: str, seed: int) -> list[str]:
    digest = sha256_text(f"{seed}\0{task_id}\0direct-spa-order")
    variants = list(VARIANTS)
    random.Random(int(digest[:16], 16)).shuffle(variants)
    return variants


def load_decoder_manifest(path: Path, task_limit: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != 1:
        raise ValueError("unsupported decoder manifest schema")
    contract = value.get("decoder_input_contract", {})
    if contract.get("contains_hidden_tests") is not False:
        raise RuntimeError("decoder manifest does not prove hidden-test isolation")
    tasks = value.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("decoder manifest has no tasks")
    for task in tasks:
        if not isinstance(task, dict) or set(task) != SAFE_TASK_KEYS:
            raise ValueError("decoder manifest contains unexpected task fields")
        messages = task.get("prompt_messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("task has no prompt messages")
        if sha256_json(messages) != task.get("prompt_sha256"):
            raise ValueError(f"prompt digest mismatch for {task.get('task_id')}")
    tasks = sorted(tasks, key=lambda task: str(task["task_id"]))
    if len({str(task["task_id"]) for task in tasks}) != len(tasks):
        raise ValueError("decoder manifest has duplicate task IDs")
    if task_limit:
        tasks = tasks[:task_limit]
    return tasks, value


async def preflight(args: argparse.Namespace) -> dict[str, Any]:
    root = endpoint_root(args.endpoint_url)
    async with aiohttp.ClientSession() as session:
        models, health = await asyncio.gather(
            get_json(session, root + "/v1/models"), get_json(session, root + "/health")
        )
    snapshot = {"models": models, "health": health}
    if not all(value.get("ok") for value in snapshot.values()):
        raise RuntimeError(f"endpoint preflight failed: {snapshot}")
    config = health["response"].get("config", {})
    if not {"direct", "spa"}.issubset(set(config.get("decoding_modes", []))):
        raise RuntimeError("endpoint does not advertise direct and SPA decoding")
    if config.get("hidden_tests_exposed") is not False:
        raise RuntimeError("endpoint does not attest hidden-test isolation")
    if config.get("answer_candidates") != 1 or config.get("llm_calls") != 1:
        raise RuntimeError("endpoint is not a one-answer, one-call decoder")
    for key, expected in {
        "mask_token": args.mask_token,
        "default_anchor_mode": args.anchor_mode,
    }.items():
        if config.get(key) != expected:
            raise RuntimeError(f"endpoint {key} does not match requested configuration")
    if abs(float(config.get("default_strength")) - args.strength) > 1e-9:
        raise RuntimeError("endpoint strength does not match requested configuration")
    return snapshot


def make_body(args: argparse.Namespace, task: dict[str, Any], variant: str, seed: int) -> dict[str, Any]:
    mode = "spa" if variant == "spa_greedy" else "direct"
    return {
        "model": f"{args.model_id}-{mode}",
        "messages": task["prompt_messages"],
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


def configuration(args: argparse.Namespace, tasks: list[dict[str, Any]], source: dict[str, Any], endpoint: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "objective": "confirmatory one-pass direct-vs-SPA quality comparison",
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
            "endpoint_snapshot": endpoint,
        },
        "dataset": {
            "decoder_manifest_sha256": file_sha256(args.task_manifest),
            "tasks_sha256": sha256_json(tasks),
            "task_count": len(tasks),
            "selection": source["selection"],
            "official_source": source["official_source"],
            "hidden_tests_exposed_to_decoder": False,
        },
        "messages": {
            "per_task_sha256": {task["task_id"]: task["prompt_sha256"] for task in tasks},
            "identical_between_variants": True,
        },
        "variants": {
            "direct_greedy": {"decoding_mode": "direct", "temperature": 0.0, "top_p": 1.0, "max_tokens": args.max_tokens, "answer_candidates": 1, "llm_calls": 1},
            "spa_greedy": {"decoding_mode": "spa", "temperature": 0.0, "top_p": 1.0, "max_tokens": args.max_tokens, "answer_candidates": 1, "llm_calls": 1, "strength": args.strength, "anchor_mode": args.anchor_mode, "mask_token": args.mask_token, "paper": PAPER_URL},
        },
        "seeds": args.seeds,
        "variant_order": "per-task deterministic SHA-256-seeded shuffle",
        "retry_policy": {"infrastructure_only": True, "retries": args.retries, "backoff_seconds": args.retry_backoff, "retry_statuses": ["timeout", "transport_error"]},
        "concurrency": 1,
        "harness_source_sha256": {
            "livecodebench_benchmark.py": file_sha256(Path(__file__).resolve()),
            "paired_core.py": file_sha256(ROOT / "benchmarks" / "paired_core.py"),
            "anchored_generation.py": file_sha256(ROOT / "anchored_generation.py"),
        },
    }


def record_from_outcome(args: argparse.Namespace, task: dict[str, Any], variant: str, seed: int, order_index: int, outcome: Any) -> dict[str, Any]:
    payload = outcome.payload or {}
    usage = payload.get("usage", {}) if isinstance(payload.get("usage"), dict) else {}
    metadata = payload.get("decoding_metadata", {}) if isinstance(payload.get("decoding_metadata"), dict) else {}
    record: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "run_id": args.run_id,
        "task_id": task["task_id"],
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
        "endpoint_attempts": outcome.attempts,
        "retry_history": outcome.retry_history,
        "finish_reason": outcome.finish_reason,
        "variant_order_index": order_index,
        "task_messages_sha256": task["prompt_sha256"],
        "decoding_metadata": metadata,
        "evaluator_message": "pending separate official LiveCodeBench evaluation",
    }
    if outcome.error:
        record["error"] = outcome.error
    return record


async def execute(args: argparse.Namespace, tasks: list[dict[str, Any]], store: ResultStore) -> None:
    total = len(tasks) * len(args.seeds) * len(VARIANTS)
    completed = len(store.records)
    headers_base = {"Content-Type": "application/json", "Authorization": f"Bearer {args.api_key}"}
    async with aiohttp.ClientSession() as session:
        for task in tasks:
            for seed in args.seeds:
                for order_index, variant in enumerate(variant_order(task["task_id"], seed)):
                    key = (args.run_id, args.model_id, task["task_id"], variant, seed)
                    if store.has(key):
                        continue
                    headers = {**headers_base, "X-Benchmark-Variant": variant, "X-Benchmark-Task-Id": task["task_id"]}
                    outcome = await call_openai_endpoint(session, url=args.endpoint_url, body=make_body(args, task, variant, seed), headers=headers, timeout_seconds=args.timeout, retries=args.retries, backoff_seconds=args.retry_backoff)
                    record = record_from_outcome(args, task, variant, seed, order_index, outcome)
                    store.append(record)
                    completed += 1
                    print(f"[{completed:>4}/{total}] {task['task_id']:<12} {variant:<15} {record['status']:<16} {record['latency_ms']:.1f}ms", flush=True)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--task-manifest", type=Path, required=True)
    value.add_argument("--endpoint-url", required=True)
    value.add_argument("--model-id", default="llama31-8b3bit")
    value.add_argument("--model-revision", default="unspecified")
    value.add_argument("--quantization", default="unspecified")
    value.add_argument("--weights-fingerprint", default="unspecified")
    value.add_argument("--backend-name", default="mlx-anchored-generation")
    value.add_argument("--backend-versions", default="unspecified")
    value.add_argument("--backend-machine", default="unspecified")
    value.add_argument("--api-key", default="local-research")
    value.add_argument("--seeds", type=parse_seeds, default=[0])
    value.add_argument("--max-tokens", type=int, default=2000)
    value.add_argument("--timeout", type=float, default=600.0)
    value.add_argument("--retries", type=int, default=2)
    value.add_argument("--retry-backoff", type=float, default=0.5)
    value.add_argument("--strength", type=float, default=DEFAULT_STRENGTH)
    value.add_argument("--anchor-mode", choices=("natural_language", "last_user"), default=DEFAULT_ANCHOR_MODE)
    value.add_argument("--mask-token", default=DEFAULT_MASK_TOKEN)
    value.add_argument("--task-limit", type=int, default=0, help="smoke only; zero evaluates every selected task")
    value.add_argument("--run-root", type=Path, default=ROOT / "benchmark_runs")
    value.add_argument("--run-id", default="")
    return value


async def async_main(args: argparse.Namespace) -> int:
    if args.max_tokens <= 0 or args.timeout <= 0 or args.task_limit < 0:
        raise ValueError("max-tokens and timeout must be positive; task-limit cannot be negative")
    if not args.run_id:
        args.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-lcb-" + uuid.uuid4().hex[:8]
    tasks, source = load_decoder_manifest(args.task_manifest, args.task_limit)
    endpoint = await preflight(args)
    run_dir = args.run_root.resolve() / args.run_id
    partial_path, final_path, results_path = run_dir / "manifest.partial.json", run_dir / "manifest.json", run_dir / "results.jsonl"
    frozen = configuration(args, tasks, source, endpoint)
    fingerprint = sha256_json(frozen)
    if final_path.exists():
        print(f"Run already finalized: {final_path}")
        return 0
    if partial_path.exists():
        partial = json.loads(partial_path.read_text(encoding="utf-8"))
        if partial.get("configuration_fingerprint") != fingerprint:
            raise RuntimeError("resume configuration does not match immutable partial manifest")
    else:
        partial = {"manifest_schema_version": 1, "run_id": args.run_id, "started_at": utc_now(), "configuration_fingerprint": fingerprint, "configuration": frozen, "git": git_snapshot(), "runner_machine": machine_snapshot(), "platform": platform.platform()}
        write_json_exclusive(partial_path, partial)
    store = ResultStore(results_path)
    await execute(args, tasks, store)
    expected = {(args.run_id, args.model_id, task["task_id"], variant, seed) for task in tasks for seed in args.seeds for variant in VARIANTS}
    missing = sorted(expected - set(store.records))
    if missing:
        print(f"Partial run preserved at {run_dir}; {len(missing)} records missing")
        return 2
    finalized = {**partial, "ended_at": utc_now(), "result_count": len(store.records), "results_sha256": file_sha256(results_path)}
    write_json_exclusive(final_path, finalized)
    print(f"Finalized run: {final_path}")
    return 0


def main() -> int:
    try:
        return asyncio.run(async_main(parser().parse_args()))
    except KeyboardInterrupt:
        print("Interrupted; partial manifest and completed JSONL records are preserved", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())

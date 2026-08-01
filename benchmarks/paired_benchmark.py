#!/usr/bin/env python3
"""Paired Direct/ISRA benchmark runner with immutable provenance and resume."""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import aiohttp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.paired_core import (  # noqa: E402
    RESULT_SCHEMA_VERSION,
    VARIANT_SPECS,
    EvalPlusHumanEvalEvaluator,
    ResultStore,
    call_openai_endpoint,
    primary_key,
    randomized_variant_order,
    sha256_json,
    sha256_text,
    task_messages,
    write_json_exclusive,
)


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
            decoded = json.loads(raw)
            return {"ok": True, "response": decoded}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


async def endpoint_preflight(args: argparse.Namespace) -> dict[str, Any]:
    async with aiohttp.ClientSession() as session:
        direct_models, isra_models, isra_health = await asyncio.gather(
            get_json(session, endpoint_root(args.direct_url) + "/v1/models"),
            get_json(session, endpoint_root(args.isra_url) + "/v1/models"),
            get_json(session, endpoint_root(args.isra_url) + "/health"),
        )
    snapshot = {
        "direct_models": direct_models,
        "isra_models": isra_models,
        "isra_health": isra_health,
    }
    if args.skip_preflight:
        return snapshot
    failures = [name for name, value in snapshot.items() if not value.get("ok")]
    if failures:
        raise RuntimeError(f"endpoint preflight failed: {', '.join(failures)}")

    direct_ids = {
        item.get("id")
        for item in direct_models["response"].get("data", [])
        if isinstance(item, dict)
    }
    health = isra_health["response"]
    isra_backend_model = health.get("backend_model")
    if args.model not in direct_ids and not args.allow_model_mismatch:
        raise RuntimeError(f"requested model {args.model!r} not listed by Direct endpoint: {sorted(direct_ids)}")
    if isra_backend_model and isra_backend_model != args.model and not args.allow_model_mismatch:
        raise RuntimeError(
            f"ISRA backend model {isra_backend_model!r} differs from Direct model {args.model!r}"
        )
    capabilities = health.get("config", {}) if isinstance(health.get("config"), dict) else {}
    if not capabilities.get("benchmark_temperature_override") and not args.allow_unverified_isra:
        raise RuntimeError("ISRA endpoint does not advertise the greedy benchmark override")
    if not capabilities.get("phase_metrics") and not args.allow_unverified_isra:
        raise RuntimeError("ISRA endpoint does not advertise phase metrics")
    return snapshot


def run_command(*command: str) -> str:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else ""


def git_snapshot() -> dict[str, Any]:
    status = run_command("git", "status", "--porcelain=v1")
    return {
        "commit": run_command("git", "rev-parse", "HEAD") or None,
        "branch": run_command("git", "branch", "--show-current") or None,
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


def package_versions(names: list[str]) -> dict[str, str | None]:
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
        "runtime_packages": package_versions(["aiohttp", "evalplus", "numpy"]),
    }


def orchestrator_snapshot() -> dict[str, Any]:
    import isra_orchestrator as orchestrator

    prompt_names = [
        "PHASE1_SYSTEM_GENERAL",
        "PHASE1_SYSTEM_CODE",
        "PHASE1_SYSTEM_MATH",
        "PHASE2_SYSTEM",
        "PHASE2_SYSTEM_MATH",
        "PHASE3_SYSTEM",
        "PHASE4_SYSTEM",
    ]
    return {
        "source_sha256": file_sha256(ROOT / "isra_orchestrator.py"),
        "prompt_sha256": {
            name: sha256_text(getattr(orchestrator, name)) for name in prompt_names
        },
        "phase_parameters": orchestrator.PHASE_PARAMS,
        "max_iterations": orchestrator.MAX_ITERATIONS,
        "confidence_threshold": orchestrator.CONFIDENCE_THRESHOLD,
        "stagnation_threshold": orchestrator.STAGNATION_THRESHOLD,
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


def manifest_configuration(
    args: argparse.Namespace,
    problems: dict[str, dict[str, Any]],
    dataset: dict[str, Any],
    endpoint_snapshot: dict[str, Any],
) -> dict[str, Any]:
    message_hashes = {
        task_id: sha256_json(task_messages(problem)) for task_id, problem in problems.items()
    }
    variants = {
        name: {
            "endpoint_kind": spec.endpoint_kind,
            "temperature": spec.temperature,
            "top_p": spec.top_p,
            "max_tokens": args.max_tokens,
            "isra_temperature_override": spec.isra_temperature_override,
            "enable_thinking": False,
        }
        for name, spec in VARIANT_SPECS.items()
    }
    return {
        "schema_version": 1,
        "model": {
            "identifier": args.model,
            "weights_revision": args.model_revision,
            "quantization": args.quantization,
            "weights_fingerprint": args.weights_fingerprint,
        },
        "backend": {
            "direct_url": args.direct_url,
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
        "task_message_template": {
            "system_sha256": sha256_text(task_messages(next(iter(problems.values())))[0]["content"]),
            "per_task_messages_sha256": message_hashes,
        },
        "variants": variants,
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
        "orchestrator": orchestrator_snapshot(),
        "harness_source_sha256": {
            "paired_benchmark.py": file_sha256(Path(__file__).resolve()),
            "paired_core.py": file_sha256(ROOT / "benchmarks" / "paired_core.py"),
        },
    }


def make_body(
    args: argparse.Namespace,
    problem: dict[str, Any],
    variant: str,
    seed: int,
) -> dict[str, Any]:
    spec = VARIANT_SPECS[variant]
    body: dict[str, Any] = {
        "model": args.model if spec.endpoint_kind == "direct" else args.isra_model,
        "messages": task_messages(problem),
        "temperature": spec.temperature,
        "top_p": spec.top_p,
        "max_tokens": args.max_tokens,
        "stream": False,
        "seed": seed,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if spec.isra_temperature_override is not None:
        body["isra_temperature_override"] = spec.isra_temperature_override
    return body


async def execute_run(
    args: argparse.Namespace,
    problems: dict[str, dict[str, Any]],
    store: ResultStore,
    evaluator: EvalPlusHumanEvalEvaluator,
) -> None:
    headers_base = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {args.api_key}",
    }
    total = len(problems) * len(args.seeds) * len(VARIANT_SPECS)
    completed = len(store.records)
    async with aiohttp.ClientSession() as session:
        for task_id, problem in problems.items():
            messages = task_messages(problem)
            messages_sha256 = sha256_json(messages)
            for seed in args.seeds:
                order = randomized_variant_order(task_id, seed)
                for order_index, variant in enumerate(order):
                    key = (args.run_id, args.model, task_id, variant, seed)
                    if store.has(key):
                        continue
                    spec = VARIANT_SPECS[variant]
                    url = args.direct_url if spec.endpoint_kind == "direct" else args.isra_url
                    headers = dict(headers_base)
                    headers["X-Session-Id"] = sha256_text("\0".join(map(str, key)))[:24]
                    headers["X-Benchmark-Variant"] = variant
                    headers["X-Benchmark-Task-Id"] = task_id
                    outcome = await call_openai_endpoint(
                        session,
                        url=url,
                        body=make_body(args, problem, variant, seed),
                        headers=headers,
                        timeout_seconds=args.timeout,
                        retries=args.retries,
                        backoff_seconds=args.retry_backoff,
                    )
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
                        "latency_ms": outcome.latency_ms,
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "llm_calls": 0,
                        "phase_metrics": [],
                        "termination_reason": outcome.status,
                        "evaluator_message": "not evaluated",
                        "endpoint_attempts": outcome.attempts,
                        "retry_history": outcome.retry_history,
                        "finish_reason": outcome.finish_reason,
                        "variant_order_index": order_index,
                        "task_messages_sha256": messages_sha256,
                    }
                    if outcome.error:
                        record["error"] = outcome.error
                    payload = outcome.payload or {}
                    usage = payload.get("usage", {}) if isinstance(payload.get("usage"), dict) else {}
                    metadata = (
                        payload.get("isra_metadata", {})
                        if isinstance(payload.get("isra_metadata"), dict)
                        else {}
                    )
                    record["prompt_tokens"] = int(usage.get("prompt_tokens", 0) or 0)
                    record["completion_tokens"] = int(usage.get("completion_tokens", 0) or 0)
                    record["llm_calls"] = int(
                        metadata.get(
                            "llm_calls",
                            1 if spec.endpoint_kind == "direct" and payload else 0,
                        )
                        or 0
                    )
                    record["phase_metrics"] = metadata.get("phase_metrics", []) or []
                    record["termination_reason"] = metadata.get(
                        "termination_reason",
                        "direct" if spec.endpoint_kind == "direct" and payload else outcome.status,
                    )
                    if outcome.status == "completed":
                        evaluator_start = time.perf_counter()
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
                            (time.perf_counter() - evaluator_start) * 1000, 3
                        )
                    store.append(record)
                    completed += 1
                    pass_label = "-" if record["passed"] is None else ("PASS" if record["passed"] else "FAIL")
                    print(
                        f"[{completed:>3}/{total}] {task_id:<14} {variant:<15} "
                        f"{record['status']:<16} {pass_label} {record['latency_ms']:.1f}ms",
                        flush=True,
                    )


def parse_seeds(value: str) -> list[int]:
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return seeds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct-url", required=True)
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
    parser.add_argument("--max-tokens", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-backoff", type=float, default=0.5)
    parser.add_argument("--run-root", type=Path, default=ROOT / "benchmark_runs")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument(
        "--evaluator-sanity-only",
        action="store_true",
        help="run canonical/known-bad evaluator gates and exit before endpoint preflight",
    )
    parser.add_argument("--allow-model-mismatch", action="store_true")
    parser.add_argument("--allow-unverified-isra", action="store_true")
    return parser


async def async_main(args: argparse.Namespace) -> int:
    if args.task_count <= 0 or args.max_tokens <= 0 or args.timeout <= 0:
        raise ValueError("task-count, max-tokens, and timeout must be positive")
    if args.retries < 0 or args.retry_backoff < 0:
        raise ValueError("retry settings cannot be negative")
    if not args.run_id:
        args.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]

    if platform.system() == "Darwin":
        if args.evaluator_sanity_only:
            # EvalPlus 0.3.1 tries to set RLIMIT_AS on macOS, where the current
            # hard limit can be lower than its requested 4 GiB limit. The
            # official environment switch disables only that memory cap. This
            # mode executes trusted canonical solutions and an intentional
            # raise, never model output.
            os.environ.setdefault("EVALPLUS_MAX_MEMORY_BYTES", "-1")
        else:
            raise RuntimeError(
                "Full EvalPlus execution is disabled on macOS. Run this benchmark "
                "inside the pinned EvalPlus Docker image; --evaluator-sanity-only "
                "is the only local Darwin execution mode."
            )

    problems, dataset = load_tasks(args)
    evaluator = EvalPlusHumanEvalEvaluator(problems, dataset["content_md5"])
    evaluator_sanity = evaluator.sanity_check()
    if args.evaluator_sanity_only:
        print(json.dumps(evaluator_sanity, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    endpoint_snapshot = await endpoint_preflight(args)
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
        started_at = partial["started_at"]
    else:
        started_at = utc_now()
        partial = {
            "manifest_schema_version": 1,
            "run_id": args.run_id,
            "started_at": started_at,
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
        (args.run_id, args.model, task_id, variant, seed)
        for task_id in problems
        for seed in args.seeds
        for variant in VARIANT_SPECS
    }
    missing = sorted(expected - set(store.records))
    if missing:
        print(f"Partial run preserved at {run_dir}; {len(missing)} records are still missing")
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
        print("Interrupted; manifest.partial.json and completed JSONL records were preserved", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())

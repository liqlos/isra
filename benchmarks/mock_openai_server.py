#!/usr/bin/env python3
"""Deterministic OpenAI-compatible fake used only to smoke-test the harness."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from aiohttp import web


def make_app(model: str, scenario: str, capture: Path | None) -> web.Application:
    from evalplus.data import get_human_eval_plus

    problems = get_human_eval_plus()
    attempts: Counter[tuple[str, str, str]] = Counter()

    async def models(_: web.Request) -> web.Response:
        return web.json_response(
            {"object": "list", "data": [{"id": model, "object": "model", "created": int(time.time())}]}
        )

    async def health(_: web.Request) -> web.Response:
        return web.json_response(
            {
                "status": "ok",
                "service": "mock-isra-orchestrator",
                "backend_model": model,
                "config": {
                    "benchmark_temperature_override": True,
                    "phase_metrics": True,
                    "code_pipeline_variants": [
                        "legacy_isra",
                        "phase1_only",
                        "phase1_unguided_retry",
                        "grounded_one_repair",
                    ],
                    "frozen_initial_candidate": True,
                    "trusted_evidence_rollback": True,
                },
            }
        )

    async def completions(request: web.Request) -> web.Response:
        body = await request.json()
        variant = request.headers.get("X-Benchmark-Variant", "direct_greedy")
        pipeline_variant = body.get("isra_variant")
        task_id = request.headers.get("X-Benchmark-Task-Id", "")
        seed = str(body.get("seed", 0))
        if task_id not in problems:
            return web.json_response({"error": {"message": f"unknown task: {task_id}"}}, status=400)
        key = (task_id, variant, seed)
        attempts[key] += 1
        if scenario == "paired-disagreements" and task_id == "HumanEval/2" and attempts[key] == 1:
            return web.json_response({"error": {"message": "synthetic overload"}}, status=503)

        problem = problems[task_id]
        correct_solution = problem["prompt"] + problem["canonical_solution"]
        wrong_solution = problem["prompt"] + "    raise NotImplementedError('synthetic wrong answer')\n"
        correct_content = f"```python\n{correct_solution}\n```"
        wrong_content = f"```python\n{wrong_solution}\n```"
        trace = None
        incremental_calls = None

        if scenario == "mechanism-fanout" and pipeline_variant:
            parent_is_wrong = task_id in {"HumanEval/0", "HumanEval/1"}
            if pipeline_variant == "phase1_only":
                content = wrong_content if parent_is_wrong else correct_content
                incremental_calls = 1
                trace = {
                    "candidate_a": content,
                    "candidate_b": None,
                    "selected": "A",
                    "selection_reason": "phase1_only",
                    "provided_initial_candidate": False,
                }
            else:
                initial = body.get("isra_initial_candidate")
                if not isinstance(initial, str):
                    return web.json_response(
                        {"error": {"message": "missing frozen initial candidate"}}, status=400
                    )
                should_improve = (
                    (task_id == "HumanEval/0" and pipeline_variant == "grounded_one_repair")
                    or (task_id == "HumanEval/1" and pipeline_variant == "phase1_unguided_retry")
                )
                attempted = parent_is_wrong
                content = correct_content if should_improve else initial
                incremental_calls = 1 if attempted else 0
                trace = {
                    "candidate_a": initial,
                    "candidate_b": correct_content if should_improve else (initial if attempted else None),
                    "selected": "B" if should_improve else "A",
                    "selection_reason": "synthetic_improvement" if should_improve else "synthetic_rollback",
                    "provided_initial_candidate": True,
                }
        else:
            wrong = (
                scenario == "paired-disagreements"
                and (
                    (task_id == "HumanEval/0" and variant.startswith("isra_"))
                    or (task_id == "HumanEval/1" and variant.startswith("direct_"))
                )
            )
            content = wrong_content if wrong else correct_content

        messages = body.get("messages", [])
        if capture:
            capture.parent.mkdir(parents=True, exist_ok=True)
            with capture.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "task_id": task_id,
                            "variant": variant,
                            "pipeline_variant": pipeline_variant,
                            "seed": seed,
                            "messages": messages,
                            "initial_candidate": body.get("isra_initial_candidate"),
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )

        prompt_tokens = sum(len(str(message.get("content", "")).split()) for message in messages)
        completion_tokens = len(content.split())
        is_isra = bool(pipeline_variant) or variant.startswith("isra_")
        llm_calls = incremental_calls if incremental_calls is not None else (3 if is_isra else 1)
        phase_metrics = []
        if is_isra:
            greedy = body.get("isra_temperature_override") == 0.0
            temperatures = ([0.0] * llm_calls) if greedy else ([0.6, 0.2, 0.1][:llm_calls])
            phase_metrics = [
                {
                    "call_index": index,
                    "phase": phase,
                    "temperature": temperature,
                    "latency_ms": 1.0,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "finish_reason": "stop",
                    "status": "completed",
                }
                for index, (phase, temperature) in enumerate(zip(range(1, llm_calls + 1), temperatures), start=1)
            ]
            prompt_tokens *= llm_calls
            completion_tokens *= llm_calls
        payload = {
            "id": f"mock-{task_id.replace('/', '-')}-{variant}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
        if is_isra:
            payload["isra_metadata"] = {
                "llm_calls": llm_calls,
                "phase_metrics": phase_metrics,
                "termination_reason": pipeline_variant or "MOCK_PIPELINE",
                "candidate_trace": trace,
            }
        return web.json_response(payload)

    app = web.Application()
    app.router.add_get("/v1/models", models)
    app.router.add_get("/health", health)
    app.router.add_post("/v1/chat/completions", completions)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--model", default="mock-model")
    parser.add_argument(
        "--scenario",
        choices=("parity", "paired-disagreements", "mechanism-fanout"),
        default="parity",
    )
    parser.add_argument("--capture", type=Path)
    args = parser.parse_args()
    web.run_app(make_app(args.model, args.scenario, args.capture), host=args.host, port=args.port)


if __name__ == "__main__":
    main()

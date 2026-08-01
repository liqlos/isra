#!/usr/bin/env python3
"""Deterministic direct-versus-SPA OpenAI-compatible fake for harness tests."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from aiohttp import web


def make_app(model: str, capture: Path | None) -> web.Application:
    from evalplus.data import get_human_eval_plus

    problems = get_human_eval_plus()

    async def models(_: web.Request) -> web.Response:
        return web.json_response(
            {
                "object": "list",
                "data": [
                    {"id": f"{model}-direct", "object": "model", "created": int(time.time())},
                    {"id": f"{model}-spa", "object": "model", "created": int(time.time())},
                ],
            }
        )

    async def health(_: web.Request) -> web.Response:
        return web.json_response(
            {
                "status": "ok",
                "service": "mock-anchored-generation",
                "model_id": model,
                "config": {
                    "decoding_modes": ["direct", "spa"],
                    "anchor_modes": ["natural_language", "last_user"],
                    "default_strength": 1.28,
                    "default_anchor_mode": "natural_language",
                    "mask_token": "<|finetune_right_pad_id|>",
                    "mask_token_id": 128004,
                    "hidden_tests_exposed": False,
                    "answer_candidates": 1,
                    "llm_calls": 1,
                },
            }
        )

    async def completions(request: web.Request) -> web.Response:
        body = await request.json()
        task_id = request.headers.get("X-Benchmark-Task-Id", "")
        mode = str(body.get("decoding_mode", "direct"))
        if task_id not in problems:
            return web.json_response({"error": {"message": f"unknown task: {task_id}"}}, status=400)
        if mode not in {"direct", "spa"}:
            return web.json_response({"error": {"message": f"unknown mode: {mode}"}}, status=400)

        problem = problems[task_id]
        correct = str(problem["prompt"]) + str(problem["canonical_solution"])
        wrong = str(problem["prompt"]) + "    raise NotImplementedError('synthetic wrong answer')\n"
        wrong_mode = (
            (task_id == "HumanEval/20" and mode == "direct")
            or (task_id == "HumanEval/21" and mode == "spa")
        )
        content = wrong if wrong_mode else correct
        messages = body.get("messages", [])

        if capture:
            capture.parent.mkdir(parents=True, exist_ok=True)
            with capture.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "task_id": task_id,
                            "mode": mode,
                            "seed": body.get("seed", 0),
                            "messages": messages,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )

        prompt_tokens = sum(len(str(message.get("content", "")).split()) for message in messages)
        completion_tokens = len(content.split())
        return web.json_response(
            {
                "id": f"mock-{task_id.replace('/', '-')}-{mode}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": f"{model}-{mode}",
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
                "decoding_metadata": {
                    "algorithm": "selective_prompt_anchoring" if mode == "spa" else "direct",
                    "anchoring": {
                        "enabled": mode == "spa",
                        "strength": body.get("anchoring_strength", 1.28),
                        "anchor_mode": body.get("anchor_mode", "natural_language"),
                        "mask_token": "<|finetune_right_pad_id|>",
                    },
                    "forward_batch_size": 2 if mode == "spa" else 1,
                    "masked_prompt_tokens": 12 if mode == "spa" else 0,
                    "llm_calls": 1,
                    "peak_memory_gb": 0.1,
                },
            }
        )

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
    parser.add_argument("--capture", type=Path)
    args = parser.parse_args()
    web.run_app(make_app(args.model, args.capture), host=args.host, port=args.port)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

import pytest
from aiohttp import web

from benchmarks.mechanism_benchmark import (
    ALL_VARIANTS,
    branch_order,
    execute_mechanism_run,
)
from benchmarks.paired_core import ResultStore, sha256_text


async def start_server(handler):
    app = web.Application()
    app.router.add_post("/v1/chat/completions", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    return runner, f"http://127.0.0.1:{port}/v1/chat/completions"


def test_branch_order_is_stable():
    assert branch_order("HumanEval/0", 7) == branch_order("HumanEval/0", 7)
    assert set(branch_order("HumanEval/0", 7)) == {
        "phase1_unguided_retry",
        "grounded_one_repair",
    }


@pytest.mark.asyncio
async def test_mechanism_runner_fans_out_exact_candidate_and_accounts_parent_cost(
    tmp_path: Path,
):
    frozen = "def f(x):\n    return x"
    repaired = "def f(x):\n    return x + 1"
    captured = []

    async def handler(request):
        body = await request.json()
        captured.append(body)
        variant = body["isra_variant"]
        if variant == "phase1_only":
            content = frozen
            trace = {
                "candidate_a": frozen,
                "selected": "A",
                "provided_initial_candidate": False,
            }
            calls = 1
            prompt_tokens = 10
            completion_tokens = 5
        else:
            assert body["isra_initial_candidate"] == frozen
            content = repaired if variant == "grounded_one_repair" else frozen
            trace = {
                "candidate_a": frozen,
                "candidate_b": content,
                "selected": "B" if content == repaired else "A",
                "provided_initial_candidate": True,
            }
            calls = 1
            prompt_tokens = 7
            completion_tokens = 3
        return web.json_response(
            {
                "choices": [
                    {
                        "message": {"content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                },
                "isra_metadata": {
                    "llm_calls": calls,
                    "phase_metrics": [{"role": variant}],
                    "termination_reason": variant,
                    "candidate_trace": trace,
                },
            }
        )

    class Evaluator:
        def evaluate(self, task_id, response):
            passed = "x + 1" in response
            return {"passed": passed, "message": "PASS" if passed else "FAIL"}

    runner, url = await start_server(handler)
    args = argparse.Namespace(
        run_id="mechanism-run",
        model="model",
        isra_model="isra",
        isra_url=url,
        api_key="test",
        proposer_temperature=0.0,
        repair_temperature=0.0,
        top_p=0.95,
        max_tokens=100,
        timeout=2,
        retries=0,
        retry_backoff=0,
        seeds=[7],
    )
    problems = {"HumanEval/0": {"prompt": "def f(x):\n    pass\n"}}
    store = ResultStore(tmp_path / "results.jsonl")
    try:
        await execute_mechanism_run(args, problems, store, Evaluator())
    finally:
        await runner.cleanup()

    assert len(captured) == 3
    assert {record["variant"] for record in store.records.values()} == set(ALL_VARIANTS)
    parent = next(record for record in store.records.values() if record["variant"] == "phase1_only")
    grounded = next(
        record for record in store.records.values() if record["variant"] == "grounded_one_repair"
    )
    retry = next(
        record for record in store.records.values() if record["variant"] == "phase1_unguided_retry"
    )

    assert parent["response"] == frozen
    assert grounded["response"] == repaired
    assert grounded["parent_candidate_sha256"] == sha256_text(frozen)
    assert retry["parent_candidate_sha256"] == sha256_text(frozen)
    assert grounded["llm_calls"] == 2
    assert grounded["incremental_llm_calls"] == 1
    assert grounded["prompt_tokens"] == 17
    assert grounded["completion_tokens"] == 8
    assert grounded["passed"] is True
    assert retry["passed"] is False


@pytest.mark.asyncio
async def test_mechanism_runner_records_skipped_branches_when_parent_has_no_text(
    tmp_path: Path,
):
    calls = 0

    async def handler(request):
        nonlocal calls
        calls += 1
        return web.json_response(
            {
                "choices": [{"message": {"content": ""}, "finish_reason": "length"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20},
            }
        )

    class Evaluator:
        def evaluate(self, task_id, response):
            raise AssertionError("an unavailable candidate must not be evaluated")

    runner, url = await start_server(handler)
    args = argparse.Namespace(
        run_id="parent-failure-run",
        model="model",
        isra_model="isra",
        isra_url=url,
        api_key="test",
        proposer_temperature=0.0,
        repair_temperature=0.0,
        top_p=0.95,
        max_tokens=100,
        timeout=2,
        retries=0,
        retry_backoff=0,
        seeds=[0],
    )
    problems = {"HumanEval/0": {"prompt": "def f(x):\n    pass\n"}}
    store = ResultStore(tmp_path / "results.jsonl")
    try:
        await execute_mechanism_run(args, problems, store, Evaluator())
    finally:
        await runner.cleanup()

    assert calls == 1
    records = list(store.records.values())
    parent = next(record for record in records if record["variant"] == "phase1_only")
    branches = [record for record in records if record["variant"] != "phase1_only"]
    assert parent["status"] == "model_error"
    assert len(branches) == 2
    assert {record["status"] for record in branches} == {"skipped_parent_unavailable"}
    assert all(record["latency_ms"] == parent["latency_ms"] for record in branches)
    assert all(record["incremental_llm_calls"] == 0 for record in branches)

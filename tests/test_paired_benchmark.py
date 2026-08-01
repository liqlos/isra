from __future__ import annotations

import argparse
import json
from pathlib import Path

import aiohttp
import pytest
from aiohttp import web

from benchmarks.paired_benchmark import execute_run, manifest_configuration
from benchmarks.paired_core import (
    EvalPlusHumanEvalEvaluator,
    VARIANT_SPECS,
    ResultStore,
    call_openai_endpoint,
    categorize_http_status,
    exact_mcnemar_p,
    extract_code,
    extract_numeric_answer,
    primary_key,
    randomized_variant_order,
    task_messages,
    write_json_exclusive,
)


def result_record(**overrides):
    record = {
        "run_id": "run-1",
        "task_id": "HumanEval/0",
        "model": "model",
        "variant": "direct_greedy",
        "seed": 0,
        "status": "completed",
        "passed": True,
        "response": "ok",
        "latency_ms": 1,
        "prompt_tokens": 1,
        "completion_tokens": 1,
        "llm_calls": 1,
        "phase_metrics": [],
        "termination_reason": "direct",
        "evaluator_message": "PASS",
    }
    record.update(overrides)
    return record


def test_code_extraction_prefers_entry_point_and_handles_fallback():
    text = """analysis
```python
def unrelated():
    return 0
```
```Python
from typing import List

def target(xs: List[int]):
    return sum(xs)
```
"""
    code = extract_code(text, "target")
    assert "def target" in code
    assert "def unrelated" not in code
    assert extract_code("Explanation\n\ndef target():\n    return 1", "target").startswith("def target")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("work\n#### -1,234.50", "-1234.5"),
        (r"answer is \boxed{42}", "42"),
        ("The final answer is $0.125.", "0.125"),
        ("steps: 2 then 7", "7"),
        ("no numeric answer", None),
    ],
)
def test_numeric_answer_extraction(text, expected):
    assert extract_numeric_answer(text) == expected


def test_error_categorization_separates_infrastructure_from_model_errors():
    assert categorize_http_status(503) == "transport_error"
    assert categorize_http_status(429) == "transport_error"
    assert categorize_http_status(408) == "transport_error"
    assert categorize_http_status(400) == "model_error"
    assert categorize_http_status(404) == "model_error"


def test_result_store_resume_and_immutable_write(tmp_path: Path):
    path = tmp_path / "results.jsonl"
    first = result_record()
    store = ResultStore(path)
    store.append(first)
    resumed = ResultStore(path)
    assert resumed.has(primary_key(first))
    with pytest.raises(ValueError, match="already exists"):
        resumed.append(first)

    manifest = tmp_path / "manifest.json"
    write_json_exclusive(manifest, {"run_id": "run-1"})
    with pytest.raises(FileExistsError):
        write_json_exclusive(manifest, {"run_id": "run-2"})


def test_result_store_ignores_only_incomplete_trailing_line(tmp_path: Path):
    path = tmp_path / "results.jsonl"
    path.write_text(json.dumps(result_record()) + "\n" + '{"run_id":', encoding="utf-8")
    assert len(ResultStore(path).records) == 1


def test_variant_order_is_stable_and_complete():
    first = randomized_variant_order("HumanEval/7", 42)
    second = randomized_variant_order("HumanEval/7", 42)
    assert first == second
    assert set(first) == set(VARIANT_SPECS)


def test_manifest_generation_has_required_provenance():
    args = argparse.Namespace(
        model="model",
        model_revision="revision",
        quantization="4-bit",
        weights_fingerprint="sha256:abc",
        direct_url="http://direct/v1/chat/completions",
        isra_url="http://isra/v1/chat/completions",
        backend_name="mock",
        backend_versions="mock=1",
        backend_machine="test machine",
        max_tokens=2000,
        seeds=[0, 1],
        timeout=60.0,
        retries=2,
        retry_backoff=0.5,
    )
    problems = {"HumanEval/0": {"prompt": "def f():\n    pass\n"}}
    dataset = {
        "name": "HumanEval+",
        "version": "v0.1.10",
        "content_md5": "abc",
        "task_ids": ["HumanEval/0"],
    }
    manifest = manifest_configuration(args, problems, dataset, {"ok": True})
    assert manifest["model"]["weights_revision"] == "revision"
    assert manifest["dataset"]["content_md5"] == "abc"
    assert set(manifest["variants"]) == set(VARIANT_SPECS)
    assert manifest["retry_policy"]["infrastructure_only"] is True
    assert manifest["orchestrator"]["prompt_sha256"]


async def start_server(handler):
    app = web.Application()
    app.router.add_post("/v1/chat/completions", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    return runner, f"http://127.0.0.1:{port}/v1/chat/completions"


@pytest.mark.asyncio
async def test_retry_only_for_infrastructure_failure():
    attempts = 0

    async def handler(_):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return web.json_response({"error": "overloaded"}, status=503)
        return web.json_response(
            {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}
        )

    runner, url = await start_server(handler)
    try:
        async with aiohttp.ClientSession() as session:
            outcome = await call_openai_endpoint(
                session,
                url=url,
                body={"messages": []},
                headers={},
                timeout_seconds=1,
                retries=2,
                backoff_seconds=0,
            )
        assert outcome.status == "completed"
        assert outcome.attempts == 2
        assert outcome.retry_history[0]["status"] == "transport_error"
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_model_error_is_not_retried():
    attempts = 0

    async def handler(_):
        nonlocal attempts
        attempts += 1
        return web.json_response({"error": "bad model"}, status=400)

    runner, url = await start_server(handler)
    try:
        async with aiohttp.ClientSession() as session:
            outcome = await call_openai_endpoint(
                session,
                url=url,
                body={"messages": []},
                headers={},
                timeout_seconds=1,
                retries=3,
                backoff_seconds=0,
            )
        assert outcome.status == "model_error"
        assert outcome.attempts == 1
        assert attempts == 1
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_empty_success_retains_payload_for_cost_accounting():
    payload = {
        "choices": [{"message": {"content": ""}, "finish_reason": "length"}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 23},
        "isra_metadata": {"llm_calls": 1, "phase_metrics": [{"role": "proposer"}]},
    }

    async def handler(_):
        return web.json_response(payload)

    runner, url = await start_server(handler)
    try:
        async with aiohttp.ClientSession() as session:
            outcome = await call_openai_endpoint(
                session,
                url=url,
                body={"messages": []},
                headers={},
                timeout_seconds=1,
                retries=2,
                backoff_seconds=0,
            )
        assert outcome.status == "model_error"
        assert outcome.payload == payload
        assert outcome.finish_reason == "length"
        assert outcome.attempts == 1
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_evaluator_and_task_content_are_identical_for_all_variants(tmp_path: Path):
    captured_messages = []

    async def handler(request):
        body = await request.json()
        captured_messages.append(body["messages"])
        variant = request.headers["X-Benchmark-Variant"]
        payload = {
            "choices": [{"message": {"content": "```python\ndef f():\n    return 1\n```"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 3},
        }
        if variant.startswith("isra_"):
            payload["isra_metadata"] = {
                "llm_calls": 2,
                "phase_metrics": [{"phase": 1}],
                "termination_reason": "test",
            }
        return web.json_response(payload)

    class RecordingEvaluator:
        def __init__(self):
            self.calls = []

        def evaluate(self, task_id, response):
            self.calls.append((task_id, response))
            return {"passed": True, "message": "same evaluator"}

    runner, url = await start_server(handler)
    args = argparse.Namespace(
        run_id="run-1",
        model="model",
        isra_model="isra",
        direct_url=url,
        isra_url=url,
        api_key="test",
        timeout=1,
        retries=0,
        retry_backoff=0,
        max_tokens=100,
        seeds=[0],
    )
    problems = {"HumanEval/0": {"prompt": "def f():\n    pass\n"}}
    evaluator = RecordingEvaluator()
    store = ResultStore(tmp_path / "results.jsonl")
    try:
        await execute_run(args, problems, store, evaluator)
    finally:
        await runner.cleanup()
    assert len(evaluator.calls) == 4
    assert len(captured_messages) == 4
    assert all(messages == task_messages(problems["HumanEval/0"]) for messages in captured_messages)
    assert {record["evaluator_message"] for record in store.records.values()} == {"same evaluator"}


def test_exact_mcnemar():
    assert exact_mcnemar_p(0, 0) == 1.0
    assert exact_mcnemar_p(5, 0) == pytest.approx(0.0625)


def test_evaluator_sanity_checks_oracle_and_known_bad_mutation():
    evaluator = object.__new__(EvalPlusHumanEvalEvaluator)
    evaluator.problems = {
        "HumanEval/0": {
            "prompt": "def f():\n    \"\"\"Return one.\"\"\"\n",
            "canonical_solution": "    return 1\n",
        }
    }

    def fake_evaluate(task_id, response):
        passed = "return 1" in response and "intentional evaluator sanity mutation" not in response
        return {"passed": passed, "message": "PASS" if passed else "FAIL"}

    evaluator.evaluate = fake_evaluate
    result = evaluator.sanity_check()

    assert result["status"] == "passed"
    assert result["canonical_solutions_passed"] == 1
    assert result["known_bad_mutations_rejected"] == 1

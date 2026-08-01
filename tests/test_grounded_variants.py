from __future__ import annotations

import pytest

import isra_orchestrator as orchestrator


TASK = '''Complete this function:

def increment(x: int) -> int:
    """Return x plus one.

    >>> increment(1)
    2
    >>> increment(-1)
    0
    """
'''

CORRECT = "def increment(x: int) -> int:\n    return x + 1"
WRONG = "def increment(x: int) -> int:\n    return x"


def test_trusted_checks_use_only_public_examples():
    correct = orchestrator.verify_code_with_trusted_checks(TASK, CORRECT)
    wrong = orchestrator.verify_code_with_trusted_checks(TASK, WRONG)

    assert correct["passed"] is True
    assert correct["semantic_verified"] is True
    assert correct["public_tests_total"] == 2
    assert wrong["passed"] is False
    assert wrong["actionable_failure"] is True
    assert wrong["failed_test_ids"] == ["public_example:0", "public_example:1"]
    assert all(failure["kind"] == "public_example_failure" for failure in wrong["failures"])


def test_compile_only_pass_is_not_semantic_verification():
    task_without_examples = "Complete this function:\n\ndef f(x):\n    pass"
    checked = orchestrator.verify_code_with_trusted_checks(
        task_without_examples, "def f(x):\n    return x"
    )

    assert checked["passed"] is True
    assert checked["semantic_verified"] is False
    assert checked["actionable_failure"] is False


def test_trusted_comparison_requires_strict_non_regressing_improvement():
    original = orchestrator.verify_code_with_trusted_checks(TASK, WRONG)
    repaired = orchestrator.verify_code_with_trusted_checks(TASK, CORRECT)
    still_wrong = orchestrator.verify_code_with_trusted_checks(
        TASK, "def increment(x: int) -> int:\n    return x - 1"
    )

    assert orchestrator.compare_trusted_verification(original, repaired) == (
        True,
        "trusted_suite_passed",
    )
    accepted, reason = orchestrator.compare_trusted_verification(original, still_wrong)
    assert accepted is False
    assert reason == "no_strict_trusted_improvement"

    partially_repaired = {
        **repaired,
        "passed": False,
        "semantic_verified": False,
        "failed_test_ids": ["public_example:1"],
        "failures": [{"id": "public_example:1"}],
    }
    assert orchestrator.compare_trusted_verification(original, partially_repaired) == (
        False,
        "trusted_failures_reduced_but_not_cleared",
    )


def test_role_seeds_are_stable_and_distinct():
    proposer = orchestrator.derive_role_seed(17, "proposer")
    repair = orchestrator.derive_role_seed(17, "grounded_repair")

    assert proposer == orchestrator.derive_role_seed(17, "proposer")
    assert proposer != repair
    assert orchestrator.derive_role_seed(None, "proposer") is None


@pytest.mark.asyncio
async def test_grounded_variant_skips_second_call_without_trusted_failure(monkeypatch):
    calls = []

    async def fake_call_backend(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("no model call should be made for a provided passing candidate")

    monkeypatch.setattr(orchestrator, "call_backend", fake_call_backend)
    result = await orchestrator._run_code_pipeline_variant_impl(
        None,
        TASK,
        variant="grounded_one_repair",
        initial_candidate=CORRECT,
        seed=1,
        repair_temperature=0.1,
        stream_final=False,
        write_chunk=None,
    )

    assert calls == []
    assert result["final_answer"] == CORRECT
    assert result["termination_reason"] == "NO_TRUSTED_FAILURE"
    assert result["candidate_trace"]["selected"] == "A"


@pytest.mark.asyncio
async def test_grounded_variant_accepts_evidence_backed_repair(monkeypatch):
    calls = []

    async def fake_call_backend(session, system_prompt, user_prompt, **kwargs):
        calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                **kwargs,
            }
        )
        return f"```python\n{CORRECT}\n```"

    monkeypatch.setattr(orchestrator, "call_backend", fake_call_backend)
    result = await orchestrator._run_code_pipeline_variant_impl(
        None,
        TASK,
        variant="grounded_one_repair",
        initial_candidate=WRONG,
        seed=7,
        repair_temperature=0.2,
        stream_final=False,
        write_chunk=None,
    )

    assert len(calls) == 1
    assert calls[0]["metric_role"] == "grounded_repair"
    assert calls[0]["role_temperature_override"] == 0.2
    assert "TRUSTED FAILURE EVIDENCE" in calls[0]["user_prompt"]
    assert "public_example:0" in calls[0]["user_prompt"]
    assert result["final_answer"] == CORRECT
    assert result["termination_reason"] == "REPAIR_ACCEPTED"
    assert result["candidate_trace"]["selected"] == "B"


@pytest.mark.asyncio
async def test_grounded_variant_rolls_back_non_improving_repair(monkeypatch):
    async def fake_call_backend(*args, **kwargs):
        return "```python\ndef increment(x: int) -> int:\n    return x - 1\n```"

    monkeypatch.setattr(orchestrator, "call_backend", fake_call_backend)
    result = await orchestrator._run_code_pipeline_variant_impl(
        None,
        TASK,
        variant="grounded_one_repair",
        initial_candidate=WRONG,
        seed=3,
        repair_temperature=0.0,
        stream_final=False,
        write_chunk=None,
    )

    assert result["final_answer"] == WRONG
    assert result["termination_reason"] == "REPAIR_ROLLBACK"
    assert result["candidate_trace"]["selected"] == "A"
    assert result["candidate_trace"]["selection_reason"] == "no_strict_trusted_improvement"


@pytest.mark.asyncio
async def test_unguided_retry_receives_no_candidate_or_feedback(monkeypatch):
    calls = []

    async def fake_call_backend(session, system_prompt, user_prompt, **kwargs):
        calls.append((system_prompt, user_prompt, kwargs))
        return f"```python\n{CORRECT}\n```"

    monkeypatch.setattr(orchestrator, "call_backend", fake_call_backend)
    result = await orchestrator._run_code_pipeline_variant_impl(
        None,
        TASK,
        variant="phase1_unguided_retry",
        initial_candidate=WRONG,
        seed=11,
        repair_temperature=None,
        stream_final=False,
        write_chunk=None,
    )

    assert len(calls) == 1
    _, prompt, kwargs = calls[0]
    assert prompt == orchestrator.phase1_first_turn_prompt(TASK)
    assert WRONG not in prompt
    assert "TRUSTED FAILURE" not in prompt
    assert kwargs["metric_role"] == "unguided_retry"
    assert result["final_answer"] == CORRECT
    assert result["termination_reason"] == "RETRY_ACCEPTED"

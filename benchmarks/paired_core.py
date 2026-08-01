#!/usr/bin/env python3
"""Shared, testable primitives for paired one-pass code benchmarks."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import platform
import random
import re
import time
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable

import aiohttp


RESULT_SCHEMA_VERSION = 1
RESULT_STATUSES = {
    "completed",
    "model_error",
    "timeout",
    "transport_error",
    "evaluator_error",
}
INFRASTRUCTURE_STATUSES = {"timeout", "transport_error"}

TASK_SYSTEM_MESSAGE = (
    "Complete the Python function. Return only executable Python code containing "
    "the complete function and any required imports."
)
TASK_USER_PREFIX = "Complete this function:\n\n"


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_text(canonical_json(value))


def task_messages(task: dict[str, Any]) -> list[dict[str, str]]:
    """Return the exact same external task messages for every variant."""
    return [
        {"role": "system", "content": TASK_SYSTEM_MESSAGE},
        {"role": "user", "content": f"{TASK_USER_PREFIX}{task['prompt']}"},
    ]


def primary_key(record: dict[str, Any]) -> tuple[str, str, str, str, int]:
    return (
        str(record["run_id"]),
        str(record["model"]),
        str(record["task_id"]),
        str(record["variant"]),
        int(record["seed"]),
    )


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(
    r"```[ \t]*(?:python|py)?[ \t]*\r?\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)


def extract_code(text: str, entry_point: str | None = None) -> str:
    """Extract one complete Python candidate without using hidden tests."""
    if not text:
        return ""
    visible = _THINK_RE.sub("", text).strip()
    blocks = [block.strip() for block in _FENCE_RE.findall(visible) if block.strip()]
    if blocks:
        if entry_point:
            entry_re = re.compile(rf"\bdef\s+{re.escape(entry_point)}\s*\(")
            matching = [block for block in blocks if entry_re.search(block)]
            if matching:
                return matching[-1]
        return blocks[-1]

    visible = re.sub(r"^\s*\[/?CODE\]\s*$", "", visible, flags=re.MULTILINE | re.IGNORECASE)
    lines = visible.splitlines()
    start = None
    for index, line in enumerate(lines):
        if line.lstrip().startswith(("def ", "async def ", "class ", "import ", "from ", "@")):
            start = index
            break
    if start is not None:
        candidate = "\n".join(lines[start:]).strip()
        candidate = re.split(r"\n\s*(?:\[CONCLUSIONS\]|Explanation:|The code above)", candidate, maxsplit=1)[0]
        return candidate.strip()
    return visible


_NUMBER = r"[-+]?\d[\d,]*(?:\.\d+)?"


def _normalize_number(raw: str) -> str | None:
    try:
        value = Decimal(raw.replace(",", ""))
    except (InvalidOperation, ValueError):
        return None
    if value == value.to_integral():
        return str(value.quantize(Decimal(1)))
    return format(value.normalize(), "f")


def extract_numeric_answer(text: str) -> str | None:
    """Extract GSM-style numeric answers, including signs and decimals."""
    patterns = (
        rf"####\s*({_NUMBER})",
        rf"\\boxed\{{\s*({_NUMBER})\s*\}}",
        rf"(?:final\s+answer|answer|result)\s*(?:is|=|:)\s*\$?\s*({_NUMBER})",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return _normalize_number(match.group(1))
    matches = re.findall(_NUMBER, text)
    return _normalize_number(matches[-1]) if matches else None


def prepare_humaneval_solution(code: str, problem: dict[str, Any]) -> str:
    """Support both full-function answers and official body-only completions."""
    code = code.strip()
    if not code:
        return ""
    entry_point = str(problem["entry_point"])
    if re.search(rf"\bdef\s+{re.escape(entry_point)}\s*\(", code):
        prompt_imports = [
            line
            for line in str(problem["prompt"]).splitlines()
            if line.startswith(("import ", "from "))
        ]
        missing = [line for line in prompt_imports if line not in code]
        return ("\n".join(missing) + "\n" + code).strip() if missing else code
    return str(problem["prompt"]) + code


def extract_response_text(payload: dict[str, Any]) -> tuple[str, str | None]:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return "", None
    choice = choices[0] if isinstance(choices[0], dict) else {}
    message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
    content = message.get("content") or ""
    reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
    if content:
        return str(content), choice.get("finish_reason")
    return re.sub(r"</?think>", "", str(reasoning), flags=re.IGNORECASE).strip(), choice.get("finish_reason")


def categorize_http_status(http_status: int) -> str:
    if http_status in {408, 425, 429} or http_status >= 500:
        return "transport_error"
    return "model_error"


@dataclass
class EndpointOutcome:
    status: str
    response: str
    payload: dict[str, Any] | None
    latency_ms: float
    attempts: int
    retry_history: list[dict[str, Any]]
    error: str | None = None
    finish_reason: str | None = None


async def call_openai_endpoint(
    session: aiohttp.ClientSession,
    *,
    url: str,
    body: dict[str, Any],
    headers: dict[str, str],
    timeout_seconds: float,
    retries: int,
    backoff_seconds: float,
) -> EndpointOutcome:
    """Call an endpoint with retries restricted to infrastructure failures."""
    overall_start = time.perf_counter()
    retry_history: list[dict[str, Any]] = []
    final_status = "transport_error"
    final_error = "request did not run"
    final_payload: dict[str, Any] | None = None
    final_finish_reason: str | None = None

    for attempt in range(1, retries + 2):
        attempt_start = time.perf_counter()
        payload: dict[str, Any] | None = None
        response_text = ""
        finish_reason = None
        try:
            timeout = aiohttp.ClientTimeout(total=timeout_seconds)
            async with session.post(url, json=body, headers=headers, timeout=timeout) as response:
                raw = await response.text()
                if response.status != 200:
                    final_status = categorize_http_status(response.status)
                    final_error = f"HTTP {response.status}: {raw[:500]}"
                else:
                    try:
                        decoded = json.loads(raw)
                    except json.JSONDecodeError as exc:
                        final_status = "transport_error"
                        final_error = f"invalid JSON response: {exc}"
                    else:
                        if not isinstance(decoded, dict):
                            final_status = "transport_error"
                            final_error = "response JSON is not an object"
                        else:
                            payload = decoded
                            response_text, finish_reason = extract_response_text(decoded)
                            final_payload = payload
                            final_finish_reason = finish_reason
                            if response_text:
                                return EndpointOutcome(
                                    status="completed",
                                    response=response_text,
                                    payload=payload,
                                    latency_ms=round((time.perf_counter() - overall_start) * 1000, 3),
                                    attempts=attempt,
                                    retry_history=retry_history,
                                    finish_reason=finish_reason,
                                )
                            final_status = "model_error"
                            final_error = "successful response contained no model text"
        except (asyncio.TimeoutError, TimeoutError) as exc:
            final_status = "timeout"
            final_error = f"{type(exc).__name__}: request exceeded {timeout_seconds}s"
        except aiohttp.ClientError as exc:
            final_status = "transport_error"
            final_error = f"{type(exc).__name__}: {exc}"

        attempt_ms = round((time.perf_counter() - attempt_start) * 1000, 3)
        retry_history.append(
            {"attempt": attempt, "status": final_status, "latency_ms": attempt_ms, "error": final_error}
        )
        if final_status not in INFRASTRUCTURE_STATUSES or attempt > retries:
            break
        await asyncio.sleep(backoff_seconds * (2 ** (attempt - 1)))

    return EndpointOutcome(
        status=final_status,
        response="",
        payload=final_payload,
        latency_ms=round((time.perf_counter() - overall_start) * 1000, 3),
        attempts=len(retry_history),
        retry_history=retry_history,
        error=final_error,
        finish_reason=final_finish_reason,
    )


class EvalPlusHumanEvalEvaluator:
    """Official EvalPlus execution and oracle, shared by every variant."""

    VERSION = "evalplus-0.3.1/HumanEvalPlus-v0.1.10"

    def __init__(self, problems: dict[str, dict[str, Any]], dataset_hash: str):
        from evalplus.evaluate import get_groundtruth

        self.problems = problems
        smoke_hash = sha256_text(dataset_hash + "\0" + "\0".join(problems))[:32]
        self.expected = get_groundtruth(problems, smoke_hash, [])

    def evaluate(self, task_id: str, response: str) -> dict[str, Any]:
        from evalplus.eval import PASS
        from evalplus.evaluate import check_correctness

        problem = self.problems[task_id]
        code = extract_code(response, str(problem["entry_point"]))
        solution = prepare_humaneval_solution(code, problem)
        if not solution:
            return {
                "passed": False,
                "message": "No executable Python code found",
                "base_status": "fail",
                "plus_status": "fail",
            }
        result = check_correctness(
            "humaneval",
            0,
            problem,
            solution,
            self.expected[task_id],
            base_only=False,
            fast_check=False,
            identifier=task_id,
        )
        base_status, base_details = result["base"]
        plus_status, plus_details = result["plus"]
        base_failed = sum(not item for item in base_details)
        plus_failed = sum(not item for item in plus_details)
        return {
            "passed": base_status == PASS and plus_status == PASS,
            "message": (
                f"base={base_status} ({base_failed}/{len(base_details)} failed); "
                f"plus={plus_status} ({plus_failed}/{len(plus_details)} failed)"
            ),
            "base_status": base_status,
            "plus_status": plus_status,
            "extracted_code_sha256": sha256_text(solution),
        }

    def sanity_check(self) -> dict[str, Any]:
        """Prove the evaluator accepts its oracle and rejects a known-bad body.

        This runs before any model request so evaluator/runtime failures cannot
        be recorded as ordinary model failures.
        """
        started = time.perf_counter()
        canonical_failures: list[dict[str, str]] = []
        mutation_false_accepts: list[dict[str, str]] = []
        for task_id, problem in self.problems.items():
            canonical = str(problem["prompt"]) + str(problem["canonical_solution"])
            try:
                canonical_result = self.evaluate(task_id, canonical)
            except Exception as exc:
                raise RuntimeError(
                    f"canonical evaluator sanity crashed for {task_id}: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            if not canonical_result["passed"]:
                canonical_failures.append(
                    {"task_id": task_id, "message": canonical_result["message"]}
                )

            known_bad = (
                str(problem["prompt"])
                + "\n    raise AssertionError('intentional evaluator sanity mutation')\n"
            )
            try:
                mutation_result = self.evaluate(task_id, known_bad)
            except Exception as exc:
                raise RuntimeError(
                    f"mutation evaluator sanity crashed for {task_id}: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            if mutation_result["passed"]:
                mutation_false_accepts.append(
                    {"task_id": task_id, "message": mutation_result["message"]}
                )

        if canonical_failures or mutation_false_accepts:
            raise RuntimeError(
                "evaluator sanity failed: "
                f"canonical_failures={canonical_failures[:5]}, "
                f"mutation_false_accepts={mutation_false_accepts[:5]}"
            )
        return {
            "status": "passed",
            "tasks_checked": len(self.problems),
            "canonical_solutions_passed": len(self.problems),
            "known_bad_mutations_rejected": len(self.problems),
            "platform": platform.platform(),
            "evalplus_max_memory_bytes": os.getenv("EVALPLUS_MAX_MEMORY_BYTES"),
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        }


class ResultStore:
    """Append-only JSONL store with crash-tolerant resume by primary key."""

    def __init__(self, path: Path):
        self.path = path
        self.records: dict[tuple[str, str, str, str, int], dict[str, Any]] = {}
        if path.exists():
            lines = path.read_text(encoding="utf-8").splitlines()
            for index, line in enumerate(lines, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    if index == len(lines):
                        break
                    raise ValueError(f"invalid JSONL at {path}:{index}")
                key = primary_key(record)
                if key in self.records:
                    raise ValueError(f"duplicate result primary key at {path}:{index}: {key}")
                self.records[key] = record

    def has(self, key: tuple[str, str, str, str, int]) -> bool:
        return key in self.records

    def append(self, record: dict[str, Any]) -> None:
        if record.get("status") not in RESULT_STATUSES:
            raise ValueError(f"invalid result status: {record.get('status')}")
        key = primary_key(record)
        if key in self.records:
            raise ValueError(f"result already exists: {key}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.records[key] = record


def write_json_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
    return records


def exact_mcnemar_p(fixes: int, regressions: int) -> float:
    discordant = fixes + regressions
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(fixes, regressions) + 1)) / (2**discordant)
    return min(1.0, 2 * tail)


def paired_bootstrap_delta(
    pairs: Iterable[tuple[bool, bool]], *, samples: int = 10_000, seed: int = 0
) -> tuple[float, float]:
    pairs = list(pairs)
    if not pairs:
        raise ValueError("no completed pairs")
    rng = random.Random(seed)
    deltas = []
    for _ in range(samples):
        draw = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        deltas.append(
            sum(int(treatment) - int(direct) for direct, treatment in draw) / len(draw)
        )
    low = percentile(deltas, 0.025)
    high = percentile(deltas, 0.975)
    assert low is not None and high is not None
    return low, high

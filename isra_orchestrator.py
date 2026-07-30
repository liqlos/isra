#!/usr/bin/env python3
"""
ISRA Orchestrator — Iterative Self-Refinement Architecture.

OpenAI-compatible HTTP server that runs a 4-phase pipeline
(Deep Thinking → Skeptical Review → Essence Extraction → Final Synthesis)
with iterative refinement, using a local MLX backend via the model router.

Architecture:
  Client (pi/goose/curl) → :8083/v1/chat/completions
    → Phase 1: Deep Thinking      (temp=0.7, max_tokens=2048)
    → Phase 2: Skeptical Review   (temp=0.2, max_tokens=600)
    → Phase 3: Essence Extraction (temp=0.1, max_tokens=500)
    → [Decision Gate: confidence >= 90 OR stagnation OR max_iters]
    → Phase 4: Final Synthesis     (temp=0.2, max_tokens=1000)
    → OpenAI-format response (streaming or non-streaming)

Each phase calls the model router at http://127.0.0.1:8080 with
model="qwen3-a3b" and per-request temperature/max_tokens (supported by
mlx_lm 0.31.3 — verified from source).

Endpoints:
  POST /v1/chat/completions  — OpenAI-compatible (stream + non-stream)
  GET  /v1/models            — lists isra-a3b
  GET  /health               — status + config
"""

import asyncio
import hashlib
import json
import logging
import os
import re
import ast
import sys
import time
import uuid
from difflib import SequenceMatcher
from typing import Any

from aiohttp import web, ClientSession, ClientTimeout
import aiohttp

# ─── Configuration ──────────────────────────────────────────────────────────

ROUTER_URL = os.environ.get("ISRA_ROUTER_URL", "http://127.0.0.1:8080")
BACKEND_MODEL = os.environ.get("ISRA_BACKEND_MODEL", "qwen3-a3b")
ISRA_MODEL_ID = "isra-a3b"
PORT = int(os.environ.get("ISRA_PORT", "8083"))
HOST = os.environ.get("ISRA_HOST", "0.0.0.0")

MAX_ITERATIONS = int(os.environ.get("ISRA_MAX_ITERS", "3"))
CONFIDENCE_THRESHOLD = int(os.environ.get("ISRA_CONFIDENCE_THRESHOLD", "75"))
STAGNATION_THRESHOLD = float(os.environ.get("ISRA_STAGNATION_THRESHOLD", "0.75"))

# Per-phase sampling parameters (from ISRA spec section 2)
# max_tokens accommodates qwen3 thinking mode (reasoning + content).
# Phase 2 disables thinking — with thinking ON, qwen3 returns everything
# in the reasoning field and content stays empty, breaking tag parsing.
# Phase 3 disables thinking (pure structured extraction, no reasoning needed).
# Phase 4 enables thinking for re-reasoning and constraint verification.
PHASE_PARAMS = {
    # B10: Phase 1 temp 0.6/0.95 per Qwen3 thinking mode spec (was 0.7/0.9)
    # max_tokens reduced from 3000→2000 to avoid KV cache memory pressure (jetsam)
    0: {"temperature": 0.0, "top_p": 0.95, "max_tokens": 50, "enable_thinking": False},  # Phase 0: quick prediction
    1: {"temperature": 0.6, "top_p": 0.95, "max_tokens": 2000, "enable_thinking": True},
    2: {"temperature": 0.2, "top_p": 0.95, "max_tokens": 1500, "enable_thinking": False},
    3: {"temperature": 0.1, "top_p": 0.95, "max_tokens": 1000, "enable_thinking": False},
    # Phase 4 max_tokens reduced from 2500→2000 for same reason
    4: {"temperature": 0.2, "top_p": 0.95, "max_tokens": 2000, "enable_thinking": True},
}

# Backend request timeout — pipeline phases can take a while (Phase 1 thinking
# can generate 10K+ chars, needs generous timeout)
BACKEND_TIMEOUT = ClientTimeout(total=420, sock_read=180)

# Session storage — accumulates confirmed insights across messages in a conversation
# Keyed by session_id (from X-Session-Id header or auto-generated)
# Sessions expire after SESSION_TTL seconds of inactivity
SESSION_TTL = int(os.environ.get("ISRA_SESSION_TTL", "3600"))  # 1 hour
_sessions: dict[str, dict] = {}
_sessions_lock = asyncio.Lock()


def _get_session(session_id: str) -> dict:
    """Get or create a session state."""
    now = time.time()
    sess = _sessions.get(session_id)
    if sess is None or (now - sess["last_active"]) > SESSION_TTL:
        sess = {
            "confirmed_insights": [],
            "disputed_claims": [],
            "constraints": [],
            "last_active": now,
        }
        _sessions[session_id] = sess
    sess["last_active"] = now
    return sess


def _cleanup_sessions() -> None:
    """Remove expired sessions."""
    now = time.time()
    expired = [sid for sid, s in _sessions.items() if (now - s["last_active"]) > SESSION_TTL]
    for sid in expired:
        del _sessions[sid]
    if expired:
        log.info(f"Cleaned up {len(expired)} expired sessions")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [ISRA] %(levelname)s %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("isra")

# ─── Task classification & helpers (A1, A8) ────────────────────────────────────


def classify_task(query: str) -> str:
    """Lightweight heuristic classifier: CODE, MATH, or GENERAL.
    Conservative — only classifies as CODE/MATH when markers are strong.
    Defaults to GENERAL to avoid false positives (e.g., 'capital of France' ≠ MATH).
    B1 fix: CODE check runs FIRST with expanded NL markers; code-context disambiguates math."""
    q = query.lower()
    # CODE: strong markers — function definitions, imports, fenced code blocks, NL code verbs
    # B1: Expanded with natural-language code request patterns
    code_markers = (
        "def ", "import ", "```python", "```python\n",
        # NL code verbs (B1 fix for misrouting "Write a function..." to MATH/GENERAL)
        "write a function", "write a python function", "implement a function",
        "write a function that", "write code", "complete the function",
        "fill in the", "function that takes", "function that returns",
        "write a program", "write a script", "implement the function",
        "implement a python", "write a method", "define a function",
        "complete the function", "complete the following", "complete the python",
    )
    # "class " is a code marker only when followed by a capital letter (Python class definition)
    # "dance class" or "class of students" should NOT trigger CODE
    if re.search(r"\bclass\s+[A-Z]", q):
        return "CODE"
    if any(m in q for m in code_markers):
        return "CODE"
    # MATH: strong markers — explicit math operations or word-problem patterns
    # Avoid false positives: "what is the capital" should NOT be MATH
    # B1: code-context disambiguation — if code markers present, already returned CODE above
    math_strong = ("calculate", "solve for", "how many", "sum of", "product of",
                   "divided by", "multiply", "subtract", "add ", "percent of",
                   "how much", "total cost", "how long", "how old", "how load",
                   "how fast", "how far", "how big", "how tall", "how deep",
                   "what is the total", "what is the sum", "what is the difference",
                   "how many more", "how many less", "how much does", "how much is",
                   "how much did", "how much will", "how much would", "how much can",
                   "how much should", "how much more", "how much less")
    if any(m in q for m in math_strong):
        return "MATH"
    # Check for arithmetic expressions (digits + operators) — but require actual math context
    # "2+2" or "5 * 3" → MATH; "Year 2024" → GENERAL
    if re.search(r"\d+\s*[+\-*/]\s*\d+", q):
        return "MATH"
    # GSM8K-style word problems: queries with multiple numbers and question words
    has_numbers = len(re.findall(r"\d", q)) >= 3
    has_question = any(w in q for w in ("how", "what", "find", "determine", "calculate"))
    if has_numbers and has_question:
        return "MATH"
    return "GENERAL"


def is_trivial_query(query: str) -> bool:
    """Conservative trivial-query filter for A8 direct bypass.
    Only bypasses for greetings, simple definitions, and chit-chat.
    NEVER bypasses code or math tasks (consults classify_task)."""
    if len(query) > 30:
        return False
    q_lower = query.lower().strip()
    # Never bypass if classify_task detects CODE or MATH
    task_type = classify_task(query)
    if task_type in ("CODE", "MATH"):
        return False
    # Never bypass questions (even short ones)
    reasoning_words = ("how", "why", "explain", "write", "create", "solve", "what", "when", "where", "who", "which")
    if any(w in q_lower for w in reasoning_words):
        return False
    # Only bypass greetings, thanks, and simple statements
    trivial_patterns = ("hello", "hi ", "hey", "thanks", "thank you", "ok", "yes", "no", "bye", "sure", "cool", "nice")
    if any(q_lower.startswith(p) or q_lower == p.strip() for p in trivial_patterns):
        return True
    return False


def execute_code_safely(code: str) -> tuple[bool, str]:
    """Check code for syntax errors via compile(). Returns (success, stderr).
    NOTE: Does NOT execute the code — exec() in-process can segfault/loop and kill ISRA.
    subprocess spawning causes jetsam. compile() is zero-cost and catches the most
    common error (SyntaxError) that indicates broken code generation."""
    try:
        compile(code, "<string>", "exec")
        return True, ""
    except SyntaxError as e:
        return False, f"SyntaxError: {e.msg} at line {e.lineno}"


async def generate_and_run_test_cases(session, user_query: str, code: str) -> str:
    """Self-generated test cases: ask model to generate tests from docstring,
    then execute them against the generated code. Returns error message or empty string.
    This helps for CODE tasks that have no >>> doctest examples in the query."""
    # Ask model to generate test cases
    gen_prompt = (
        f"Based on this function signature and docstring, write 3-5 test cases as Python assert statements.\n"
        f"Only output the assert statements, nothing else.\n"
        f"Include edge cases (empty input, single element, etc.).\n\n"
        f"{user_query}"
    )
    gen_output = await call_backend(
        session, "You are a test case generator. Output only assert statements.",
        gen_prompt, phase=1, enable_thinking_override=False,
    )
    # Extract assert statements from output
    assert_lines = []
    for line in gen_output.split("\n"):
        line = line.strip()
        if line.startswith("assert ") and not any(d in line for d in ("open(", "os.", "subprocess", "__import__", "eval(", "exec(")):
            assert_lines.append(line)
    if not assert_lines:
        return ""  # No tests generated, skip
    # Run tests in subprocess
    import subprocess
    test_script = code + "\n\n" + "\n".join(assert_lines)
    try:
        result = subprocess.run(
            ["python3", "-c", test_script],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            err = result.stderr.strip().split("\n")[-1] if result.stderr else "unknown error"
            return f"[SELF_TEST_FAILED] {err[:120]}"
    except subprocess.TimeoutExpired:
        return "[SELF_TEST_FAILED] timeout"
    except Exception as e:
        return f"[SELF_TEST_FAILED] {e}"
    return ""


def extract_doctest_examples(query: str) -> list:
    """B2: Extract >>> examples from docstring in user query.
    Returns list of (expression, expected_output) tuples.
    Skips destructive examples (open/os/subprocess)."""
    examples = []
    lines = query.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith(">>> "):
            expr = line[4:]
            expected_lines = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith(">>>") and lines[i].strip() and '"""' not in lines[i]:
                expected_lines.append(lines[i].strip())
                i += 1
            expected = "\n".join(expected_lines)
            # Skip destructive examples
            if not any(d in expr for d in ("open(", "os.", "subprocess", "__import__", "eval(", "exec(")):
                examples.append((expr, expected))
        else:
            i += 1
    return examples


def run_doctest_examples(code: str, examples: list) -> tuple[bool, str]:
    """B2: Check doctest examples by running them in a subprocess.
    Safe: isolated process, can't crash ISRA. Timeout: 5s per test.
    Returns (all_passed, failure_msg)."""
    if not examples or not code:
        return True, ""
    # First: compile check (fast, catches syntax errors without subprocess)
    try:
        compile(code, "<string>", "exec")
    except SyntaxError as e:
        return False, f"[DOCSTRING_TEST_FAILED] SyntaxError: {e.msg} at line {e.lineno}"

    # Run each doctest example in a subprocess for real execution feedback
    import subprocess
    failures = []
    for expr, expected in examples:
        # Build test script: code + expression + comparison
        # Use repr() for comparison — doctest output shows repr of values
        # (e.g. strings show with quotes: 'hello' not hello)
        test_script = code + f"\n\n_result = repr({expr})\n_expected = {expected!r}\nassert _result == _expected, f'Got {{_result}}, expected {{_expected}}'\n"
        try:
            result = subprocess.run(
                ["python3", "-c", test_script],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode != 0:
                err = result.stderr.strip().split("\n")[-1] if result.stderr else "unknown error"
                failures.append(f"{expr[:50]} → {err[:80]}")
        except subprocess.TimeoutExpired:
            failures.append(f"{expr[:50]} → timeout")
        except Exception as e:
            failures.append(f"{expr[:50]} → {e}")

    if failures:
        return False, f"[DOCSTRING_TEST_FAILED] {len(failures)} of {len(examples)} tests failed: {'; '.join(failures[:3])}"
    return True, ""


def _parse_bullets(text: str) -> list[str]:
    """Parse a bullet list into a list of strings."""
    bullets = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^[-*]\s+", "", line)
        if line:
            bullets.append(line)
    return bullets


def _extract_constraints_from_phase1(phase1_output: str) -> list[str]:
    raw = extract_tag(phase1_output, "CONSTRAINTS") or ""
    return _parse_bullets(raw)


# ─── Prompt Templates (ISRA spec section 4) ─────────────────────────────────

PHASE1_SYSTEM_GENERAL = """You are an advanced, analytical reasoning engine. Your goal is to solve the query by step-by-step logical exploration.

INSTRUCTIONS:
1. First, think step-by-step inside the <think>...</think> block. Address any previous critique points directly.
2. List ALL constraints, rules, and conditions from the user query verbatim under [CONSTRAINTS]. Constraints are the stated rules the answer must satisfy (e.g. "Alice doesn't like red", "output must be 5 steps"). Copy them word-for-word — do NOT paraphrase or omit.
3. After thinking, output an explicit list of distinct factual findings under the [CONCLUSIONS] header. Each conclusion MUST be consistent with ALL constraints.
4. Keep conclusions clear, objective, and atomic. Mark uncertain points with [UNVERIFIED].
5. Before finalizing, verify each conclusion against the listed constraints. For code, mentally trace execution with at least one sample input. For math, re-derive the final number via a second method or re-check arithmetic.
6. If the query asks for a numeric/math answer, put the final number in [ANSWER] as just the number (or "#### number" for GSM8K-style).

REQUIRED OUTPUT FORMAT:
<think>
[Detailed step-by-step exploration, math verification, edge case testing]
</think>

[CONSTRAINTS]
- Constraint 1 (verbatim from query)
- Constraint 2 (verbatim from query)

[CONCLUSIONS]
- Conclusion item 1
- Conclusion item 2
- Conclusion item 3

[ANSWER]
[Final answer: one-sentence summary, or the number, or "Refer to conclusions/code"]

EXAMPLE:
USER QUERY: What are the benefits of exercise?
<think>
Exercise improves cardiovascular health, mental health, and longevity. Verify against constraints: all are health benefits.
</think>
[CONSTRAINTS]
- Answer must cover health benefits
[CONCLUSIONS]
- Improves cardiovascular health
- Reduces stress and improves mental health
- Increases lifespan
[ANSWER]
Exercise improves heart health, reduces stress, and increases longevity."""

PHASE1_SYSTEM_CODE = """You are an advanced, analytical reasoning engine. Your goal is to solve the query by step-by-step logical exploration.

INSTRUCTIONS:
1. First, think step-by-step inside the <think>...</think> block. Address any previous critique points directly.
2. List ALL constraints, rules, and conditions from the user query verbatim under [CONSTRAINTS]. Constraints are the stated rules the answer must satisfy (e.g. function signature, input types). Copy them word-for-word — do NOT paraphrase or omit.
3. After thinking, output an explicit list of distinct factual findings under the [CONCLUSIONS] header. Each conclusion MUST be consistent with ALL constraints.
4. Keep conclusions clear, objective, and atomic. Mark uncertain points with [UNVERIFIED].
5. Before finalizing, verify each conclusion against the listed constraints. Mentally trace execution with at least one sample input. Extract the function signature and any `>>>` examples from the docstring. Trace your code with the docstring's example inputs before finalizing. Consider edge cases: empty input, single-element input, boundary values (0, negative, max int). Handle them explicitly.
6. Output the COMPLETE code solution verbatim in a [CODE] section. The code MUST be in a single fenced ```python block, with correct indentation, complete function signatures, and ready to execute. Do NOT describe the code in prose — output the actual code.

REQUIRED OUTPUT FORMAT:
<think>
[Detailed step-by-step exploration, edge case testing, trace with sample input]
</think>

[CONSTRAINTS]
- Constraint 1 (verbatim from query)
- Constraint 2 (verbatim from query)

[CONCLUSIONS]
- Conclusion item 1
- Conclusion item 2
- Conclusion item 3

[CODE]
```python
[Complete code solution — ONLY if the query asks for code]
```

EXAMPLE:
USER QUERY: Write a function that returns the sum of two numbers.
<think>
Function needs two parameters and returns their sum. Trace: add(2,3) → 5. Correct.
</think>
[CONSTRAINTS]
- Function takes two numbers
- Returns their sum
[CONCLUSIONS]
- Define function add(a, b) that returns a + b
[CODE]
```python
def add(a, b):
    return a + b
```"""

PHASE1_SYSTEM_MATH = """You are an advanced, analytical reasoning engine. Your goal is to solve the query by step-by-step logical exploration.

INSTRUCTIONS:
1. First, think step-by-step inside the <think>...</think> block. Address any previous critique points directly.
2. List ALL constraints, rules, and conditions from the user query verbatim under [CONSTRAINTS]. Constraints are the stated rules the answer must satisfy. Copy them word-for-word — do NOT paraphrase or omit.
3. After thinking, output an explicit list of distinct factual findings under the [CONCLUSIONS] header. Each conclusion MUST be consistent with ALL constraints.
4. Keep conclusions clear, objective, and atomic. Mark uncertain points with [UNVERIFIED].
5. Before finalizing, verify each conclusion against the listed constraints. Re-derive the final number via a second method or re-check arithmetic.
6. Put the final numeric answer in [ANSWER] using the GSM8K format: #### <number>.

REQUIRED OUTPUT FORMAT:
<think>
[Detailed step-by-step exploration, math verification, re-derive via second method]
</think>

[CONSTRAINTS]
- Constraint 1 (verbatim from query)
- Constraint 2 (verbatim from query)

[CONCLUSIONS]
- Conclusion item 1
- Conclusion item 2
- Conclusion item 3

[ANSWER]
#### [number]

EXAMPLE:
USER QUERY: James has 3 apples. He buys 5 more. How many does he have?
<think>
James starts with 3 apples. He buys 5 more. 3 + 5 = 8. Verify: 8 - 5 = 3, correct.
</think>
[CONSTRAINTS]
- James starts with 3 apples
- He buys 5 more apples
[CONCLUSIONS]
- James has 8 apples total
[ANSWER]
#### 8

MULTI-STEP EXAMPLE:
USER QUERY: A bakery sells 12 loaves of bread at $3 each and 8 pastries at $2 each. How much money did they make?
<think>
Bread revenue: 12 x $3 = $36. Pastry revenue: 8 x $2 = $16. Total = $36 + $16 = $52.
Verify: $52 - $36 = $16 = 8 x $2. Correct. $52 - $16 = $36 = 12 x $3. Correct.
</think>
[CONSTRAINTS]
- 12 loaves at $3 each
- 8 pastries at $2 each
- Find total money made
[CONCLUSIONS]
- Bread revenue: $36
- Pastry revenue: $16
- Total revenue: $52
[ANSWER]
#### 52"""

PHASE1_SYSTEM = PHASE1_SYSTEM_GENERAL

PHASE2_SYSTEM = """You are a careful, objective quality-assurance reviewer. Your task is to assess whether the conclusions are correct, complete, and consistent with the original query's constraints.

INSTRUCTIONS:
Review the conclusions against the original query. Report genuine issues only — do not invent problems. If the conclusions are correct and satisfy all constraints, say so explicitly and assign high confidence.

For CODE tasks: check syntax, logic, edge cases, and whether the code satisfies the function signature and docstring. If the code looks correct, assign high confidence. Do NOT nitpick style — focus on correctness. Trace execution of the code with the docstring's example input (or one simple input) step-by-step. State the expected output and the output your trace produces. If they differ, flag as [CODE_QUALITY] HIGH.

Default to skepticism. A plausible-sounding answer is NOT correct. Before assigning confidence >= 85%, you MUST have explicitly verified: (a) for MATH — re-derived the arithmetic and substituted back; (b) for CODE — traced execution with a concrete input and confirmed the output. If you have not done these verifications, cap confidence at 75%.

CONFIDENCE CALIBRATION GUIDE:
- 90-100%: No significant issues. Conclusions are correct, complete, and satisfy all constraints.
- 75-89%: Minor issues found (LOW severity). Conclusions mostly correct.
- 60-74%: Moderate issues found (MEDIUM severity). Some corrections needed.
- Below 60%: Major errors found (HIGH severity). Significant rework needed.

REQUIRED OUTPUT FORMAT:

[ISSUES]
(Only include categories where you found real issues. Omit empty categories.)
- [CONSTRAINT_VIOLATION] | {HIGH|MEDIUM|LOW} | {REMOVE|VERIFY|INVESTIGATE} | Which conclusion violates which constraint from the query
- [LOGIC] | {HIGH|MEDIUM|LOW} | {REMOVE|VERIFY|INVESTIGATE} | Description of logical issue
- [FACT] | {HIGH|MEDIUM|LOW} | {REMOVE|VERIFY|INVESTIGATE} | Description of factual error or hallucination
- [SCOPE] | {HIGH|MEDIUM|LOW} | {REMOVE|VERIFY|INVESTIGATE} | Description of missing edge case or scope error
- [CODE_QUALITY] | {HIGH|MEDIUM|LOW} | {REMOVE|VERIFY|INVESTIGATE} | Syntax error, missing return, wrong signature, or runtime error in code

[STRENGTHS]
- Bullet point of verified solid reasoning
- Bullet point of accurate conclusion

[CONFIDENCE]
{0-100}%

[RECOMMENDATION]
{STOP|REFINE_SPECIFIC|RETHINK}"""

# B3+B4: Math-specific critic with independent re-derivation and substitute verification
PHASE2_SYSTEM_MATH = """You are a careful, objective quality-assurance reviewer for MATH tasks. Your task is to verify the conclusions are arithmetically correct.

INSTRUCTIONS:
For MATH tasks: BEFORE reviewing the conclusions, independently solve the problem step-by-step from scratch. Then compare your independent solution to the conclusions. Flag any discrepancy as a [LOGIC] HIGH issue. Your independent solution is the ground truth for verification.

After re-deriving, substitute the proposed answer back into the problem's original quantities and verify consistency (e.g., if the answer is 8 and the problem says 'James had 3, bought 5', check 8 - 5 = 3 matches). If substitution fails, flag as [LOGIC] HIGH.

Default to skepticism. A plausible-sounding answer is NOT correct. Before assigning confidence >= 85%, you MUST have explicitly re-derived the arithmetic AND substituted back to verify. If you have not done these verifications, cap confidence at 75%.

CONFIDENCE CALIBRATION GUIDE:
- 90-100%: Independently re-derived and substituted back — both match the conclusions.
- 75-89%: Minor issues found (LOW severity). Conclusions mostly correct.
- 60-74%: Moderate issues found (MEDIUM severity). Some corrections needed.
- Below 60%: Major errors found (HIGH severity). Arithmetic does not match independent re-derivation.

REQUIRED OUTPUT FORMAT:

[ISSUES]
(Only include categories where you found real issues. Omit empty categories.)
- [CONSTRAINT_VIOLATION] | {HIGH|MEDIUM|LOW} | {REMOVE|VERIFY|INVESTIGATE} | Which conclusion violates which constraint from the query
- [LOGIC] | {HIGH|MEDIUM|LOW} | {REMOVE|VERIFY|INVESTIGATE} | Description of arithmetic or logical issue
- [FACT] | {HIGH|MEDIUM|LOW} | {REMOVE|VERIFY|INVESTIGATE} | Description of factual error

[STRENGTHS]
- Bullet point of verified solid reasoning
- Bullet point of accurate conclusion

[CONFIDENCE]
{0-100}%

[RECOMMENDATION]
{STOP|REFINE_SPECIFIC|RETHINK}"""

PHASE3_SYSTEM = """You are a context compression engine. Your task is to merge original conclusions with a critic's audit report to produce a sanitized, factual state for the next phase.

RULES:
1. REMOVE any claim associated with a HIGH severity issue marked REMOVE.
2. Prefix any claim associated with HIGH/MEDIUM issues marked VERIFY with [DISPUTED].
3. Prefix items flagged as INVESTIGATE with [NEEDS_REVIEW].
4. Preserve clean, verified statements under [CONFIRMED_INSIGHTS]. Write ACTUAL content from the conclusions — do NOT write placeholder text like "Validated bullet point 1".
5. Copy ALL constraints from the [CONSTRAINTS] section verbatim into [CONSTRAINTS]. Do NOT compress, paraphrase, or omit any constraint. Constraints are NEVER removed — they are rules the answer must satisfy.
6. If the conclusions contain a [CODE] section or code blocks, copy the code VERBATIM into [CODE]. Do NOT paraphrase, summarize, or truncate code. Preserve indentation and newlines exactly.
7. Do NOT add new information not present in the input context.

REQUIRED OUTPUT FORMAT:

[CONSTRAINTS]
- (copy each constraint verbatim from the input)

[CONFIRMED_INSIGHTS]
- (actual validated findings from the conclusions — NOT placeholders)

[DISPUTED_CLAIMS]
- [DISPUTED] (actual disputed claims — omit section if none)

[OPEN_QUESTIONS]
- [NEEDS_REVIEW] (actual items needing exploration — omit section if none)

[CODE]
```python
(actual code from conclusions — omit section if no code)
```"""

PHASE4_SYSTEM = """You are a clear, authoritative expert communicator. Synthesize the final user answer from the accumulated verified insights and the original query.

RULES:
1. Before writing the answer, verify each confirmed insight against the user query's constraints. If any insight contradicts a stated constraint in the query, DISCARD that insight and re-derive the answer from the query's constraints directly.
2. Build a direct, comprehensive answer to the user query. Use the confirmed insights as primary material, but the original query's constraints ALWAYS take precedence over accumulated insights.
3. If forced to reference a disputed claim, explicitly state "[Requires further verification]".
4. Do NOT include meta-commentary about iterations, critics, or the execution system.
5. Output cleanly formatted Markdown.
6. If the query asks for CODE and a [CODE] section is provided, output the code VERBATIM in a fenced ```python block. Do NOT rewrite, paraphrase, or "improve" the code unless it has a clear syntax error. The code from [CODE] is the answer — output it directly.
7. If the query asks for a numeric/math answer, output the final number clearly (e.g., "#### 42" for GSM8K-style, or just the number).
8. Do NOT include commentary about the code, the reasoning process, or "here is the solution". Output ONLY what the user asked for."""

# ─── Parsing helpers (regex with fallback defaults) ─────────────────────────

def extract_tag(text: str, tag: str) -> str | None:
    """Extract content between [TAG]...[/TAG] or [TAG]...next-tag/EOF."""
    # Try [TAG]...[/TAG] first
    m = re.search(rf"\[{tag}\](.*?)\[/{tag}\]", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # Try [TAG]... until next [TAG2] or EOF
    m = re.search(rf"\[{tag}\]\s*(.*?)(?=\n\[[A-Z_]+\]|\Z)", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return None


def extract_think_block(text: str) -> str:
    """Extract <think>...</think> content (discarded after Phase 1)."""
    m = re.search(r"<think>(.*?)</think>", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # Unclosed <think> — take to end (truncation case)
    m = re.search(r"<think>(.*)", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return ""


def parse_conclusions(text: str) -> str:
    """Extract [CONCLUSIONS] section from Phase 1 output."""
    conclusions = extract_tag(text, "CONCLUSIONS")
    if conclusions:
        return conclusions
    # Fallback: if no [CONCLUSIONS] tag, strip <think> block and return rest
    stripped = re.sub(r" <think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    # If there's an [ANSWER] tag, exclude it
    answer_match = re.search(r"\[ANSWER\]", stripped, re.IGNORECASE)
    if answer_match:
        stripped = stripped[:answer_match.start()].strip()
    return stripped if stripped else text.strip()


def parse_code(text: str) -> str:
    """Extract [CODE] section from Phase 1 output. Returns empty string if no code.
    Preserves the code block verbatim (including indentation and newlines).
    Fallback: if no [CODE] tag, search for fenced ```python blocks in the text."""
    code_section = extract_tag(text, "CODE")
    if code_section:
        # Strip the fenced code block markers but keep the code inside
        m = re.search(r"```(?:python)?\s*\n(.*?)```", code_section, re.DOTALL)
        if m:
            return m.group(1).strip()
        return code_section.strip()
    # Fallback: search for ```python blocks anywhere in the text
    # (qwen3 with thinking ON may put code in reasoning without [CODE] tag)
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return ""


def parse_answer(text: str) -> str:
    """Extract [ANSWER] section from Phase 1 output."""
    return (extract_tag(text, "ANSWER") or "").strip()


def _sanitize_math_answer(answer: str, fallback: str = "") -> str:
    """C2: Ensure MATH final answer is in #### N format.
    Searches for #### N pattern first, then falls back to extracting the last number.
    If fallback is provided (from Phase 1 [ANSWER]), use it when no number is found."""
    # Already has #### N format
    m = re.search(r"####\s*([\d,]+(?:\.\d+)?)", answer)
    if m:
        return f"#### {m.group(1).replace(',', '')}"
    # Try \\boxed{N}
    m = re.search(r"\\boxed\{([\d,]+(?:\.\d+)?)\}", answer)
    if m:
        return f"#### {m.group(1).replace(',', '')}"
    # If we have a fallback from Phase 1, prefer it over guessing from prose.
    # Phase 4 often generates explanatory text with numbers that aren't the answer
    # (e.g., "The answer is 6 km/h" when Phase 1 said 12). Using fallback is safer.
    if fallback:
        return f"#### {fallback}"
    # Last resort: try to find a number in the last few lines
    lines = [l.strip() for l in answer.strip().split("\n") if l.strip()]
    for line in reversed(lines[-5:]):
        m = re.search(r"([\d,]+(?:\.\d+)?)", line)
        if m:
            return f"#### {m.group(1).replace(',', '')}"
    return answer


def parse_critique(text: str) -> dict[str, Any]:
    """Parse Phase 2 critique output. Returns dict with issues, confidence, recommendation."""
    issues_raw = extract_tag(text, "ISSUES") or ""
    strengths_raw = extract_tag(text, "STRENGTHS") or ""
    confidence_raw = extract_tag(text, "CONFIDENCE") or ""

    # Parse confidence — prefer number followed by %, fall back to any number.
    # Two-step search avoids picking "2" from "2 issues found. Confidence: 85%".
    conf_match = re.search(r"(\d+(?:\.\d+)?)\s*%", confidence_raw)
    if not conf_match:
        conf_match = re.search(r"(\d+(?:\.\d+)?)", confidence_raw)
    if conf_match:
        confidence = min(100, max(0, int(round(float(conf_match.group(1))))))
    else:
        confidence = 50  # default 50% forces re-pass

    # Parse recommendation
    rec_raw = extract_tag(text, "RECOMMENDATION") or ""
    rec_upper = rec_raw.upper().strip()
    if "RETHINK" in rec_upper:
        recommendation = "RETHINK"
    elif "REFINE" in rec_upper:
        recommendation = "REFINE_SPECIFIC"
    elif "STOP" in rec_upper:
        recommendation = "STOP"
    else:
        recommendation = "REFINE_SPECIFIC"  # default to refine

    # Parse issues into structured list
    issues = []
    for line in issues_raw.split("\n"):
        line = line.strip().lstrip("-").strip()
        if not line:
            continue
        # Format: [CATEGORY] | SEVERITY | ACTION | Description
        m = re.match(
            r"\[?(CONSTRAINT_VIOLATION|LOGIC|FACT|SCOPE|CODE_QUALITY)\]?\s*\|\s*(HIGH|MEDIUM|LOW)\s*\|\s*(REMOVE|VERIFY|INVESTIGATE)\s*\|\s*(.+)",
            line, re.IGNORECASE,
        )
        if m:
            issues.append({
                "category": m.group(1).upper(),
                "severity": m.group(2).upper(),
                "action": m.group(3).upper(),
                "description": m.group(4).strip(),
            })
        else:
            # Unparseable issue line — keep as low-severity
            issues.append({
                "category": "UNKNOWN",
                "severity": "LOW",
                "action": "INVESTIGATE",
                "description": line,
            })

    return {
        "issues": issues,
        "strengths": [s.strip().lstrip("-").strip() for s in strengths_raw.split("\n") if s.strip()],
        "confidence": confidence,
        "recommendation": recommendation,
        "raw": text,
    }


def parse_essence(text: str) -> dict[str, Any]:
    """Parse Phase 3 essence extraction output."""
    constraints_raw = extract_tag(text, "CONSTRAINTS") or ""
    confirmed_raw = extract_tag(text, "CONFIRMED_INSIGHTS") or ""
    disputed_raw = extract_tag(text, "DISPUTED_CLAIMS") or ""
    questions_raw = extract_tag(text, "OPEN_QUESTIONS") or ""
    code_raw = extract_tag(text, "CODE") or ""

    def parse_bullets(raw: str) -> list[str]:
        """Parse bullet list. Preserves fenced code blocks as single bullets."""
        bullets = []
        # Split into lines but keep fenced code blocks together
        lines = raw.split("\n")
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            if not stripped:
                i += 1
                continue
            # Check if this line starts a fenced code block
            if stripped.startswith("```"):
                # Collect all lines until closing ```
                code_lines = [line]
                i += 1
                while i < len(lines):
                    code_lines.append(lines[i])
                    if lines[i].strip().startswith("```"):
                        i += 1
                        break
                    i += 1
                code_block = "\n".join(code_lines)
                # Strip leading "- " if the fence was a bullet
                code_block = re.sub(r"^[-*]\s+", "", code_block)
                bullets.append(code_block)
                continue
            # Regular bullet
            line = re.sub(r"^[-*]\s+", "", stripped)
            if line:
                bullets.append(line)
            i += 1
        return bullets

    # Extract code from [CODE] section (preserve fenced block content)
    code = ""
    if code_raw:
        m = re.search(r"```(?:python)?\s*\n(.*?)```", code_raw, re.DOTALL)
        if m:
            code = m.group(1).strip()
        else:
            code = code_raw.strip()

    return {
        "constraints": parse_bullets(constraints_raw),
        "confirmed_insights": parse_bullets(confirmed_raw),
        "disputed_claims": parse_bullets(disputed_raw),
        "open_questions": parse_bullets(questions_raw),
        "code": code,
        "raw": text,
    }


# ─── Stagnation detection ───────────────────────────────────────────────────

def state_similarity(a: str, b: str) -> float:
    """Compute text similarity between two compressed states (0.0–1.0)."""
    norm_a = " ".join(a.split())
    norm_b = " ".join(b.split())
    if not norm_a or not norm_b:
        return 0.0
    return SequenceMatcher(None, norm_a, norm_b).ratio()


# ─── Iteration control (ISRA spec section 3.1) ──────────────────────────────

def evaluate_iteration_control(
    confidence: int,
    issues: list[dict],
    iteration: int,
    max_iterations: int,
    similarity: float,
    recommendation: str = "",
) -> str:
    has_high_severity = any(i["severity"] == "HIGH" for i in issues)
    has_high_remove = any(i["severity"] == "HIGH" and i["action"] == "REMOVE" for i in issues)

    # Trust critic's STOP recommendation (model is better at binary STOP/REFINE than calibrated numbers)
    if recommendation == "STOP" and not has_high_remove:
        return "TERMINATE_SUCCESS"
    # Confidence gate (lowered to 75, only block on HIGH+REMOVE not HIGH+VERIFY)
    if confidence >= CONFIDENCE_THRESHOLD and not has_high_remove:
        return "TERMINATE_SUCCESS"
    if iteration >= max_iterations:
        return "TERMINATE_MAX_ITER"
    if similarity >= STAGNATION_THRESHOLD:
        return "TERMINATE_STAGNATION"
    if confidence < 60 or has_high_severity:
        return "LOOP_RETHINK"
    return "LOOP_REFINE"


# ─── Backend LLM caller ─────────────────────────────────────────────────────

async def call_backend(
    session: ClientSession,
    system_prompt: str,
    user_prompt: str,
    phase: int,
    stream: bool = False,
    enable_thinking_override: bool | None = None,
    temp_override: float | None = None,
) -> str:
    """Call the model router for a single phase. Returns full text output.
    enable_thinking_override: if set, overrides PHASE_PARAMS[phase]["enable_thinking"].
        Used for A7 (MATH tasks enable thinking for Phase 2) without mutating shared state.
    temp_override: if set, overrides PHASE_PARAMS[phase]["temperature"].
        Used for Phase 0 second call (diverse sampling).
    Uses shared session with force_close connector to prevent memory accumulation."""
    params = PHASE_PARAMS[phase]
    thinking_enabled = enable_thinking_override if enable_thinking_override is not None else params.get("enable_thinking", True)
    temp = temp_override if temp_override is not None else params["temperature"]
    body = {
        "model": BACKEND_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temp,
        "top_p": params["top_p"],
        "max_tokens": params["max_tokens"],
        "stream": False,  # always non-stream for phases (need full output to parse)
    }
    # Disable thinking mode for phases 2-4 (saves tokens, structured output only)
    if not thinking_enabled:
        body["chat_template_kwargs"] = {"enable_thinking": False}

    url = f"{ROUTER_URL}/v1/chat/completions"
    try:
        # Use urllib instead of aiohttp to avoid memory accumulation in aiohttp connection pool
        import urllib.request
        import json as _json
        req_body = _json.dumps(body).encode("utf-8")
        req = urllib.request.Request(url, data=req_body, headers={"Content-Type": "application/json"})
        loop = asyncio.get_event_loop()
        def _do_request():
            with urllib.request.urlopen(req, timeout=420) as resp2:
                return _json.loads(resp2.read())
        data = await loop.run_in_executor(None, _do_request)
        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError("Backend returned no choices")
        msg = choices[0].get("message", {})
        content = msg.get("content", "")
        reasoning = msg.get("reasoning", "") or msg.get("reasoning_content", "")
        # Combine: reasoning (thinking) + content (answer)
        # Parsers will extract tags from the combined text
        if reasoning and content:
            result = f"<think>{reasoning}</think>\n{content}"
        elif reasoning:
            result = re.sub(r"</?think>", "", reasoning, flags=re.IGNORECASE).strip()
        else:
            result = content
        del data, choices, msg
        return result
    except asyncio.TimeoutError:
        log.error(f"Phase {phase} backend timeout")
        raise RuntimeError(f"Phase {phase} timed out")
    except Exception as e:
        log.error(f"Phase {phase} backend exception: {e}")
        raise


async def stream_backend(
    session: ClientSession,
    system_prompt: str,
    user_prompt: str,
    phase: int,
    write_chunk,
) -> str:
    """Call backend with streaming and forward chunks to write_chunk callback.
    Returns the full accumulated text."""
    params = PHASE_PARAMS[phase]
    body = {
        "model": BACKEND_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": params["temperature"],
        "top_p": params["top_p"],
        "max_tokens": params["max_tokens"],
        "stream": True,
    }
    # Disable thinking for Phase 4 (streaming final answer, no thinking needed)
    if not params.get("enable_thinking", True):
        body["chat_template_kwargs"] = {"enable_thinking": False}

    url = f"{ROUTER_URL}/v1/chat/completions"
    full_text = []
    try:
        async with session.post(url, json=body, timeout=BACKEND_TIMEOUT) as resp:
            if resp.status != 200:
                err_text = await resp.text()
                raise RuntimeError(f"Backend returned {resp.status}: {err_text[:200]}")
            async for line in resp.content:
                line = line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        # Stream only content (not reasoning) to client
                        content = delta.get("content", "")
                        if content:
                            full_text.append(content)
                            await write_chunk(content)
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        log.error(f"Phase {phase} stream exception: {e}")
        raise
    return "".join(full_text)


# ─── ISRA Pipeline ──────────────────────────────────────────────────────────

async def run_isra_pipeline(
    session: ClientSession,
    user_query: str,
    conversation_context: str = "",
    session_state: dict | None = None,
    stream_final: bool = False,
    write_chunk=None,
) -> dict[str, Any]:
    """Run the full ISRA pipeline. Returns dict with final_answer and metadata.

    Args:
        user_query: The current user message (latest in conversation).
        conversation_context: Previous dialogue history (formatted string)
            to include in Phase 1 context so ISRA is aware of prior turns.
        session_state: Persistent state dict with 'confirmed_insights' and
            'disputed_claims' accumulated across messages in the session.
    """

    start_time = time.time()
    task_type = classify_task(user_query)
    log.info(f"Task classified as {task_type}")

    # ── Phase 0: Quick Prediction Check (P3 — predictive coding) ──────────
    # Two fast calls (thinking OFF, 20 tokens each). If answers match →
    # high confidence → return immediately, bypassing entire ISRA pipeline.
    # Saves ~47s for easy queries. Only for GENERAL (factual questions).
    # MATH is excluded — without reasoning, model guesses (sometimes consistently
    # wrong). DEER handles MATH early exit after Phase 1 with reasoning.
    # CODE is always routed to full pipeline (can't generate code in 20 tokens).
    if task_type == "GENERAL" and not conversation_context:
        phase0_prompt_map = {
            "MATH": f"{user_query}\n\nSolve and answer with just the number.",
            "GENERAL": f"{user_query}\n\nAnswer briefly in one sentence.",
        }
        phase0_prompt = phase0_prompt_map[task_type]

        # Call 1+2 in parallel (MLX server supports continuous batching)
        p0_a, p0_b = await asyncio.gather(
            call_backend(
                session, "You are a helpful assistant. Answer directly.", phase0_prompt,
                phase=0, enable_thinking_override=False,
            ),
            call_backend(
                session, "You are a helpful assistant. Answer directly.", phase0_prompt,
                phase=0, enable_thinking_override=False, temp_override=0.5,
            ),
        )

        # Normalize and compare
        def _norm(s):
            return s.lower().strip(".,!?;:\"'()[]{}").strip()

        a1_norm = _norm(p0_a.strip())
        a2_norm = _norm(p0_b.strip())

        if a1_norm and a1_norm == a2_norm:
            elapsed_p0 = time.time() - start_time
            log.info(f"Phase 0 HIT: answers match ({a1_norm[:50]!r}) — returning in {elapsed_p0:.1f}s")
            return {
                "final_answer": p0_a.strip(),
                "iterations": 0,
                "phases_run": ["phase0"],
                "phase0_hit": True,
                "elapsed_seconds": elapsed_p0,
                "termination_reason": "PHASE0_CONFIDENT",
                "task_type": task_type,
            }
        else:
            log.info(f"Phase 0 MISS: a1={a1_norm[:30]!r} vs a2={a2_norm[:30]!r} — escalating to full pipeline")

    # ── End Phase 0 ────────────────────────────────────────────────────────

    accumulated_confirmed: list[str] = []
    accumulated_disputed: list[str] = []
    accumulated_constraints: list[str] = []
    iterations_log: list[dict] = []

    # Seed accumulated state from session (persisted across messages in conversation)
    if session_state:
        accumulated_confirmed = list(session_state.get("confirmed_insights", []))
        accumulated_disputed = list(session_state.get("disputed_claims", []))
        accumulated_constraints = list(session_state.get("constraints", []))
        log.info(
            f"Session state loaded: {len(accumulated_confirmed)} confirmed, "
            f"{len(accumulated_disputed)} disputed, "
            f"{len(accumulated_constraints)} constraints from prior turns"
        )

    # Track the best code solution across iterations (for code tasks)
    best_code = ""
    # C1: Track the best math answer across iterations (for MATH short-circuit)
    best_math_answer = ""
    # C1: Track whether the last Phase 2 verified the answer (STOP + high confidence)
    math_answer_verified = False
    execution_attempted = False
    doctest_attempted = False  # B2: doctest execution tracking
    prev_error_sig = None  # B7: stuck-loop detection
    # B8: Best-guess fallback tracking
    best_iteration_confidence = -1
    best_iteration_index = 0
    best_iteration_state = None

    previous_state_text = ""
    decision = "TERMINATE_MAX_ITER"  # default if loop doesn't execute (MAX_ITERATIONS=0)

    for iteration in range(1, MAX_ITERATIONS + 1):
        log.info(f"--- Iteration {iteration}/{MAX_ITERATIONS} ---")

        # Reset doctest flag each iteration — code changes between iterations,
        # so we need to re-run doctests against the new code.
        if iteration > 1:
            doctest_attempted = False

        # Build Phase 1 input
        if iteration == 1 and not accumulated_confirmed and not accumulated_disputed:
            prev_insights = "(none — first iteration)"
            prev_critique = "(none — first iteration)"
            prev_open_questions = ""
            rethink_mode = False
        else:
            prev_insights = "\n".join(f"- {i}" for i in accumulated_confirmed) or "(none)"
            prev_disputed = "\n".join(f"- {d}" for d in accumulated_disputed) or "(none)"
            prev_insights = f"{prev_insights}\n\nDisputed claims from previous iteration:\n{prev_disputed}"
            prev_critique = iterations_log[-1]["critique_summary"] if iterations_log else "(none — first iteration of this turn)"
            # Forward open questions from Phase 3 to Phase 1 (focus guidance)
            prev_open_questions = iterations_log[-1].get("open_questions_text", "") if iterations_log else ""
            # Check if previous decision was RETHINK (fundamental errors → start over)
            rethink_mode = iterations_log[-1].get("decision", "").startswith("LOOP_RETHINK") if iterations_log else False

        # Include conversation context (prior dialogue turns) so ISRA is aware of history
        context_block = ""
        if conversation_context:
            context_block = f"\n\nCONVERSATION HISTORY (prior turns in this session):\n{conversation_context}\n"

        phase1_user = (
            f"USER QUERY: {user_query}\n"
            f"{context_block}\n"
            f"PREVIOUS INSIGHTS (VERIFIED):\n{prev_insights}\n\n"
            f"PREVIOUS CRITIQUE / FOCUS AREAS:\n{prev_critique}"
        )
        if prev_open_questions:
            phase1_user += f"\n\nOPEN QUESTIONS TO ADDRESS (from previous iteration):\n{prev_open_questions}"
        # For CODE tasks in rethink mode, include previous code so model can fix it
        # instead of regenerating from scratch
        if rethink_mode and task_type == "CODE" and best_code:
            phase1_user += (
                f"\n\nPREVIOUS CODE (has bugs — fix it):\n```python\n{best_code}\n```"
                "\n\n*** FIX MODE: Your previous code has errors. "
                "Fix the specific bugs identified in the critique above. "
                "Keep the parts that work. Do NOT rewrite from scratch. ***"
            )
        elif rethink_mode:
            phase1_user += (
                "\n\n*** RESTART MODE: The previous attempt had MAJOR errors (HIGH severity). "
                "Do NOT just refine — re-examine the problem from scratch. "
                "Discard previous insights that may be wrong and re-derive your conclusions independently. ***"
            )

        # Phase 1: Deep Thinking
        phase1_system = PHASE1_SYSTEM_GENERAL
        if task_type == "CODE":
            phase1_system = PHASE1_SYSTEM_CODE
        elif task_type == "MATH":
            phase1_system = PHASE1_SYSTEM_MATH

        # CODE tasks: disable thinking mode for Phase 1.
        # Thinking mode causes: (1) jetsam kills from large KV cache, (2) model outputs
        # reasoning text ("The user wants...") instead of actual code.
        # Code generation is direct — no extensive reasoning needed.
        phase1_thinking_override = False if task_type == "CODE" else None

        phase1_output = await call_backend(session, phase1_system, phase1_user, phase=1, enable_thinking_override=phase1_thinking_override)
        conclusions = parse_conclusions(phase1_output)
        # Extract code from Phase 1 output (for code tasks)
        phase1_code = parse_code(phase1_output)
        if phase1_code:
            best_code = phase1_code  # keep the latest code (refined across iterations)
            log.info(f"Phase 1 done: {len(conclusions)} chars conclusions, {len(phase1_code)} chars code")
        else:
            log.info(f"Phase 1 done: {len(conclusions)} chars conclusions")

        # C1: Extract numeric answer from Phase 1 [ANSWER] tag for MATH short-circuit
        if task_type == "MATH":
            answer_text = parse_answer(phase1_output)
            if answer_text:
                m = re.search(r"####\s*([\d,]+(?:\.\d+)?)", answer_text)
                if not m:
                    m = re.search(r"([\d,]+(?:\.\d+)?)", answer_text)
                if m:
                    best_math_answer = m.group(1).replace(",", "")
                    log.info(f"Extracted math answer: #### {best_math_answer}")

        # A3: Static Analysis Pre-Check for Code
        # B1: Gate on code presence (not task_type) — defense-in-depth against classifier misrouting
        syntax_error_msg = ""
        syntax_check_failed = False
        if phase1_code and ("def " in phase1_code or "class " in phase1_code):
            try:
                ast.parse(phase1_code)
            except SyntaxError as e:
                syntax_error_msg = f"[EXECUTION_ERROR] SyntaxError: {e}"
                syntax_check_failed = True
                log.info(f"Syntax error detected in iteration {iteration}: {e}")

        # A4: Execution Feedback Loop for Code (only once per pipeline, only complete programs)
        # B1: Gate on code presence (not task_type) — defense-in-depth against classifier misrouting
        execution_error_msg = ""
        if best_code and not execution_attempted and ("def " in best_code or "class " in best_code):
            success, stderr = execute_code_safely(best_code)
            execution_attempted = True
            if not success:
                execution_error_msg = f"[EXECUTION_ERROR] {stderr}"
                log.info(f"Execution failed in iteration {iteration}: {stderr[:100]}")

        # B2: Docstring Example Execution (doctest feedback)
        # Run >>> examples from the user query's docstring against the generated code
        # Skip if execution already failed (no point testing doctests on broken code)
        doctest_error_msg = ""
        if best_code and not doctest_attempted and not execution_error_msg and ("def " in best_code or "class " in best_code):
            doctest_examples = extract_doctest_examples(user_query)
            if doctest_examples:
                doctest_attempted = True
                all_passed, fail_msg = run_doctest_examples(best_code, doctest_examples)
                if not all_passed:
                    doctest_error_msg = fail_msg
                    log.info(f"Doctest failed in iteration {iteration}: {fail_msg[:100]}")
            else:
                # No doctest examples in query — generate test cases from docstring
                # This catches logic errors for tasks without >>> examples (HE/1, HE/14, HE/17)
                log.info(f"No doctest examples found — generating self-tests for CODE task")
                self_test_error = await generate_and_run_test_cases(session, user_query, best_code)
                doctest_attempted = True
                if self_test_error:
                    doctest_error_msg = self_test_error
                    log.info(f"Self-test failed in iteration {iteration}: {self_test_error[:100]}")
                else:
                    log.info(f"Self-test passed in iteration {iteration}")

        # B7: Stuck-Loop Detection via Error Pattern Repetition
        # If same error signature appears 2 consecutive iterations, terminate early
        # Use full error message (not truncated) to avoid false positives on similar-but-different errors
        raw_error = syntax_error_msg or execution_error_msg or doctest_error_msg or ""
        # Extract error type + first meaningful line (e.g., "SyntaxError: invalid syntax at line 5")
        # This avoids false positives where [:100] truncation catches only the traceback header
        err_match = re.search(r"(\w+Error:\s*[^\n]+)", raw_error)
        current_error_sig = err_match.group(1) if err_match else (raw_error[:200] if raw_error else "")
        if current_error_sig and current_error_sig == prev_error_sig:
            log.info(f"Stuck loop detected in iteration {iteration} — same error signature repeated: {current_error_sig[:80]}")
            decision = "TERMINATE_STAGNATION"
            break
        prev_error_sig = current_error_sig

        # ── DEER: Dynamic Early Exit after Phase 1 (P2) ────────────────────
        # If Phase 1 produced a confident answer AND a quick re-solve matches,
        # skip Phase 2/3/4 entirely. Saves ~15-20s for confident Phase 1 outputs.
        # Only on iteration 1 (no point in later iterations — if Phase 1 was
        # confident, we wouldn't be iterating).

        # Build Phase 2 input (needed for Phase 2 call)
        phase2_input = conclusions
        if phase1_code:
            phase2_input += f"\n\n[CODE]\n```python\n{phase1_code}\n```"
        # Pass execution/doctest errors to critic so it can review with knowledge of failures
        if doctest_error_msg:
            phase2_input += f"\n\n[EXECUTION FEEDBACK]\n{doctest_error_msg}"
        if execution_error_msg:
            phase2_input += f"\n\n[EXECUTION FEEDBACK]\n{execution_error_msg}"
        phase2_user = (
            f"USER QUERY: {user_query}\n\n"
            f"CONCLUSIONS TO AUDIT:\n{phase2_input}"
        )
        phase2_thinking = True if task_type == "MATH" else None
        phase2_system_to_use = PHASE2_SYSTEM_MATH if task_type == "MATH" else PHASE2_SYSTEM
        if task_type == "MATH":
            log.info("Enabling thinking + math critic for Phase 2 (MATH task)")

        if iteration == 1 and task_type == "MATH" and best_math_answer:
            deer_prompt = (
                f"Solve this problem independently. Show your work step by step.\n"
                f"End with #### <number>\n\n"
                f"Problem: {user_query}"
            )
            deer_output = await call_backend(
                session, PHASE1_SYSTEM_MATH, deer_prompt,
                phase=1, enable_thinking_override=False,
            )
            deer_answer_text = parse_answer(deer_output)
            deer_answer = None
            if deer_answer_text:
                m_d = re.search(r"####\s*([\d,]+(?:\.\d+)?)", deer_answer_text)
                if not m_d:
                    m_d = re.search(r"([\d,]+(?:\.\d+)?)", deer_answer_text)
                if m_d:
                    deer_answer = m_d.group(1).replace(",", "")
            if deer_answer is not None and deer_answer == best_math_answer:
                elapsed_deer = time.time() - start_time
                log.info(f"DEER early exit: Phase 1 answer {best_math_answer} matches re-solve — skipping Phase 2/3/4 ({elapsed_deer:.1f}s)")
                final_answer = f"#### {best_math_answer}"
                if stream_final and write_chunk:
                    await write_chunk(final_answer)
                decision = "DEER_EARLY_EXIT"
                return {
                    "final_answer": final_answer,
                    "iterations": 1,
                    "phases_run": ["phase1", "deer_check"],
                    "deer_exit": True,
                    "elapsed_seconds": elapsed_deer,
                    "termination_reason": "DEER_EARLY_EXIT",
                    "task_type": task_type,
                    "confidence": 90,
                }
            else:
                log.info(f"DEER check: Phase 1={best_math_answer}, re-solve={deer_answer} — mismatch, proceeding to Phase 2")

        # ── End DEER ────────────────────────────────────────────────────────

        # Phase 2: Skeptical Review
        phase2_output = await call_backend(session, phase2_system_to_use, phase2_user, phase=2, enable_thinking_override=phase2_thinking)
        critique = parse_critique(phase2_output)
        log.info(
            f"Phase 2 done: confidence={critique['confidence']}%, "
            f"issues={len(critique['issues'])}, rec={critique['recommendation']}"
        )

        # A5: Skip Phase 3 on Confident STOP
        # C9: For MATH tasks, use higher threshold (85) since critic did independent re-derivation (B3)
        # C10: Self-consistency check for MATH — before short-circuit, independently
        # re-solve and compare. If answers differ, DON'T short-circuit (critic may
        # have made the same arithmetic error as Phase 1). Fall through to Phase 3/4.
        skip_phase3_threshold = 85 if task_type == "MATH" else CONFIDENCE_THRESHOLD
        can_skip_phase3 = (
            critique["recommendation"] == "STOP"
            and not any(i["severity"] == "HIGH" for i in critique["issues"])
            and critique["confidence"] >= skip_phase3_threshold
        )
        # C10: Self-consistency for MATH — quick independent re-solve
        if can_skip_phase3 and task_type == "MATH" and best_math_answer:
            sc_prompt = (
                f"Solve this problem independently. Show your work step by step.\n"
                f"End with #### <number>\n\n"
                f"Problem: {user_query}"
            )
            sc_output = await call_backend(session, PHASE1_SYSTEM_MATH, sc_prompt, phase=1, enable_thinking_override=False)
            sc_answer_text = parse_answer(sc_output)
            sc_answer = None
            if sc_answer_text:
                m_sc = re.search(r"####\s*([\d,]+(?:\.\d+)?)", sc_answer_text)
                if not m_sc:
                    m_sc = re.search(r"([\d,]+(?:\.\d+)?)", sc_answer_text)
                if m_sc:
                    sc_answer = m_sc.group(1).replace(",", "")
            if sc_answer is not None and sc_answer == best_math_answer:
                log.info(f"MATH self-consistency OK: Phase1={best_math_answer}, re-solve={sc_answer} — verified")
                math_answer_verified = True
            elif sc_answer is not None and sc_answer != best_math_answer:
                log.info(f"MATH self-consistency MISMATCH: Phase1={best_math_answer}, re-solve={sc_answer} — NOT short-circuiting, falling through to Phase 3/4")
                can_skip_phase3 = False  # Force Phase 3/4
            else:
                log.info(f"MATH self-consistency: re-solve produced no answer, trusting critic")
                math_answer_verified = True
        elif can_skip_phase3 and task_type == "MATH" and best_math_answer:
            math_answer_verified = True

        if can_skip_phase3:
            log.info(f"Phase 2 says STOP with high confidence ({critique['confidence']}% >= {skip_phase3_threshold}) — skipping Phase 3")
            essence = {
                "constraints": _extract_constraints_from_phase1(phase1_output),
                "confirmed_insights": _parse_bullets(conclusions),
                "disputed_claims": [],
                "open_questions": [],
                "code": phase1_code,
                "raw": "",
            }
        else:
            # Phase 3: Essence Extraction
            # Pass user_query so Phase 3 can check relevance and extract constraints
            phase3_user = (
                f"ORIGINAL USER QUERY:\n{user_query}\n\n"
                f"ORIGINAL CONCLUSIONS:\n{conclusions}\n\n"
                f"CRITIC AUDIT:\n{critique['raw']}"
            )
            phase3_output = await call_backend(session, PHASE3_SYSTEM, phase3_user, phase=3)
            essence = parse_essence(phase3_output)
            log.info(
                f"Phase 3 done: {len(essence['confirmed_insights'])} confirmed, "
                f"{len(essence['disputed_claims'])} disputed, "
                f"{len(essence['open_questions'])} open"
            )

        # Accumulate state across iterations (with dedup for ALL fields)
        # Dedup confirmed insights by normalized text
        existing_confirmed_norm = {" ".join(c.split()).lower() for c in accumulated_confirmed}
        for c in essence["confirmed_insights"]:
            norm = " ".join(c.split()).lower()
            if norm and norm not in existing_confirmed_norm:
                accumulated_confirmed.append(c)
                existing_confirmed_norm.add(norm)
        # Dedup disputed claims
        existing_disputed_norm = {" ".join(c.split()).lower() for c in accumulated_disputed}
        for d in essence["disputed_claims"]:
            norm = " ".join(d.split()).lower()
            if norm and norm not in existing_disputed_norm:
                accumulated_disputed.append(d)
                existing_disputed_norm.add(norm)
        # Constraints: dedup by normalized text (constraints should not duplicate)
        existing_constraints_norm = {" ".join(c.split()).lower() for c in accumulated_constraints}
        for c in essence["constraints"]:
            norm = " ".join(c.split()).lower()
            if norm and norm not in existing_constraints_norm:
                accumulated_constraints.append(c)
                existing_constraints_norm.add(norm)

        # Build current compressed state text for stagnation check
        # Exclude "raw" field — it dominates similarity and never converges
        # Use normalized sorted structured bullets for meaningful comparison
        state_for_comparison = {
            "constraints": sorted(" ".join(c.split()).lower() for c in essence["constraints"]),
            "confirmed": sorted(" ".join(c.split()).lower() for c in essence["confirmed_insights"]),
            "disputed": sorted(" ".join(c.split()).lower() for c in essence["disputed_claims"]),
        }
        current_state_text = json.dumps(state_for_comparison, ensure_ascii=False)
        similarity = state_similarity(previous_state_text, current_state_text)
        previous_state_text = current_state_text

        # Build critique summary for next iteration's Phase 1
        # Pass structured issues (actionable) instead of raw 500-char blob
        if critique["issues"]:
            critique_summary = "\n".join(
                f"- [{i['category']}|{i['severity']}|{i['action']}] {i['description']}"
                for i in critique["issues"]
            )[:800]
        else:
            critique_summary = "No issues found. Conclusions are solid."

        # A3/A4/B2: Inject static analysis, execution, and doctest errors into critique summary
        if syntax_error_msg:
            critique_summary = syntax_error_msg + "\n" + critique_summary
        if execution_error_msg:
            critique_summary = execution_error_msg + "\n" + critique_summary
        if doctest_error_msg:
            critique_summary = doctest_error_msg + "\n" + critique_summary

        # Build open questions text for next iteration's Phase 1
        open_questions_text = "\n".join(f"- {q}" for q in essence["open_questions"]) if essence["open_questions"] else ""

        # Decision gate (computed BEFORE appending to log so it's available)
        decision = evaluate_iteration_control(
            confidence=critique["confidence"],
            issues=critique["issues"],
            iteration=iteration,
            max_iterations=MAX_ITERATIONS,
            similarity=similarity,
            recommendation=critique["recommendation"],
        )

        # A3: Override STOP if syntax check failed (force another iteration to fix syntax)
        # Skip for CODE tasks — memory guard prevents multiple iterations
        if syntax_check_failed and decision.startswith("TERMINATE") and task_type != "CODE":
            original_decision = decision
            decision = "LOOP_RETHINK"
            log.info(f"Overriding {original_decision} decision due to syntax error")

        # B2: Override STOP if doctest failed (force another iteration to fix logic)
        # For CODE tasks, allow 1 extra iteration (memory guard still limits to 2 total)
        if doctest_error_msg and decision.startswith("TERMINATE") and iteration < 2:
            original_decision = decision
            decision = "LOOP_RETHINK"
            log.info(f"Overriding {original_decision} decision due to doctest failure")

        # CODE task memory guard: limit to 1 iteration to prevent jetsam kills.
        # Multiple iterations accumulate KV cache in MLX server → memory pressure → jetsam.
        # Code generation is single-shot: Phase 1 produces code, Phase 2 reviews it.
        # Rethinking iterations rarely improve code and risk crashing the server.
        # EXCEPTION: if doctest failed, allow 1 more iteration to fix the code.
        if task_type == "CODE" and decision.startswith("LOOP_") and iteration >= 1:
            if doctest_error_msg and iteration < 2:
                log.info(f"CODE task: doctest failed, allowing 1 more iteration to fix")
            else:
                decision = "TERMINATE_SUCCESS"
                log.info(f"CODE task: limiting to {iteration} iteration(s) to avoid memory pressure")

        log.info(f"Decision: {decision} (similarity={similarity:.3f})")

        iterations_log.append({
            "iteration": iteration,
            "conclusions": conclusions[:500],
            "critique": {
                "confidence": critique["confidence"],
                "recommendation": critique["recommendation"],
                "issue_count": len(critique["issues"]),
            },
            "essence": {
                "confirmed": len(essence["confirmed_insights"]),
                "disputed": len(essence["disputed_claims"]),
                "open": len(essence["open_questions"]),
            },
            "similarity": round(similarity, 3),
            "critique_summary": critique_summary,
            "open_questions_text": open_questions_text,
            "decision": decision,
        })

        # B8: Track best-confidence iteration for fallback
        exec_success_this_iter = not (syntax_error_msg or execution_error_msg or doctest_error_msg)
        if critique["confidence"] > best_iteration_confidence:
            best_iteration_confidence = critique["confidence"]
            best_iteration_index = iteration
            best_iteration_state = {
                "accumulated_confirmed": list(accumulated_confirmed),
                "accumulated_disputed": list(accumulated_disputed),
                "accumulated_constraints": list(accumulated_constraints),
                "best_code": best_code,
                "exec_success": exec_success_this_iter,
            }

        if decision.startswith("TERMINATE"):
            break

    # B8: Best-Guess Fallback for Non-Converging Tasks
    # If TERMINATE_MAX_ITER and an earlier iteration was materially better, restore it
    if decision == "TERMINATE_MAX_ITER" and best_iteration_state is not None and iterations_log:
        last_confidence = iterations_log[-1]["critique"]["confidence"]
        last_exec_success = not (syntax_error_msg or execution_error_msg or doctest_error_msg)
        best_exec_success = best_iteration_state.get("exec_success", False)
        confidence_delta = best_iteration_confidence - last_confidence
        # Restore if confidence delta >= 15pp, OR (for code) best iteration had exec success and last didn't
        if confidence_delta >= 15 or (best_code and best_exec_success and not last_exec_success):
            log.info(f"B8: Best-guess fallback — using iteration {best_iteration_index} (conf {best_iteration_confidence}) instead of last (conf {last_confidence}, delta {confidence_delta})")
            accumulated_confirmed = best_iteration_state["accumulated_confirmed"]
            accumulated_disputed = best_iteration_state["accumulated_disputed"]
            accumulated_constraints = best_iteration_state["accumulated_constraints"]
            if best_iteration_state["best_code"]:
                best_code = best_iteration_state["best_code"]

    # Phase 4: Final Synthesis
    log.info("--- Phase 4: Final Synthesis ---")
    all_confirmed = "\n".join(f"- {i}" for i in accumulated_confirmed) or "(none)"
    all_disputed = "\n".join(f"- {d}" for d in accumulated_disputed) or "(none)"
    all_constraints = "\n".join(f"- {c}" for c in accumulated_constraints) or "(none)"

    # Build Phase 4 input — include code if available (for code tasks)
    phase4_user = (
        f"USER QUERY: {user_query}\n\n"
        f"CONSTRAINTS (the answer MUST satisfy these):\n{all_constraints}\n\n"
        f"ACCUMULATED CONFIRMED INSIGHTS:\n{all_confirmed}\n\n"
        f"ACCUMULATED DISPUTED CLAIMS:\n{all_disputed}"
    )
    if best_code:
        phase4_user += f"\n\n[CODE]\n```python\n{best_code}\n```"
        log.info(f"Phase 4 input includes {len(best_code)} chars of code")

    final_answer = ""
    # CODE TASKS: if we have code from Phase 1, return it directly.
    # Phase 4 (qwen3 with thinking ON) tends to return prose instead of code,
    # which destroys the code signal. Bypass Phase 4 for code tasks.
    if best_code and ("def " in best_code or "import " in best_code or "return " in best_code or "class " in best_code):
        log.info("Code task detected — bypassing Phase 4, returning code directly")
        final_answer = f"```python\n{best_code}\n```"
        if stream_final and write_chunk:
            await write_chunk(final_answer)
    elif task_type == "MATH" and best_math_answer and math_answer_verified:
        # C1: MATH answer short-circuit — Phase 1 produced a clean #### N
        # AND Phase 2 verified it (STOP + high confidence). Bypass Phase 4.
        log.info(f"MATH task with verified answer — bypassing Phase 4, returning #### {best_math_answer}")
        final_answer = f"#### {best_math_answer}"
        if stream_final and write_chunk:
            await write_chunk(final_answer)
    elif stream_final and write_chunk:
        final_answer = await stream_backend(
            session, PHASE4_SYSTEM, phase4_user, phase=4, write_chunk=write_chunk
        )
        # C2: Post-process MATH answers to ensure #### N format
        if task_type == "MATH":
            final_answer = _sanitize_math_answer(final_answer, best_math_answer)
    else:
        raw_answer = await call_backend(session, PHASE4_SYSTEM, phase4_user, phase=4)
        # Strip think blocks from final answer (Phase 4 has thinking enabled
        # for re-reasoning, but the user should only see the final answer)
        final_answer = re.sub(r"<think>.*?</think>", "", raw_answer, flags=re.DOTALL | re.IGNORECASE).strip()
        # If everything was in think (no content after), use the raw answer stripped of tags
        if not final_answer:
            final_answer = re.sub(r"</?think>", "", raw_answer, flags=re.IGNORECASE).strip()
        # C2: Post-process MATH answers to ensure #### N format
        if task_type == "MATH":
            final_answer = _sanitize_math_answer(final_answer, best_math_answer)

    elapsed = time.time() - start_time
    log.info(f"Pipeline complete: {elapsed:.1f}s, {len(iterations_log)} iterations")

    # Persist accumulated insights back to session state for future turns
    if session_state is not None:
        # Cap stored insights to avoid unbounded growth (keep most recent 20)
        session_state["confirmed_insights"] = accumulated_confirmed[-20:]
        session_state["disputed_claims"] = accumulated_disputed[-20:]
        session_state["constraints"] = accumulated_constraints[-30:]
        log.info(
            f"Session state saved: {len(session_state['confirmed_insights'])} confirmed, "
            f"{len(session_state['disputed_claims'])} disputed, "
            f"{len(session_state['constraints'])} constraints"
        )

    return {
        "final_answer": final_answer,
        "iterations": len(iterations_log),
        "elapsed_seconds": round(elapsed, 1),
        "iterations_log": iterations_log,
        "termination_reason": decision,
    }


def _gc_cleanup():
    """Force garbage collection after each pipeline run to prevent memory accumulation.
    Python's allocator doesn't always return memory to OS — gc.collect() helps."""
    import gc
    gc.collect()


# ─── HTTP Handlers (OpenAI-compatible) ──────────────────────────────────────

async def handle_chat_completions(request: web.Request) -> web.StreamResponse:
    """OpenAI-compatible /v1/chat/completions endpoint."""
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": {"message": "Invalid JSON body"}}, status=400)

    messages = body.get("messages", [])
    if not messages:
        return web.json_response({"error": {"message": "No messages provided"}}, status=400)

    # Normalize message content (handle multimodal list format)
    def norm_content(content):
        if isinstance(content, list):
            return " ".join(p.get("text", "") for p in content if p.get("type") == "text")
        return content or ""

    # Separate system messages (passed to pipeline as system prompt prefix — ignored,
    # ISRA uses its own phase-specific system prompts) from the conversation.
    # The LAST user message is the current query; everything before it is history.
    conv_messages = [m for m in messages if m.get("role") in ("user", "assistant")]
    if not conv_messages:
        return web.json_response({"error": {"message": "No user/assistant messages"}}, status=400)

    # Find the last user message — that's the current query
    last_user_idx = None
    for i in range(len(conv_messages) - 1, -1, -1):
        if conv_messages[i].get("role") == "user":
            last_user_idx = i
            break
    if last_user_idx is None:
        return web.json_response({"error": {"message": "No user message"}}, status=400)

    user_query = norm_content(conv_messages[last_user_idx].get("content", "")).strip()
    if not user_query:
        return web.json_response({"error": {"message": "Empty user message"}}, status=400)

    # Build conversation context from prior turns (everything before the last user message)
    history_msgs = conv_messages[:last_user_idx]
    context_lines = []
    for m in history_msgs:
        role = m.get("role", "user")
        text = norm_content(m.get("content", ""))
        if not text:
            continue
        label = "User" if role == "user" else "Assistant"
        # Truncate long assistant messages to keep context budget manageable
        if len(text) > 800:
            text = text[:800] + " [...]"
        context_lines.append(f"{label}: {text}")
    conversation_context = "\n".join(context_lines) if context_lines else ""

    # Session ID — from X-Session-Id header, or derived from conversation hash
    # so that goose/pi (which don't send custom headers) still get session continuity
    # across messages in the same conversation.
    session_id = request.headers.get("X-Session-Id", "")
    if not session_id:
        # Derive a stable session id from the first history message (or current
        # user query on the first turn) so the same conversation maps to the same
        # session across requests. Seeding from history_msgs[:2] would break
        # continuity: turn 1 seeds from [user_query], turn 2 from [u1, a1] →
        # different hashes → session state never retrieved.
        if history_msgs:
            seed = norm_content(history_msgs[0].get("content", "")).strip()[:100]
        else:
            seed = user_query[:100]
        session_id = "conv-" + hashlib.sha256(seed.encode()).hexdigest()[:16]

    # Periodic cleanup
    _cleanup_sessions()
    session_state = _get_session(session_id)
    log.info(
        f"Session {session_id}: query={user_query[:80]!r}, "
        f"history={len(history_msgs)} msgs, context={len(conversation_context)} chars"
    )

    stream = body.get("stream", False)
    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    http_session = request.app["http_session"]

    # A8: Task-Aware Router — bypass ISRA for trivial queries
    if is_trivial_query(user_query):
        log.info(f"Trivial query detected — bypassing ISRA pipeline for {user_query[:80]!r}")
        trivial_system = "You are a helpful assistant. Answer the user's question directly."
        raw_trivial = await call_backend(http_session, trivial_system, user_query, phase=4)
        trivial_answer = re.sub(r"<think>.*?</think>", "", raw_trivial, flags=re.DOTALL | re.IGNORECASE).strip()
        if not trivial_answer:
            trivial_answer = re.sub(r"</?think>", "", raw_trivial, flags=re.IGNORECASE).strip()

        if stream:
            resp = web.StreamResponse(
                status=200,
                headers={
                    "Content-Type": "text/event-stream",
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )
            await resp.prepare(request)
            chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": ISRA_MODEL_ID,
                "choices": [{"index": 0, "delta": {"content": trivial_answer}, "finish_reason": None}],
            }
            await resp.write(f"data: {json.dumps(chunk)}\n\n".encode())
            final_chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": ISRA_MODEL_ID,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            await resp.write(f"data: {json.dumps(final_chunk)}\n\n".encode())
            await resp.write(b"data: [DONE]\n\n")
            await resp.write_eof()
            return resp
        else:
            response = {
                "id": request_id,
                "object": "chat.completion",
                "created": created,
                "model": ISRA_MODEL_ID,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": trivial_answer},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "isra_metadata": {"trivial_bypass": True, "elapsed_seconds": 0},
            }
            return web.json_response(response)

    if stream:
        # Streaming response — stream Phase 4 output as SSE
        resp = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )
        await resp.prepare(request)

        async def write_chunk(content: str):
            chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": ISRA_MODEL_ID,
                "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
            }
            await resp.write(f"data: {json.dumps(chunk)}\n\n".encode())

        try:
            result = await run_isra_pipeline(
                http_session, user_query,
                conversation_context=conversation_context,
                session_state=session_state,
                stream_final=True, write_chunk=write_chunk,
            )

            # Send final chunk with finish_reason
            final_chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": ISRA_MODEL_ID,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            await resp.write(f"data: {json.dumps(final_chunk)}\n\n".encode())
            await resp.write(b"data: [DONE]\n\n")
            await resp.write_eof()

            log.info(
                f"Streamed response: {result['iterations']} iters, "
                f"{result['elapsed_seconds']}s, reason={result['termination_reason']}"
            )
        except Exception as e:
            log.error(f"Pipeline error during streaming: {e}")
            error_chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": ISRA_MODEL_ID,
                "choices": [{"index": 0, "delta": {"content": f"\n\n[ISRA Error: {e}]"}, "finish_reason": "stop"}],
            }
            await resp.write(f"data: {json.dumps(error_chunk)}\n\n".encode())
            await resp.write(b"data: [DONE]\n\n")
            await resp.write_eof()
        finally:
            _gc_cleanup()
        return resp

    else:
        # Non-streaming response
        try:
            result = await run_isra_pipeline(
                http_session, user_query,
                conversation_context=conversation_context,
                session_state=session_state,
                stream_final=False,
            )

            response = {
                "id": request_id,
                "object": "chat.completion",
                "created": created,
                "model": ISRA_MODEL_ID,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": result["final_answer"]},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "isra_metadata": {
                    "session_id": session_id,
                    "iterations": result["iterations"],
                    "elapsed_seconds": result["elapsed_seconds"],
                    "termination_reason": result["termination_reason"],
                    "history_msgs": len(history_msgs),
                },
            }
            return web.json_response(response)
        except Exception as e:
            log.error(f"Pipeline error: {e}")
            return web.json_response(
                {"error": {"message": f"ISRA pipeline error: {e}"}}, status=502
            )
        finally:
            _gc_cleanup()


async def handle_models(request: web.Request) -> web.Response:
    """OpenAI-compatible /v1/models endpoint."""
    return web.json_response({
        "object": "list",
        "data": [
            {
                "id": ISRA_MODEL_ID,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "isra",
            }
        ],
    })


async def handle_health(request: web.Request) -> web.Response:
    """Health endpoint."""
    _cleanup_sessions()
    return web.json_response({
        "status": "ok",
        "service": "isra-orchestrator",
        "port": PORT,
        "backend_model": BACKEND_MODEL,
        "router_url": ROUTER_URL,
        "active_sessions": len(_sessions),
        "config": {
            "max_iterations": MAX_ITERATIONS,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "stagnation_threshold": STAGNATION_THRESHOLD,
            "session_ttl_seconds": SESSION_TTL,
        },
    })


async def handle_sessions(request: web.Request) -> web.Response:
    """List active sessions (for debugging/monitoring)."""
    now = time.time()
    sessions_info = []
    for sid, s in _sessions.items():
        sessions_info.append({
            "session_id": sid,
            "confirmed_insights": len(s["confirmed_insights"]),
            "disputed_claims": len(s["disputed_claims"]),
            "constraints": len(s.get("constraints", [])),
            "idle_seconds": int(now - s["last_active"]),
        })
    return web.json_response({"sessions": sessions_info, "total": len(sessions_info)})


async def handle_session_clear(request: web.Request) -> web.Response:
    """Clear a specific session by ID (?session_id=...) or all sessions."""
    sid = request.query.get("session_id", "")
    if sid:
        if sid in _sessions:
            del _sessions[sid]
            return web.json_response({"ok": True, "cleared": sid})
        return web.json_response({"ok": False, "error": "session not found"}, status=404)
    n = len(_sessions)
    _sessions.clear()
    return web.json_response({"ok": True, "cleared_all": n})


async def handle_root(request: web.Request) -> web.Response:
    """Root endpoint — basic info."""
    return web.json_response({
        "service": "ISRA Orchestrator",
        "endpoints": [
            "/v1/chat/completions",
            "/v1/models",
            "/health",
            "/sessions",
            "/sessions/clear?session_id=...",
        ],
        "model": ISRA_MODEL_ID,
    })


# ─── App lifecycle ──────────────────────────────────────────────────────────

async def on_startup(app: web.Application) -> None:
    # force_close=True: close connection after each request (prevents KV cache accumulation in connection pool)
    # limit=1: only 1 concurrent connection (prevents memory growth from multiple pooled connections)
    connector = aiohttp.TCPConnector(force_close=True, limit=1, use_dns_cache=False)
    app["http_session"] = ClientSession(timeout=BACKEND_TIMEOUT, connector=connector)
    log.info(f"ISRA Orchestrator starting on {HOST}:{PORT}")
    log.info(f"Backend: {BACKEND_MODEL} via {ROUTER_URL}")
    log.info(
        f"Config: max_iters={MAX_ITERATIONS}, "
        f"confidence_threshold={CONFIDENCE_THRESHOLD}, "
        f"stagnation={STAGNATION_THRESHOLD}"
    )


async def on_cleanup(app: web.Application) -> None:
    session = app.get("http_session")
    if session and not session.closed:
        await session.close()
    log.info("ISRA Orchestrator shutting down")


def create_app() -> web.Application:
    app = web.Application()
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    app.router.add_post("/v1/chat/completions", handle_chat_completions)
    app.router.add_post("/v1/completions", handle_chat_completions)  # alias
    app.router.add_get("/v1/models", handle_models)
    app.router.add_get("/health", handle_health)
    app.router.add_get("/sessions", handle_sessions)
    app.router.add_get("/sessions/clear", handle_session_clear)
    app.router.add_get("/", handle_root)
    return app


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="ISRA Orchestrator")
    parser.add_argument("--host", default=HOST, help="Bind host")
    parser.add_argument("--port", type=int, default=PORT, help="Bind port")
    args = parser.parse_args()

    app = create_app()
    web.run_app(app, host=args.host, port=args.port, access_log=None)


if __name__ == "__main__":
    main()

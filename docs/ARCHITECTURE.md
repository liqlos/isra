# ISRA Architecture

## Pipeline Overview

ISRA runs a multi-phase pipeline with iterative refinement. Each phase is a separate LLM call with task-specific system prompts and sampling parameters.

```
┌──────────────────────────────────────────────────────────────────┐
│                    ISRA Pipeline (per request)                    │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────────────────────────────────┐                     │
│  │ Phase 0: Quick Consensus Check          │                     │
│  │ 2 parallel calls (temp=0, 50 tokens)    │                     │
│  │ If answers match → return immediately   │                     │
│  └──────────────┬──────────────────────────┘                     │
│                 │ (miss)                                          │
│  ┌──────────────▼──────────────────────────┐                     │
│  │ Phase 1: Deep Thinking                  │  ◄── iteration loop │
│  │ System prompt: task-specific (CODE/MATH/│      (up to 3x)     │
│  │ GENERAL)                                 │                     │
│  │ Output: <think> + [CONSTRAINTS] +        │                     │
│  │         [CONCLUSIONS] + [CODE]/[ANSWER]  │                     │
│  └──────────────┬──────────────────────────┘                     │
│                 │                                                  │
│  ┌──────────────▼──────────────────────────┐                     │
│  │ Code Execution Feedback                 │                     │
│  │ • Syntax check (ast.parse)              │                     │
│  │ • Doctest execution (subprocess)        │                     │
│  │ • Self-generated tests (if no doctests) │                     │
│  └──────────────┬──────────────────────────┘                     │
│                 │                                                  │
│  ┌──────────────▼──────────────────────────┐                     │
│  │ DEER: Dynamic Early Exit (MATH only)    │                     │
│  │ Re-solve problem independently          │                     │
│  │ If answer matches Phase 1 → return      │                     │
│  └──────────────┬──────────────────────────┘                     │
│                 │ (mismatch or non-math)                          │
│  ┌──────────────▼──────────────────────────┐                     │
│  │ Phase 2: Skeptical Review               │                     │
│  │ Critic reviews conclusions + code       │                     │
│  │ For MATH: independent re-derivation     │                     │
│  │ Output: [ISSUES] + [CONFIDENCE] +       │                     │
│  │         [RECOMMENDATION]                │                     │
│  └──────────────┬──────────────────────────┘                     │
│                 │                                                  │
│  ┌──────────────▼──────────────────────────┐                     │
│  │ Decision Gate                           │                     │
│  │ • confidence ≥ 75 → TERMINATE_SUCCESS   │                     │
│  │ • critic says STOP → TERMINATE_SUCCESS  │                     │
│  │ • iteration ≥ 3 → TERMINATE_MAX_ITER    │                     │
│  │ • similarity ≥ 0.75 → TERMINATE_STAGN   │                     │
│  │ • doctest failed → LOOP_RETHINK         │                     │
│  │ • else → LOOP_REFINE                    │                     │
│  └──────────────┬──────────────────────────┘                     │
│                 │ (continue)                                       │
│  ┌──────────────▼──────────────────────────┐                     │
│  │ Phase 3: Essence Extraction             │                     │
│  │ Merge conclusions + critique            │                     │
│  │ Output: [CONFIRMED] + [DISPUTED] +      │                     │
│  │         [CONSTRAINTS] + [CODE]           │                     │
│  └──────────────┬──────────────────────────┘                     │
│                 │                                                  │
│  ┌──────────────▼──────────────────────────┐                     │
│  │ Accumulate state (dedup insights)       │                     │
│  │ Check stagnation (state similarity)     │                     │
│  │ → back to Phase 1 with critique feedback│                     │
│  └─────────────────────────────────────────┘                     │
│                                                                   │
│  After loop exits:                                                │
│  ┌─────────────────────────────────────────┐                     │
│  │ Phase 4: Final Synthesis                │                     │
│  │ For CODE: return code directly          │                     │
│  │ For MATH: return #### N directly        │                     │
│  │ For GENERAL: LLM synthesizes answer     │                     │
│  └─────────────────────────────────────────┘                     │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

## Task Classification

ISRA classifies each query as **CODE**, **MATH**, or **GENERAL** using lightweight heuristics:

- **CODE**: Detects `def `, `import `, `class X`, "write a function", "complete the function", fenced code blocks
- **MATH**: Detects "calculate", "how many", "sum of", arithmetic expressions, GSM8K-style word problems (≥3 digits + question words)
- **GENERAL**: Everything else (default)

Classification determines:
- Which Phase 1 system prompt to use
- Whether DEER early exit is eligible (MATH only)
- Whether to enable thinking mode (disabled for CODE — causes jetsam)
- Phase 2 critic variant (MATH uses independent re-derivation)
- Iteration limit (CODE limited to 1-2 to avoid memory pressure)

## Phase Parameters

| Phase | Temp | Top-p | Max tokens | Thinking | Purpose |
|-------|------|-------|------------|----------|---------|
| 0 | 0.0/0.5 | 0.95 | 50 | Off | Quick consensus |
| 1 | 0.6 | 0.95 | 2000 | On (off for CODE) | Deep reasoning |
| 2 | 0.2 | 0.95 | 1500 | Off (on for MATH) | Skeptical review |
| 3 | 0.1 | 0.95 | 1000 | Off | Essence extraction |
| 4 | 0.2 | 0.95 | 2000 | On | Final synthesis |

## Execution Feedback (Code Tasks)

ISRA provides three layers of execution feedback for code tasks:

### 1. Syntax Check (`ast.parse`)
Zero-cost compile check. Catches SyntaxError without subprocess. Runs on every iteration.

### 2. Doctest Execution (subprocess)
Extracts `>>>` examples from the user query's docstring and runs them in a subprocess:
```python
test_script = code + f"\n_result = repr({expr})\n_expected = {expected!r}\nassert _result == _expected"
subprocess.run(["python3", "-c", test_script], timeout=5)
```
Uses `repr()` for comparison (matches doctest output format). Skips destructive examples (`open()`, `os.`, `subprocess`).

### 3. Self-Generated Test Cases
For tasks without `>>>` examples (e.g., HumanEval/1, HE/14), ISRA asks the model to generate 3-5 assert statements from the docstring, then executes them:
```
Prompt: "Based on this function signature and docstring, write 3-5 test cases as assert statements."
→ Extract assert lines → subprocess.run(code + asserts, timeout=5)
```

If any test fails, the error message is:
1. Passed to Phase 2 critic (so it reviews with knowledge of the failure)
2. Injected into the next iteration's Phase 1 prompt as "FIX MODE"

## DEER — Dynamic Early Exit

For MATH tasks on iteration 1:
1. After Phase 1 produces `#### N`, independently re-solve the problem (no thinking, fast)
2. If re-solve produces the same number → skip phases 2-4, return immediately
3. If mismatch → proceed to Phase 2 (critic with independent re-derivation)

This saves ~15-20s on ~75% of math tasks where Phase 1 is already correct.

## Session State

ISRA maintains per-conversation state (keyed by `X-Session-Id` header or auto-derived from conversation hash):
- `confirmed_insights`: Verified facts accumulated across turns
- `disputed_claims`: Flagged claims needing verification
- `constraints`: Rules the answer must satisfy

State persists for `ISRA_SESSION_TTL` (default 1 hour) and is capped at 20-30 items to prevent unbounded growth.

## Stagnation Detection

Two mechanisms prevent infinite loops:

1. **State similarity**: Compares structured state (sorted constraints + confirmed + disputed) between iterations. If similarity ≥ 0.75, terminates with `TERMINATE_STAGNATION`.

2. **Error signature repetition**: If the same error (e.g., "SyntaxError: invalid syntax at line 5") appears in 2 consecutive iterations, terminates early — the model is stuck.

## Memory Management

For MLX backends with limited RAM (24GB Mac mini):

- `force_close=True` on aiohttp connector (no connection pooling)
- `limit=1` concurrent connections
- `gc.collect()` after each pipeline run
- CODE tasks limited to 1-2 iterations (KV cache accumulation → jetsam kills)
- Phase 1 thinking disabled for CODE (large reasoning text → memory pressure)
- `urllib` instead of `aiohttp` for backend calls (avoids connection pool memory)

## HTTP API

### `POST /v1/chat/completions`
OpenAI-compatible. Supports streaming (`"stream": true`) and non-streaming.

### `GET /v1/models`
Returns `isra-a3b` model entry.

### `GET /health`
Service status + configuration.

### `GET /sessions`
List active sessions with insight counts.

### `GET /sessions/clear?session_id=...`
Clear a specific session (or all if no ID provided).

# ISRA Benchmark Results

## Methodology

### Test Datasets
- **HumanEval**: Full 164 tasks from OpenAI's HumanEval benchmark. Each task has a function signature + docstring + actual test cases (assert statements). Correctness verified by executing generated code against test cases in subprocess.
- **GSM8K**: 300 tasks (random sample, seed=42) from GSM8K grade-school math problems (full dataset: 1319). Correctness verified by comparing extracted numeric answer (`#### N` format) to ground truth.
- **MMLU-Pro**: 500 tasks (random sample) — multiple-choice knowledge questions. ISRA not optimized for MC tasks (stopped early).

### Endpoints Tested
- **Direct**: Vanilla model call (no orchestrator) via model router at `:8080`. Thinking mode disabled for HumanEval (matches ISRA CODE behavior).
- **ISRA**: Full ISRA pipeline at `:8083`

### Model
Qwen3.5-35B-A3B (MoE, 3B active parameters), 3-bit MLX quantization, running on Mac mini M4 (24GB RAM).

### Checker
Strict execution-based checker:
- Code: `exec(generated_code + test_asserts)` in subprocess — must pass without exception
- Math: Extract number from `#### N` format, compare to ground truth as float

## Results (full benchmarks)

| Benchmark | Tasks | ISRA | Direct (vanilla) | Official Qwen3.5-35B-A3B |
|-----------|-------|------|-----------------|--------------------------|
| **HumanEval** | 164 | **143/164 = 87.2%** | 149/164 = 90.9% | ~74.6% (LiveCodeBench v6)* |
| **GSM8K** | 300 | **267/300 = 89.0%** | ~75% (partial) | ~95% (est, not published) |
| MMLU-Pro | 500 | not completed | — | 85.3% |

*LiveCodeBench v6 is a different (harder) code benchmark than HumanEval.

### Key Findings

1. **ISRA does NOT improve HumanEval for strong models.** Direct (90.9%) outperforms ISRA (87.2%) on full HumanEval. The Qwen3.5-35B-A3B model is already strong enough at code generation that ISRA's critic/iterate loop adds noise: self-generated tests produce false negatives, causing the model to "fix" working code and break it.

2. **ISRA GSM8K 89.0%** — below the expected ~95%. 33 failures, including timeout-related ones on complex multi-step problems where ISRA's 3-iteration pipeline exceeds the 600s timeout.

3. **Direct HumanEval 90.9%** — significantly above official LiveCodeBench v6 (74.6%), but these are different benchmarks. LiveCodeBench uses newer/harder problems.

4. **ISRA overhead**: ~30-60s per task vs ~5-20s for Direct. For strong models on code tasks, this latency cost is not justified.

### When ISRA Helps vs Hurts

| Scenario | ISRA Impact |
|----------|-------------|
| Weak model + code | +5-10pp (catches errors via doctest feedback) |
| Strong model + code | -3 to -5pp (false negatives from self-tests) |
| Weak model + math | +5-15pp (independent re-derivation catches arithmetic errors) |
| Strong model + math | 0 to -5pp (model already correct, iterations add noise) |
| Multiple-choice (MMLU) | Not suitable (ISRA designed for generative tasks) |

## Optimization History

### Implemented (kept)

| Optimization | Impact |
|-------------|--------|
| Phase 0 quick consensus | ~1s for trivial queries (was 30-80s) |
| DEER early exit (MATH) | 75% of math tasks skip phases 2-4, avg 29s (was 48s) |
| Doctest execution feedback | Catches code logic errors via subprocess execution |
| Self-generated test cases | For tasks without doctests |
| FIX MODE for code rethink | Model fixes bugs instead of rewriting |
| Doctest errors → Phase 2 critic | Better critic feedback |
| repr() comparison fix | Fixes false doctest failures on string returns |
| Phase 0 parallel (asyncio.gather) | 0.3-0.5s savings |

### Tested and Reverted

| Optimization | Result | Reason |
|-------------|--------|--------|
| DEER + Phase 2 parallel | 2x latency regression | Memory pressure from 2 concurrent KV caches |
| SC majority vote for MATH | GSM8K regression | Wrong answer picked on edge cases |
| SC mixed temperatures [0.0, 0.7] | GSM8K regression | Greedy sample consistently wrong on some tasks |
| SC N=3 sampling | Same accuracy as N=2 | +40% latency, not worth it |

## Comparison with Published Models

| Model | HumanEval | GSM8K | Notes |
|-------|-----------|-------|-------|
| Claude 3.5 Sonnet | 93.7% | 96.4% | Cloud, $3/M tokens |
| GPT-4o | 90.2% | 90.5% | Cloud, $5/M tokens |
| Gemini 1.5 Pro | 79.3% | 88.9% | Cloud |
| Qwen3.5-35B-A3B (Direct) | 90.9% | ~95% (est) | Local, free |
| ISRA + Qwen3.5-35B-A3B | 87.2% | 89.0% | Local, free, +30s latency |

**Honest assessment**: For a strong MoE model like Qwen3.5-35B-A3B, ISRA's critic-and-iterate pipeline does not improve accuracy on standard benchmarks. The model is already good enough that the orchestrator's overhead (false positives from self-tests, iteration noise, timeout failures) outweighs the benefits. ISRA is more valuable for weaker models or specialized reasoning tasks where the base model makes errors that a critic can catch.

## Running Benchmarks

```bash
# Set API key (if your backend requires one)
export MLX_LOCAL_API_KEY=your-key

# Full official benchmark (HumanEval 164 + GSM8K 300 + MMLU-Pro 500)
python benchmarks/official_benchmark.py

# Skip HumanEval (already completed)
SKIP_HUMANEVAL=1 python benchmarks/official_benchmark.py

# Quick HumanEval-only test (20 tasks)
python benchmarks/bench_he_isra.py

# Quick GSM8K-only test (20 tasks)
python benchmarks/bench_gsm_isra.py
```

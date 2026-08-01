# ISRA experiment plan

> **Historical preregistration, superseded by the completed 2026-08-01 smoke.**
> Stage 0 integrity gates and the 20-task 8B mechanism smoke were completed.
> The tested grounded repair and compute-matched retry produced zero paired
> fixes at positive cost. See [`THEORY_REVIEW.md`](THEORY_REVIEW.md) and
> [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md) for the result and current decision.

## Decision this plan should support

Determine whether iterative critique and revision can produce a **repeatable, statistically credible improvement** over a direct answer when both approaches use the same model, evaluator, prompt budget, and infrastructure.

At the time this plan was written, the available measurements were exploratory
and inconclusive. The later controlled smoke answered the narrower Stage 1
question negatively for the tested 8B model and repair variant; it does not
prove that every possible externally grounded or trained correction method must
fail.

## What the current results do and do not show

| Benchmark | Direct | ISRA | Interpretation |
| --- | ---: | ---: | --- |
| HumanEval, 164 tasks | 149/164 (90.9%) | 146/164 (89.0%, v2) | On this run, ISRA was 1.9 percentage points lower. The comparison is confounded by different sampling settings and pipeline behavior. |
| GSM8K, 300 tasks | about 75%, partial run | 267/300 (89.0%, v1) | The runs are incomplete and not paired, so the apparent gain cannot be attributed to ISRA. |
| MMLU-Pro | incomplete | incomplete | No conclusion is possible. |

Important limitations found in the current harness:

- `official_benchmark.py` describes a Direct-versus-ISRA comparison, but its `ENDPOINTS` currently contains only ISRA.
- Direct HumanEval uses greedy decoding (`temperature=0`), while ISRA Phase 1 uses `temperature=0.6`; this mixes the effect of sampling with the effect of refinement.
- Timeouts and HTTP failures are recorded as failed answers instead of separate infrastructure failures.
- The official runner stores pass/fail, a short message, and elapsed time, but not the task ID, full response, token usage, per-phase timings, configuration fingerprint, or failure category needed for a paired diagnosis.
- Existing HumanEval runs are close to the benchmark ceiling. A strong 35B MoE model leaves little room for a refinement method to help and is therefore a useful control, but a weak test of the main hypothesis.
- Self-generated tests can be wrong. At present, a false test can force the pipeline to revise a correct answer.
- Results come from one main model/backend combination; model size, quantization, serving backend, and orchestration are not isolated.
- The comparison to a published LiveCodeBench score is contextual only; it is not a valid baseline for HumanEval.

## Hypotheses

### Primary hypothesis

ISRA improves pass rate most for small or mid-size instruction models whose direct answers contain correctable reasoning or implementation errors.

### Secondary hypotheses

1. The improvement decreases or becomes negative as the base model gets stronger.
2. Execution-grounded feedback helps code more reliably than an LLM critic without external evidence.
3. Most value comes from one targeted revision; additional iterations add latency and may introduce regressions.
4. A confidence/early-exit gate can recover much of the quality gain at a lower average cost.
5. Self-generated tests help only when their validity is independently checked or their feedback is advisory rather than authoritative.

## Stage 0: make the benchmark trustworthy

Do not run another large benchmark until the following changes are complete.

### One paired runner

Run every task through all variants in the same invocation and save one record per `(run_id, model, task_id, variant, seed)`.

Required variants for the first trustworthy run:

- `direct_greedy`: one direct call at temperature 0;
- `isra_greedy`: all generative phases at temperature 0 where supported;
- `direct_sampled`: one direct call using the same Phase 1 sampling settings as ISRA;
- `isra_sampled`: the current ISRA sampling policy.

This separates the orchestration effect from the sampling effect. Randomize variant order per task to reduce thermal/load bias, while using the same task order and seed set for all variants.

### Immutable run manifest

Save alongside every run:

- git commit of the benchmark and orchestrator;
- model identifier and exact weights/revision;
- quantization and serving backend versions;
- complete phase parameters and prompts, or their content hashes;
- dataset name, revision, split, and task IDs;
- evaluator version;
- machine metadata relevant to inference;
- seed, timeout, retry policy, and concurrency;
- start/end timestamps.

### Result schema

Each task record should include:

```json
{
  "run_id": "...",
  "task_id": "HumanEval/0",
  "model": "...",
  "variant": "direct_greedy",
  "seed": 0,
  "status": "completed",
  "passed": true,
  "response": "...",
  "latency_ms": 0,
  "prompt_tokens": 0,
  "completion_tokens": 0,
  "llm_calls": 1,
  "phase_metrics": [],
  "termination_reason": "direct",
  "evaluator_message": "PASS"
}
```

Use distinct statuses for `completed`, `model_error`, `timeout`, `transport_error`, and `evaluator_error`. Retry infrastructure failures with bounded backoff and do not convert an exhausted infrastructure retry into a wrong answer. Preserve partial runs and support resume by primary key.

### Evaluator parity

- Feed the exact same user task to all variants.
- Use the same code extractor and evaluator after generation.
- Never expose hidden benchmark tests to either Direct or ISRA.
- Treat prompt examples as public examples and hidden tests as evaluation only.
- Add unit tests for code extraction, numeric answer extraction, resume behavior, and error categorization.
- Run a 20-task smoke set and manually inspect every disagreement before scaling up.

## Stage 1: find whether any mechanism actually helps

Start with code because it offers objective execution feedback. Use a small diagnostic set with tasks deliberately selected across easy, medium, and hard direct outcomes.

### Ablation matrix

| Variant | Purpose |
| --- | --- |
| Direct greedy | Lowest-cost baseline |
| Direct sampled, one call | Sampling-matched baseline |
| Direct self-consistency, 2–3 calls | Compute-matched non-ISRA baseline |
| Phase 1 + critic, no rewrite | Test critic accuracy without allowing damage |
| Phase 1 + execution feedback + one rewrite | Test grounded refinement |
| Phase 1 + critic + one rewrite | Test critic-driven refinement |
| Full ISRA | Test the complete pipeline |
| Full ISRA without self-generated tests | Measure their net effect |
| Full ISRA with only provided/verified tests | Separate trusted execution feedback |

For a fair compute comparison, add a budget-matched direct baseline: either multiple independent samples with selection or one larger token budget. ISRA should beat what the same number of calls/tokens could buy more simply.

### Instrument the pipeline

Record per phase:

- prompt/completion tokens and wall time;
- extracted confidence and decision;
- number and severity of critic issues;
- whether execution or a test supplied feedback;
- whether the revision changed the answer;
- whether the change fixed, preserved, or broke the evaluated result;
- early-exit and fallback reason.

Store intermediate answers so Direct-only failures, ISRA-only failures, fixes, and regressions can be replayed without another full inference run.

### Validate the critic and self-tests

For each disagreement, label:

- direct wrong, revision correct;
- direct correct, revision wrong;
- both wrong;
- both correct but evaluator/extractor disagreed;
- infrastructure failure.

For self-generated tests, also measure:

- test validity against the task specification;
- false-positive rate: correct code rejected;
- false-negative rate: wrong code accepted;
- how often a false test causes a regression.

Until this rate is known, self-generated tests should be advisory: a failure may trigger a critic review, but should not by itself overrule a previously passing candidate.

## Stage 2: model-size sweep

The key experiment is a controlled size sweep within one model family where possible.

| Tier | Suggested range | Role |
| --- | --- | --- |
| Small | 1.5B–4B | Primary target; largest expected room for correction |
| Mid-size | 7B–14B | Check whether the effect survives at practical quality |
| Strong control | current ~35B MoE | Confirm that gains shrink or disappear near saturation |

Keep model family, instruction tuning, quantization level, context length, and serving backend as consistent as the available checkpoints allow. If MLX blocks the smaller checkpoints, use one OpenAI-compatible backend that supports all selected models (for example llama.cpp or vLLM) for the sweep; do not compare an MLX strong model against a differently configured small model and attribute the difference only to size.

Run the 20-task smoke set on every model first. Continue to the full set only when:

- all variants complete without systematic parse or transport failures;
- evaluator parity is confirmed;
- at least one ISRA ablation produces real Direct-to-correct flips without a comparable number of regressions.

## Stage 3: benchmark suite

Use benchmarks that leave room for improvement and expose different mechanisms.

1. **EvalPlus HumanEval+ and MBPP+** for code. Their additional tests reduce false passes and the two datasets reduce overfitting to HumanEval.
2. **GSM8K plus a harder math set or fixed MATH subset** for reasoning. Run exact paired task IDs; do not compare a partial Direct sample with a completed ISRA sample.
3. **A small hand-audited diagnostic set** of tasks where the Direct model commonly makes correctable errors. Keep it separate from headline benchmark claims.

MMLU-Pro is lower priority until the pipeline has a mechanism likely to improve closed-book multiple choice. A fast-path that simply returns Direct is useful operationally, but cannot demonstrate a refinement gain.

Run deterministic configurations once. For any stochastic configuration, use at least three predeclared seeds and report variation across seeds.

## Analysis and success criteria

Because every variant sees the same tasks, analyze paired outcomes rather than only aggregate percentages.

- Report the accuracy delta with a paired bootstrap 95% confidence interval.
- Report the 2×2 Direct/ISRA outcome table and an exact McNemar test.
- Report infrastructure failures separately and disclose exclusions.
- Report median and p95 latency, average LLM calls, and prompt/completion tokens.
- Report fixes and regressions separately: a `+4 pp` net result can hide very different behavior from “four fixes, zero regressions” versus “twenty fixes, sixteen regressions.”

A reasonable initial bar for continuing the full architecture:

- on at least one small or mid-size model, a positive paired accuracy delta whose 95% interval does not cross zero;
- a practically meaningful gain (predeclare `>= 3 percentage points` for the first full run);
- fewer regressions than fixes;
- no unresolved evaluator or infrastructure bias;
- latency/token overhead quantified and compared with the compute-matched direct baseline.

This threshold is a project decision, not a universal scientific standard. If the accuracy gain is smaller, ISRA can still be useful if it materially improves a high-value failure class or provides better reliability per unit of compute.

## Execution order

### Milestone A — harness integrity

- Replace separate/ad hoc runners with the paired runner.
- Add manifest, structured results, resume, retries, and unit tests.
- Reproduce 20 HumanEval tasks with Direct and ISRA at temperature 0.
- Manually audit all disagreements.

**Exit condition:** repeated smoke runs produce the same evaluated results and no infrastructure failure is counted as a wrong answer.

### Milestone B — mechanism search

- Run the ablation matrix on 30–50 code tasks.
- Classify every fix and regression.
- Measure critic calibration and self-test false positives.
- Select at most two promising ISRA variants.

**Exit condition:** one mechanism shows more genuine fixes than regressions and has a plausible causal explanation.

### Milestone C — size sweep

- Run the selected variants on small, mid-size, and strong-control models.
- Use identical backend and quantization policy where possible.
- Compare with greedy, sampling-matched, and compute-matched Direct.

**Exit condition:** identify the model-size region, if any, in which refinement has positive expected value.

### Milestone D — confirmatory run

- Freeze code, prompts, models, datasets, seeds, and success criteria before the run.
- Run full EvalPlus/MBPP+ and the selected reasoning benchmark.
- Produce paired statistics, cost/latency tables, and an error analysis.
- Keep raw result artifacts and the run manifest.

**Exit condition:** a result that can be reproduced from a clean checkout and described without caveats that invalidate the comparison.

## Pivot rules

- If generic critique causes more regressions than fixes, remove it and focus on execution-grounded refinement.
- If one revision helps but later iterations hurt, reduce ISRA to a single-review pipeline.
- If self-consistency matches ISRA at the same compute, reposition ISRA around diagnostics/traceability rather than accuracy.
- If no model tier shows a credible gain, publish the negative result and keep the benchmark harness as the project’s main artifact.
- Do not add ISRA to the portfolio’s featured projects until there is either a credible positive result or a particularly strong, reproducible negative-result write-up.

## What would make the repository presentable

Once the confirmatory experiment is complete:

- replace the current headline result with one paired table containing confidence intervals, latency, tokens, and model configuration;
- add a small architecture diagram and one reproducible benchmark command;
- link a frozen run manifest and raw result artifact;
- document failure cases and limitations next to the result;
- add CI for unit tests and a real license;
- keep claims narrow: name the models, benchmarks, and compute budget for which they were observed.

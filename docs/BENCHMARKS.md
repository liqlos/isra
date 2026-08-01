# Benchmark evidence and policy

## Status

This repository contains two materially different results:

| Mechanism | Controlled result | Decision |
|---|---|---|
| Legacy iterative self-refinement | 0 paired fixes in the controlled Llama 3.1 8B 20-task smoke; grounded repair cost +27.8% wall time and retry +35.8% | Retired. |
| SPAall logits intervention | 12/20 pass versus 10/20 direct on a separate preregistered 20-task development screen; 2 fixes, 0 regressions, 1.226× mean wall time | Proceed once to a disjoint confirmatory evaluation; no further tuning on this screen. |

The SPA result is not a claim that the method improves code generation in
general. The exact McNemar p-value is 0.5 and the task-paired 95% bootstrap
interval for the +10 percentage-point difference is [0, +25] percentage
points. Twenty tasks are a development gate, not adequate evidence of efficacy.

Historical HumanEval/GSM8K/MMLU numbers from the removed ISRA runtime were
exploratory, not consistently paired and sometimes mixed sampling settings.
They must not be used as baseline or marketing evidence. Their audit context is
preserved in [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) and Git history.

## SPAall development screen, 2026-08-01

### Frozen design

- Model: Meta-Llama-3.1-8B-Instruct-3bit, same checkpoint in both modes.
- Generation: greedy, one answer, maximum 800 completion tokens.
- Treatment: natural-language masking, Llama 3 finetune-right-pad token,
  omega = 1.28, enabled on every treatment task.
- Dataset: HumanEvalPlus v0.1.10, tasks 20--31 and 33--40.
- Oracle: EvalPlus 0.3.1 in the pinned local
  isra-evalplus:0.3.1 image.
- Isolation: evaluator tests did not enter the decoder; the benchmark
  repository mount was read-only.
- Provenance: task messages matched byte-for-byte per pair; each record reports
  exactly one LLM call. Mode order was deterministically randomized per task.

HumanEval/32 was excluded before model calls. EvalPlus 0.3.1's find_zero
special oracle executes continue before recording a successful input, so it
reports the canonical solution as fail (0/0). HumanEval/40 was the fixed
replacement.

### Result

| Metric | Direct | SPAall |
|---|---:|---:|
| Passed tasks | 10/20 | 12/20 |
| Direct-wrong to SPA-right | — | 2 |
| Direct-right to SPA-wrong | — | 0 |
| Mean wall time | 5,194.507 ms | 6,369.102 ms |
| Mean wall-time ratio | — | 1.226× |
| p95 wall time | 14,343.208 ms | 12,906.789 ms |
| Mean completion tokens | 270.90 | 212.35 |
| Mean reported decode throughput | 57.74 tok/s | 38.75 tok/s |
| Mean reported MLX peak allocation | 3.602 GB | 3.748 GB |

The p95 service-latency ratio is 0.900 because the modes completed different
numbers of tokens. The median per-task ratio is 1.534, while two short direct
answers make the p95 per-task ratio 2.896. Therefore the screen meets the
preregistered aggregate wall-time gate but does not establish a uniform
per-request latency bound.

No record had a model, formatting, transport or evaluator error. macOS pageouts
were unchanged during the run. The host already had substantial swap use and
other model services running; its small concurrent swap-counter change cannot be
attributed to SPA, so this shared-host run is not a clean memory-pressure
measurement.

The immutable run artifacts are intentionally Git-ignored:

~~~text
benchmark_runs/real-llama31-8b3bit-spa-dev-20-v2/
  manifest.json
  results.jsonl
  analysis-direct_greedy-vs-spa_greedy.json
  disagreements-direct_greedy-vs-spa_greedy.jsonl
~~~

## Required next evaluation

Freeze the implementation, model revision, prompt, mask and strength. Evaluate
at least 100 disjoint objective code-generation tasks, preferably a
date-filtered post-cutoff LiveCodeBench slice. Report pass@1, exact McNemar,
task-paired bootstrap confidence interval, error categories, output length,
mean and p95 wall time. Continue only if the difference is at least +2
percentage points and the confidence-interval lower bound is above zero.

HumanEval+ and MBPP+ remain useful regression suites, not the headline
benchmark.

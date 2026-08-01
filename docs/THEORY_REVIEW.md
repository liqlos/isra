# ISRA theory review and experiment gate

Status: theory-first design freeze completed; Stage 1 real-model smoke found no
quality gain from the tested inference-only repair mechanism.

This document answers a narrower question than "does a larger prompt help?":

> Can an inference-time pipeline improve the functional correctness of a local
> 7--9B model enough to justify its added latency and generated tokens?

The objective is a correctness/cost Pareto improvement. A more complicated
pipeline is not a success merely because it sometimes repairs an answer. It
must beat a strong, compute-matched retry baseline, avoid changing correct
answers into wrong ones, and report the full latency/token distribution.

## Executive conclusion

The current full ISRA pipeline is not theoretically justified as a default
code-correction mechanism. Most phases ask the same model to reinterpret its
own output without adding an independent correctness signal. Different role
prompts and temperatures change sampling behavior, but do not make the critic
independent or more knowledgeable.

The most defensible inference-only design is much smaller:

```text
one solve -> cheap trusted checks -> one targeted repair on proven failure
          -> repeat the same checks -> accept only evidence-backed improvement
```

For code, "trusted checks" means syntax/compiler results and tests or examples
provided by the task/user. A test invented by the same model is not a trusted
oracle. When no trusted failure exists, the pipeline should preserve the first
candidate. An independent retry remains an obligatory cost-matched control.

This is a hypothesis, not a promised improvement. Small models are often worse
at locating their own errors than at fixing a known, localized error.

The first 20-task HumanEval+ smoke on a 3-bit Llama 3.1 8B checkpoint supported
that skeptical prior: neither trusted-feedback repair nor an unguided retry
fixed a hidden-test failure. Both preserved the initial pass/fail outcomes, but
added calls, latency and tokens. This descriptive result is not a benchmark
claim; it is a gate against spending a larger evaluation budget on the current
variant.

## Why iteration can and cannot help

Let the initial model produce candidate `A`. A second model call can help in
only two materially different ways:

1. **Exploration:** sampling produces a different candidate. This is retry or
   test-time scaling, not evidence that the model understood its first error.
2. **Information:** the second call receives a signal correlated with actual
   correctness, such as a compiler error or a trusted failing example.

A same-model critique with no new evidence is mostly a correlated
re-interpretation of `A`. It can occasionally redirect attention, but it can
also persuade the model to abandon a correct answer. Therefore every claimed
"self-repair" gain must be separated from the gain of simply spending the same
tokens on another sample.

The verifier/selector is the bottleneck. Producing two candidates improves the
chance that one is correct, but does not improve returned pass@1 unless the
system can select the better candidate above chance.

## Evidence from primary research

| Finding | Evidence | Consequence for ISRA |
|---|---|---|
| Prompted intrinsic self-correction is not reliable in general. | The TACL critical survey found no general reliable evidence for prompted self-correction, except unusually suitable tasks; reliable external feedback and training were the recurring successful conditions. [Kamoi et al., TACL 2024](https://aclanthology.org/2024.tacl-1.78/) The ICLR study found that intrinsic correction can degrade reasoning. [Huang et al., ICLR 2024](https://openreview.net/forum?id=IkmD3fKBPQ) | Generic same-model critique is diagnostic, not a correctness oracle. |
| Code repair works best when the failure is real and localized. | Unit-test execution feedback improved MBPP results across Codex, GPT-3.5, GPT-4 and StarCoder; typically one debugging turn captured nearly all of the gain. [Chen et al., ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/2460396f2d0d421885997dd1612ac56b-Abstract-Conference.html) Self-Edit similarly uses execution of examples supplied with the problem, not invented expected outputs. [Zhang et al., ACL 2023](https://aclanthology.org/2023.acl-long.45/) | Test `one trusted failure -> one repair` first. Pass the exact failing input, expected/actual output or exception. |
| Self-generated tests are biased and can reduce correctness. | On Qwen2.5-Coder-7B-Instruct, post-execution self-debugging reduced HumanEval from 81.7 to 78.0 with label feedback and 76.2 with detailed feedback after one turn. Only 44.5% of its ten-test HumanEval suites were entirely correct. [Chen et al., ACL 2025](https://aclanthology.org/2025.acl-long.881/) | A generated test must not reject `A` or force a rewrite. Keep generated-test mechanisms out of the primary variant. |
| Runtime observations are safer than invented expected outputs, but gains on 7B were small. | In the same ACL 2025 study, in-execution traces moved Qwen2.5-Coder-7B from 81.7 to 82.3 on HumanEval, 70.6 to 72.0 on MBPP, and 35.8 to 36.4 on LiveCodeBench after two turns. | Runtime-trace refinement is a later ablation, not the low-overhead default. |
| Repair gains can disappear after accounting for inference budget. | Self-repair gains were modest, subset-dependent, and sometimes absent when compared with token-aware sampling; stronger or human feedback helped much more. [Olausson et al., ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/9ddc141bdbf9d1db510cefff56c586ad-Abstract-Conference.html) | `unguided_retry` and compute-matched sampling are mandatory controls. |
| Small models are limited by weak self-verification. | Models at or below 13B improved with a strong verifier but struggled with a weak self-verifier. [Zhang et al., Findings of ACL 2024](https://aclanthology.org/2024.findings-acl.924/) Prompt-only refinement also degraded a small-model baseline in the CoCoS study; large gains required training correction behavior. [Cho et al., Findings of EMNLP 2025](https://aclanthology.org/2025.findings-emnlp.127/) | A low-temperature copy of the same 8B model is not an independent verifier. If inference-only methods fail, training a refiner/verifier is more defensible than adding prompt phases. |
| Generated tests can help selection only when combined with multiple candidates and agreement. | CodeT ranks many programs with dual execution agreement; it does not treat a single model-generated test suite as unquestioned truth. [Chen et al., ICLR 2023](https://arxiv.org/abs/2207.10397) | A low-budget dual-execution variant may be tested later, but its candidate count and cost must be explicit. |
| Learned correction is different from a prompted scaffold. | SCoRe uses multi-turn reinforcement learning; CYCLE trains code models on execution feedback. [SCoRe, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/871ac99fdc5282d0301934d23945ebaa-Abstract-Conference.html), [CYCLE](https://arxiv.org/abs/2403.18746) | Results from trained self-correctors do not validate the current prompt-only pipeline. |
| Larger hidden-test suites expose false positives in code benchmarks. | EvalPlus expands HumanEval with far more tests and changes model rankings. [Liu et al., NeurIPS 2023](https://arxiv.org/abs/2305.01210) | HumanEval must be evaluated through pinned HumanEval+, not the original small test set. |

### Chinese research check

Chinese-language searches did not reveal stronger primary evidence than the
peer-reviewed English publications from Chinese laboratories. The most directly
relevant result is the ACL 2025 study led by researchers associated with the
Peking University high-confidence software group: it evaluates
Qwen2.5-Coder-7B on HumanEval, MBPP and LiveCodeBench and directly demonstrates
the danger of model-generated expected outputs. Self-Edit and CodeT are also
from Chinese research teams and support execution-grounded editing or
agreement-based selection rather than an unconstrained same-model critic.

Chinese summaries and paper-aggregation pages were not used as evidence.

## Audit of the current implementation

The following are correctness or attribution problems, not style issues.

| Current mechanism | Finding | Priority |
|---|---|---|
| Phase 0 agreement | Two correlated answers can agree on the same error. In benchmark override mode, the requested temperature override also suppresses the intended `0.5` second-call temperature. Agreement is an uncertainty feature, not proof. | P1 |
| Phase 1 | This is the main solve call and may be useful on its own. Because Direct and ISRA use different solve prompts, the current Direct-vs-ISRA comparison cannot separate prompt engineering from iterative processing. | P0 |
| Syntax checks | Syntax failure is detected, but the CODE path explicitly does not force a repair. The memory guard can therefore return syntactically invalid code. | P0 |
| `execute_code_safely` | Despite its name, it compiles code but does not exercise function behavior. It adds no semantic signal beyond syntax. | P0 for interpretation |
| User-supplied doctest examples | These are the strongest current evidence, although the custom parser implements only a subset of doctest semantics. | Keep, then harden |
| Self-generated tests | They are executed as authoritative tests. A false expected output can force a second generation, and the returned error often omits the test input and expected/actual values needed for a targeted repair. | P0 |
| Same-model critic | For CODE, critic feedback usually cannot trigger a new iteration: the one-iteration memory guard converts `LOOP_*` to `TERMINATE_SUCCESS`. The system pays for critique without applying it. | P0 |
| Confidence and STOP | Self-reported confidence is uncalibrated yet controls stopping and fallback selection. Some STOP paths do not require the configured confidence threshold. | P0 |
| Phase 3 compression | It is a lossy model call with no new evidence. The Phase 3 input omits exact Phase 1 constraints/code while its prompt asks for them. Previously accumulated "confirmed" claims are additive and cannot be retracted. | P0 |
| Phase 4 synthesis | It creates another chance to damage an objectively checkable answer. Code already bypasses it; objective variants should return the selected candidate verbatim. | P1 |
| Candidate selection | `best_code` is overwritten by the latest candidate before evidence-based comparison. Fallback relies partly on critic confidence, not monotonic trusted evidence. | P0 |
| Session state | Model-produced claims can cross turns under a `VERIFIED` label. Benchmark sessions are isolated, but normal auto-session behavior can contaminate independent requests. | P1 |
| Benchmark telemetry | Full A/B candidates, generated tests, critic judgments, and iteration logs are not all exposed in HTTP metadata. This prevents causal failure analysis. | P0 before ablations |

The most important implementation fact is that the current CODE path is not the
advertised `solve -> critic -> revision` loop. It is usually `solve -> critic ->
return the original/latest code`; only a doctest or self-generated-test failure
normally permits a second solve.

## Frozen baseline and new variants

The existing prompts and legacy behavior remain frozen under a clearly named
baseline. Improvements are separate variants so that a treatment cannot rewrite
its own baseline after seeing results.

### Required controls

1. `direct`: canonical one-call task prompt.
2. `phase1_only`: the exact ISRA Phase 1 solve prompt, with no critic or later
   phases. This isolates prompt engineering.
3. `phase1_unguided_retry`: a second independent solve with no feedback. The
   retry must use a different deterministic sub-seed.
4. `legacy_isra`: the frozen current pipeline.

There are two distinct experiments:

- **end-to-end system comparison:** each system receives the same task and is
  judged as a deployable black box;
- **mechanism attribution:** generate candidate `A` once per `(task, seed)`,
  store its exact bytes, then fan that same `A` out to retry, critic and grounded
  repair branches. This is the only clean way to count which treatments repair
  or damage the same starting answer. Repeating a nominally identical first
  call is weaker because even temperature zero can be nondeterministic.

Legacy ISRA remains in the end-to-end comparison until it supports injection of
a frozen initial candidate. It must not be treated as a clean mechanism
ablation before then.

### Primary treatment: `grounded_one_repair`

```text
Phase 1 solve -> immutable candidate A
    |
    +-- compile/syntax failure -------------------------------+
    |                                                         |
    +-- trusted public example/test failure ------------------+--> one repair
    |                                                              |
    +-- checks pass or no trusted semantic test -> return A         v
                                                            candidate B
                                                                 |
                                         repeat identical trusted checks
                                            |                  |
                                  all trusted failures clear  otherwise
                                            |                  |
                                          return B           rollback A
```

Rules:

- maximum one repair call;
- no Phase 2, Phase 3 or Phase 4 on this code path;
- `A` and `B` are immutable and both are logged in full;
- only task/user-provided tests can trigger semantic repair;
- generated tests may be logged in a later advisory variant but cannot reject
  a candidate;
- accept `B` only if it clears every available trusted failure on the identical
  suite; reducing but not clearing the failure set is still a known-wrong
  candidate and must be rolled back;
- if both candidates fail, return `A` and record that neither passed; never
  label a partial improvement verified;
- session state is disabled;
- exact verification evidence is included in the repair prompt and result log.

### Diagnostic variants, not deployment defaults

- `critic_label_only`: measure critic precision/recall without changing code;
- `critic_one_rewrite`: test whether intrinsic critique adds value beyond retry;
- `legacy_isra_no_self_tests`;
- `legacy_isra_no_phase3`;
- `generated_tests_advisory`;
- `in_execution_trace_one_repair`;
- low-budget dual execution agreement over 2--3 candidates.

## Temperature is a factor, not a role credential

There is no strong primary evidence that "creative high-temperature proposer +
low-temperature critic" makes a small model a better self-correcting system.
Lower temperature makes the critic more repeatable, not more correct. Higher
temperature increases exploration, but also increases the chance of breaking a
nearly correct program.

Initial clean comparisons should therefore isolate temperature:

- greedy family: proposer/repair at `T=0` (or the lowest backend-supported
  deterministic setting);
- sampled family: proposer at `T=0.6`, with a separately declared repair
  temperature;
- legacy configured family: preserve the current per-phase temperatures as the
  treatment being evaluated;
- use different derived seeds for proposer, retry, critic and repair;
- do not force every internal phase to one temperature and still call that the
  configured legacy pipeline.

Promising but unproven operating values are `T=0--0.2` for an initial pass,
`T=0.6--0.8` for an independent exploratory retry, and `T=0--0.3` for a minimal
repair grounded in an exact failure. These are experiment factors, not fixed
facts.

## Falsifiable hypotheses

These hypotheses must be registered before inspecting real benchmark outcomes.

### H1: trusted feedback has incremental value

On initially failing candidates for which a trusted public failure exists,
`grounded_one_repair` has a higher `wrong -> correct` rate than
`phase1_unguided_retry` at a matched call/token budget.

Falsified if its paired net fixes are not higher, or if the gain is explained by
more generated tokens.

### H2: evidence gating protects correct answers

`grounded_one_repair` has a lower `correct -> wrong` regression rate than
`critic_one_rewrite` and legacy ISRA.

Falsified if its rollback/selection rule changes more initially correct answers
to wrong ones.

### H3: generic critique is not enough on 7--9B

`critic_one_rewrite` does not outperform compute-matched unguided retry, and
critic labels are not sufficiently discriminative between correct and wrong
initial candidates.

This is intentionally a skeptical null-style hypothesis. Reject it only with
paired evidence and a critic confusion matrix.

### H4: self-generated tests are unsafe as an oracle

Authoritative generated-test feedback causes more regressions than advisory-only
use, particularly on initially correct candidates.

The primary experiment does not need to expose the system to this known risk;
run it only as a bounded diagnostic ablation.

### H5: adaptive repair is preferable to unconditional multi-call ISRA

The primary treatment lies on a better correctness/latency Pareto frontier than
legacy ISRA: equal or higher correctness with fewer median/p95 generated tokens,
calls and wall-clock seconds.

No opaque weighted "quality score" will be used. Report both axes.

## Evaluation stages

### Stage 0: evaluator and harness integrity only

- 20 pinned HumanEval+ tasks;
- canonical solutions must pass 100% before any model request is evaluated;
- at least one known-bad mutation per selected task must fail;
- evaluator exceptions/timeouts are infrastructure/evaluator errors, never model
  failures;
- fake endpoint run validates order, resume, retry and result schema, not model
  quality.

The existing Python 3.14 fake-endpoint smoke is invalid: EvalPlus child
processes failed while applying `RLIMIT_AS`, and the harness recorded all 80
records as ordinary failures. It must be preserved as an evaluator incident,
not interpreted as an ISRA result.

### Stage 1: mechanism smoke

- 20--40 HumanEval+/MBPP+ tasks, large enough to exercise all branches but not
  for a headline significance claim;
- `direct`, `phase1_only`, `phase1_unguided_retry`,
  `grounded_one_repair`, and frozen `legacy_isra`;
- manually inspect every pairwise disagreement and every repair trigger;
- freeze prompts, revisions and evaluator after this stage.

#### Observed Stage 1 gate result (2026-08-01)

The first 20 ordered HumanEval+ tasks were evaluated descriptively on
`Meta-Llama-3.1-8B-Instruct-3bit` with proposer and repair at temperature zero.
The exact Phase 1 candidate was fanned out to both branches.

| Variant | Pass | Fail | Model error / skipped | Extra model calls | Paired fixes | Paired regressions |
|---|---:|---:|---:|---:|---:|---:|
| `phase1_only` | 12 | 7 | 1 / 0 | 0 | — | — |
| `grounded_one_repair` | 12 | 7 | 0 / 1 | 7 | 0 | 0 |
| `phase1_unguided_retry` | 12 | 7 | 0 / 1 | 7 | 0 | 0 |

For the 19 evaluable pairs, each branch had `both pass=12`, `both fail=7`,
`treatment only=0`, and `control only=0`. Grounded repair added `59.1 s`
across the 20 tasks (`+27.8%` over parent wall time); unguided retry added
`76.2 s` (`+35.8%`). On the 19 parent calls whose usage was retained by the
pre-fix runner, grounded repair added 7,228 tokens (`+33.3%`) and retry added
8,389 (`+38.7%`). One empty-output model error lost its usage payload in that
runner revision, so those token percentages are not exact all-20 totals. The
harness now preserves cost metadata for empty successful responses.

Seven candidates triggered each second-call branch. Six were rolled back.
HumanEval/5 initially allowed a partially improved candidate that still failed
one public example; hidden tests also failed. The acceptance gate was therefore
tightened to require the whole available trusted suite to pass, and a final
real-model diagnostic confirmed rollback. There were no observed repairs and
no evidence for proceeding directly to a confirmatory LiveCodeBench run with
this mechanism.

### Stage 2: primary code experiment

- a pinned LiveCodeBench `code_generation_lite` release, restricted to tasks
  after the primary model's documented training cutoff, is the leading primary
  suite;
- use the normal Code Generation task with only the statement and public
  examples. LiveCodeBench's official Self-Repair mode reveals a failing private
  test and expected answer; that is an oracle-assisted upper bound, not the
  deployable ISRA treatment;
- HumanEval+ and MBPP+ remain regression/evaluator suites, not the headline
  proof. They are static and widely exposed in training corpora;
- a practical-library suite such as BigCodeBench is a later robustness check
  because its dependencies add substantial evaluator complexity;
- repository-level SWE-bench is out of scope for the current function-generation
  architecture and would conflate tools, retrieval and patch application.

If no sufficiently large post-cutoff LiveCodeBench window exists for a model,
that model cannot support a contamination-resistant headline claim. Use an
older checkpoint with a documented cutoff or label the result a regression
study. Do not silently call the public static set contamination-free.

Development and confirmatory task sets are disjoint. Temperatures, repair
limits, prompts and token caps may be selected only on the development set.

Final suite/version and sample-size rules are frozen in `EXPERIMENT_PLAN.md` and
the run manifest before execution.

### Stage 3: separate math experiment

Do not combine code and math into one score. GSM8K is a cheap smoke/regression
suite but is too old, public and easy to be the primary second benchmark.
MATH-500 is harder but also widely used in post-training. A pinned post-cutoff
[LiveBench Mathematics](https://arxiv.org/abs/2406.19314) release is the leading
confirmatory option; if an 8B model is at floor, use a predeclared easier stratum
or [GSM1k](https://arxiv.org/abs/2405.00332) and label the limitation.

For math, first compare direct, independent retry and self-consistency.
Calculator/execution feedback is trusted only for arithmetic it actually
checks; it does not prove that the model translated the word problem correctly.

## Metrics and decision rules

For every treatment/control pair report:

- exact paired `both pass / control only / treatment only / both fail` table;
- fixes (`wrong -> correct`) and regressions (`correct -> wrong`) separately;
- paired bootstrap confidence interval for the accuracy delta;
- exact McNemar test on disagreements;
- completed/model/transport/timeout/evaluator errors separately;
- median and p95 latency, prompt/completion tokens and LLM calls;
- conditional repair rate and repair success rate;
- for a critic: precision, recall and false-positive rate on initial candidate
  correctness;
- for generated tests: suite accuracy, false-rejection and false-acceptance rates.

For sampled conditions use 3--5 predeclared paired seeds. Resample tasks, not
individual task/seed rows, in the bootstrap. Greedy conditions also need a
small repeated-run reproducibility check before a single output is treated as
deterministic.

Smoke results are descriptive. A primary positive claim requires a frozen,
adequately sized experiment and a positive paired effect that is not erased by
the compute-matched retry control. Practical adoption additionally requires the
treatment to be on the empirical correctness/cost Pareto frontier.

## Local model and backend choice

Read-only inventory of the 24 GB Mac found:

- `Meta-Llama-3.1-8B-Instruct-3bit`: complete local MLX checkpoint based on the
  official instruction model. This is the cleanest already-installed weak-model
  control, although 3-bit quantization must be recorded as a limitation. Its
  older documented cutoff also makes a post-cutoff LiveCodeBench slice possible.
- `Qwen3.5-9B-abliterated-mixed-3_4-MTP`: complete custom 9B checkpoint, but it
  is abliterated and uses mixed 3/4-bit quantization. It is useful as a secondary
  model-family robustness check, not a clean headline control.
- `qwen3.5-9b-mtp-4bit` under `ai_models` is only a small MTP draft model, not a
  standalone 9B generator.
- the currently served 27B persona checkpoint and its public proxy are
  confounded by persona fine-tuning; the proxy also rewrites the system prompt
  and caps output length. They are unsuitable for the primary experiment.

Use one exact model, quantization, backend version and raw OpenAI-compatible
endpoint for every paired comparison. Do not download another large checkpoint
without approval. No remote model is started or switched during this theory
review.

## Gate before real evaluation

Real model evaluation remains blocked until all of the following are true:

1. canonical-solution evaluator sanity checks pass in a supported Python
   environment;
2. the invalid Python 3.14 mock incident is recorded in `EXPERIMENT_LOG.md`;
3. the frozen baseline and separate variants above are implemented and unit
   tested;
4. exact candidate/evidence telemetry, frozen-candidate fan-out and rollback
   are present;
5. primary benchmark revision, model checkpoint and derived seed policy are
   frozen in the manifest;
6. the raw local model endpoint is used, with no persona proxy or hidden prompt
   rewrite.

Only after this gate should Stage 0 and Stage 1 be rerun.

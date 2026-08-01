# Experiment log

Append-only record of benchmark work. A run appears here even when it is
invalid; invalid runs are never reused as model-quality evidence.

## 2026-08-01 — theory-first audit and harness redesign

### Status

No real model evaluation was run. Evaluation was deliberately paused until the
architecture review, frozen hypotheses, evaluator gates and mechanism controls
were in place.

The scientific and implementation review is recorded in
[`THEORY_REVIEW.md`](THEORY_REVIEW.md). Its main decision is to keep legacy ISRA
frozen and add a separate `grounded_one_repair` treatment:

```text
one solve -> compiler/public examples -> one repair on proven failure
          -> identical checks -> evidence-based accept or rollback
```

### Research conclusions applied

- Same-model critique without new evidence is not treated as verification.
- Role temperatures are independent factors; low temperature does not make the
  same weights an independent critic.
- Self-generated expected outputs cannot reject a candidate or force repair.
- One repair using a real compiler/public-test failure is the primary treatment.
- Unguided retry with the same trigger and compute is the mandatory control.
- A single initial candidate must be fanned out byte-for-byte for mechanism
  attribution.
- HumanEval+/MBPP+ are smoke/regression suites. A pinned post-cutoff
  LiveCodeBench Code Generation slice is the leading primary suite.

### Implementation changes

- Added `phase1_only`, `phase1_unguided_retry`, and
  `grounded_one_repair` as separate HTTP-selectable variants. The default
  remains `legacy_isra`; its prompt constants were not changed.
- Added stable, role-derived seeds and per-call role/seed telemetry for the new
  variants.
- Added trusted checks using compile and only user/task-provided `>>>` examples.
- Added immutable `candidate A/B` traces, exact failure evidence, monotonic
  acceptance and rollback.
- Added `benchmarks/mechanism_benchmark.py`. It generates `A` once, injects its
  exact response into both branches, verifies endpoint parity, and charges the
  parent solve latency/tokens/calls to every branch.
- Added evaluator sanity gates. Before any model request, canonical solutions
  must pass and intentional failing mutations must be rejected.
- Full local EvalPlus execution on macOS is now refused. EvalPlus 0.3.1's
  `RLIMIT_AS` guard is incompatible with this Darwin environment, and its own
  documentation recommends Docker for untrusted model code. Local Darwin is
  limited to trusted `--evaluator-sanity-only` checks with the official
  `EVALPLUS_MAX_MEMORY_BYTES=-1` switch.

### Invalid mock run: `mock-smoke-20-v1`

Location: `benchmark_runs/mock-smoke-20-v1/`

Configuration:

- HumanEval+ `v0.1.10`, tasks `HumanEval/0` through `HumanEval/19`;
- EvalPlus `0.3.1`;
- four legacy end-to-end variants, 80 records total;
- fake local OpenAI-compatible endpoint;
- runner Python 3.14 on macOS.

Observed result:

- all 80 records were written with `passed=false`;
- evaluator details for both canonical-looking and intentionally wrong fake
  responses were `base=timeout` and `plus=timeout` with zero executed details;
- EvalPlus child processes crashed in `resource.setrlimit(RLIMIT_AS, ...)` with
  `ValueError: current limit exceeds maximum limit`.

Classification: **INVALID EVALUATOR RUN**. It says nothing about Direct or
ISRA. The result directory is preserved locally for audit and ignored by git.

The same failure was independently reproduced under Python 3.11, proving that
the cause was not Python 3.14. The newly added sanity gate stopped before model
preflight and reported 20 canonical failures instead of silently producing
model FAIL records.

### Evaluator sanity after Darwin-specific safe handling

Command:

```bash
.venv311/bin/python benchmarks/paired_benchmark.py \
  --direct-url http://127.0.0.1:1/v1/chat/completions \
  --isra-url http://127.0.0.1:1/v1/chat/completions \
  --model evaluator-sanity-only \
  --task-count 20 \
  --evaluator-sanity-only
```

Exact result:

```json
{
  "canonical_solutions_passed": 20,
  "evalplus_max_memory_bytes": "-1",
  "known_bad_mutations_rejected": 20,
  "status": "passed",
  "tasks_checked": 20
}
```

Elapsed time was `25127.037 ms`. Only trusted dataset canonical solutions and
intentional `raise AssertionError` mutations were executed; no model code was
run in this local mode.

### Unit tests

Command:

```bash
.venv311/bin/python -m pytest -q
```

Result:

```text
26 passed in 0.99s
```

Covered behavior includes extraction, evaluator parity, error categorization,
immutable manifests, retry/resume, trusted-check semantics, rollback, distinct
role seeds, no-call early return, exact frozen-candidate fan-out, and inclusive
parent cost accounting.

### Read-only local-model inventory

The 24 GB Mac at `ai-server@192.168.1.101` was inspected over existing SSH key
authentication. No password was stored and no remote service/model was changed.

Primary already-installed candidate:

- `/Users/ai-server/ai_models/Meta-Llama-3.1-8B-Instruct-3bit`
- base model: `meta-llama/Llama-3.1-8B-Instruct`
- MLX affine quantization: 3-bit, group size 64
- local weights: approximately 3.5 GB

Secondary, confounded candidate:

- `/Users/ai-server/models/Qwen3.5-9B-abliterated-mixed-3_4-MTP`
- custom abliterated checkpoint with per-layer mixed 3/4-bit quantization
- useful only for model-family robustness, not the clean headline result

The currently served 27B persona model and port-8081 proxy are not suitable:
the checkpoint is persona-tuned, and the proxy replaces the system prompt and
caps completions at 25 tokens. The raw MLX server can later be reached through
an SSH tunnel without modifying the remote machine.

### Remaining gate before any model smoke

1. Run the full harness inside a pinned official EvalPlus Docker image and
   record its image digest.
2. Test the HTTP variant path end-to-end against the fake endpoint inside that
   container; canonical gates must pass first.
3. Start a raw MLX endpoint for the selected 8B checkpoint without persona
   rewriting, record the exact checkpoint hash and backend version, and expose
   it only through SSH tunneling.
4. Run a 20-task descriptive mechanism smoke with the frozen candidate fan-out.
5. Manually classify all fixes, regressions, repair triggers and rollbacks.
6. Only then freeze a disjoint, post-cutoff LiveCodeBench confirmatory run.

Next recommended experiment: Docker-contained 20-task HumanEval+ mechanism
smoke on `Meta-Llama-3.1-8B-Instruct-3bit`, comparing `phase1_only`, conditional
unguided retry, and `grounded_one_repair`. Legacy ISRA should be reported as a
separate end-to-end system result, not mixed into the frozen-candidate mechanism
claim.

## 2026-08-01 — Docker integrity gate and real 8B mechanism smoke

### Pinned evaluator environment

The nominal official image `ganler/evalplus:v0.3.1` was not accepted on its tag
alone. Its only published `linux/amd64` manifest resolved to base digest
`sha256:26b118098bef281fe8dfe999bf05f1d5b45374b4e6c00161ec0f30592aef4740`,
but inspection found `evalplus 0.4.0.dev2` inside it rather than `0.3.1`.

`benchmarks/Dockerfile.evalplus` now derives from that exact digest and replaces
only EvalPlus, without dependency upgrades, using the pinned `0.3.1` wheel hash
`sha256:cd601debb67419113d10ac5c3317689d847f27de5d8cf3837975f3cab571b75d`.
The resulting local evaluator image is:

```text
isra-evalplus:0.3.1
sha256:ad611279c9e1a0cbe9466dcde3f862f263683461465d260f4d4f6d5fdd684e4b
linux/amd64, Python 3.11.10, EvalPlus 0.3.1
numpy 2.1.2, datasets 3.0.1, transformers 4.45.2, aiohttp 3.10.10
```

The 20-task in-container gate passed in `4933.216 ms`:

```json
{
  "canonical_solutions_passed": 20,
  "known_bad_mutations_rejected": 20,
  "platform": "Linux-6.12.76-linuxkit-x86_64-with-glibc2.36",
  "status": "passed",
  "tasks_checked": 20
}
```

Adding `benchmarks/__init__.py` was necessary because the base image contains an
unrelated installed `benchmarks` package that otherwise shadows the repository
directory.

### Frozen-candidate mock gate

Run: `benchmark_runs/mock-mechanism-docker-20-v1/`

- 20 tasks, 3 dependent variants, 60 finalized records;
- evaluator canonical/mutation gate passed 20/20 and 20/20;
- endpoint parity rejected any branch not echoing the exact candidate A;
- deterministic branch order and inclusive parent cost were recorded;
- synthetic expected pass counts were reproduced exactly: parent 18/20,
  grounded 19/20, retry 19/20.

Classification: **VALID HARNESS INTEGRITY RUN, NOT MODEL EVIDENCE**.

### Real model and backend provenance

The clean weak-model smoke used a separate raw endpoint for:

```text
/Users/ai-server/ai_models/Meta-Llama-3.1-8B-Instruct-3bit
base: meta-llama/Llama-3.1-8B-Instruct
quantization: MLX affine 3-bit, group size 64
model.safetensors sha256: 0455539b7dc0d33dcd07071a719a747db7be9657a03ec6f262060e541db4c342
config.json sha256: 9e639b4f34401a6401e9d2051c0dde32fe1b61da6df449da5115276107ed97e9
tokenizer.json sha256: 6b9e4e7fb171f92fd137b777cc2714bf87d11576700a1dcd7a399e7bbe39537b
mlx-lm 0.31.3, mlx 0.32.0
machine: mac2-llm, Mac16,8, 24 GB
```

It listened only on remote `127.0.0.1:8085` and was reached through an SSH
tunnel. Existing ports 8081--8084 and the 27B service were not changed. The
temporary 8B server, tunnel and local evaluation orchestrator were stopped
after the run; the pre-existing services remained listening.

### Real HumanEval+ Stage 1 result

Finalized shards:

- `benchmark_runs/real-llama31-8b3bit-mechanism-smoke-5-v1/`: tasks 0--4;
- `benchmark_runs/real-llama31-8b3bit-mechanism-smoke-15-v2/`: tasks 5--19.

An earlier `...smoke-15-v1` shard is intentionally incomplete. HumanEval/10
produced an HTTP-success response with no executable model text, and the runner
incorrectly aborted dependent fan-out. This is preserved as a harness incident,
not merged into the result. The fix records the parent `model_error`, emits two
`skipped_parent_unavailable` dependent records, and continues the batch.

Combined descriptive outcome:

| Variant | Pass | Eval fail | Model error | Skipped | Total branch calls | Incremental calls |
|---|---:|---:|---:|---:|---:|---:|
| `phase1_only` | 12 | 7 | 1 | 0 | 19 recorded / 20 actual | — |
| `grounded_one_repair` | 12 | 7 | 0 | 1 | 26 recorded / 27 actual | 7 |
| `phase1_unguided_retry` | 12 | 7 | 0 | 1 | 26 recorded / 27 actual | 7 |

Paired over the 19 evaluable tasks, both treatments had `both pass=12`,
`both fail=7`, `treatment only=0`, `parent only=0`. Thus:

- grounded fixes: 0; grounded regressions: 0;
- retry fixes: 0; retry regressions: 0;
- grounded incremental latency: `59,070.436 ms` (`+27.8%`);
- retry incremental latency: `76,191.355 ms` (`+35.8%`);
- grounded incremental tokens: 7,228 (`+33.3%` of the 19 parent calls with
  retained usage);
- retry incremental tokens: 8,389 (`+38.7%` on the same observable base).

The pre-fix endpoint caller discarded the payload for the one empty-output
model error, so its call/tokens were undercounted in the immutable run records.
It did preserve the 35.7-second latency. The caller now retains payload, usage,
finish reason and ISRA phase metadata for such model errors; a regression test
covers this. The historical run is not rewritten.

Both repair branches reduced HumanEval/5 from two public failures to one and
the old monotonic rule selected B, but B was still known wrong and failed
EvalPlus. The acceptance rule now requires all available trusted failures to be
cleared. Finalized diagnostic
`benchmark_runs/real-llama31-8b3bit-partial-gate-task5-v3/` confirmed that both
partially improved candidates are rolled back to A with reason
`trusted_failures_reduced_but_not_cleared`.

### Decision

The current grounded one-repair prompt is **not on a better quality/cost Pareto
frontier** than its parent or compute-matched retry in this smoke. It preserved
quality through conditional no-call and rollback, and was cheaper than retry,
but yielded zero fixes for seven repair triggers. No larger HumanEval run and
no confirmatory LiveCodeBench run is justified for this exact variant.

The next defensible development work is one of:

1. compare the ordinary Direct solve prompt with `phase1_only` on a disjoint
   development slice, because prompt quality is still confounded with iteration;
2. use a genuinely stronger external verifier or user-provided tests;
3. train correction behavior on execution feedback rather than adding more
   same-model prompt phases;
4. test a small multi-candidate execution-agreement selector with its full cost
   exposed.

### Final local verification

```text
28 passed in 1.04s
compileall: passed
git diff --check: passed
```

## 2026-08-01 — SPAall replacement prototype and development screen

### Design decision

The literature and independent research review rejected another same-model
critique loop. The replacement mechanism is Selective Prompt Anchoring
(SPAall): at every generated token, run the same model over the visible prompt
and a position-preserving masked-prompt copy, then calculate:

~~~text
L_spa = omega * L + (1 - omega) * L_mask
~~~

The first implementation used the paper's cross-model omega = 1.28 preset, a
natural-language anchor, and the Llama 3 finetune-right-pad token (id 128004).
It exposes one OpenAI-compatible answer, has no critic/retry/selector and never
receives evaluator tests. The design, sources and pre-registered gates are in
[REPLACEMENT_DESIGN.md](REPLACEMENT_DESIGN.md).

### Integrity gates

- Unit tests cover prompt masking, token-position preservation, logits
  arithmetic, response metadata, preflight and paired-run provenance.
- A first five-task identity run is retained as invalid transport evidence:
  MLX stream state was mistakenly handed to a worker thread, causing ten
  transport errors. No quality conclusion was drawn.
- After keeping MLX generation on the model-loading thread, direct and
  SPA(omega=1) emitted identical completion token-ID hashes, finish reasons and
  evaluator outcomes for HumanEval+ tasks 20--24. Aggregate wall time was
  1.528× and MLX peak allocation increased from 3.674 GB to 3.877 GB.
- The pinned EvalPlus 0.3.1 evaluator revealed a dataset-specific defect:
  HumanEval/32's find_zero oracle runs continue before recording successful
  details, therefore falsely rejects even the canonical solution as fail (0/0).
  This was reproduced directly against the evaluator source. The task was
  excluded before inference and HumanEval/40 substituted.
- The exact 20-task replacement set passed 20/20 canonical-solution and
  20/20 intentional-mutation sanity checks in the pinned container.

### Real paired result

Finalized run:

~~~text
benchmark_runs/real-llama31-8b3bit-spa-dev-20-v2/
~~~

Configuration:

- Meta-Llama-3.1-8B-Instruct-3bit; same model and visible messages in both
  modes; greedy decoding; maximum 800 tokens.
- HumanEvalPlus v0.1.10 tasks 20--31 and 33--40; each pair used one stable
  seed and deterministically randomized mode order.
- EvalPlus 0.3.1 in the pinned local image
  isra-evalplus:0.3.1@sha256:ad611279c9e1a0cbe9466dcde3f862f263683461465d260f4d4f6d5fdd684e4b.
- The evaluator container saw a read-only repository plus a separate writable
  result mount. Existing remote services were not changed.

Observed outcome:

| Outcome | Count |
|---|---:|
| Direct pass / SPA pass | 10 |
| Direct fail / SPA fail | 8 |
| Direct fail / SPA pass | 2 |
| Direct pass / SPA fail | 0 |
| Model, format, transport or evaluator errors | 0 |

Direct passed 10/20 and SPA passed 12/20. The two treatment-only fixes were
HumanEval/20 and HumanEval/31; both are preserved in the immutable
disagreement record. Exact McNemar two-sided p = 0.5; the task-paired bootstrap
95% interval for the +10 percentage-point difference is [0, +25] points.

Wall time was 5,194.507 ms mean for Direct and 6,369.102 ms for SPAall, a
1.226× aggregate ratio. p95 service wall time was 14,343.208 ms versus
12,906.789 ms, but this is partly caused by treatment outputting fewer tokens:
270.90 mean Direct completion tokens versus 212.35 SPA tokens. Median
within-task latency ratio was 1.534; the p95 ratio was 2.896. MLX-reported
mean peak allocation was 3.602 GB Direct versus 3.748 GB SPA, with maxima
3.674 and 3.877 GB.

The shared 24 GB Mac already had substantial swap use and other model services.
Pageouts did not increase during this run, but swap counters changed slightly;
this cannot be attributed cleanly to the test. The run is evidence about
quality and service latency, not an isolated memory-pressure study.

### Decision

SPAall clears the pre-registered **development** quality and aggregate-latency
gates exactly once: two fixes, zero regressions, mean ratio below 1.60×, p95
service ratio below 1.75×, and no response errors. It does **not** clear a
confirmatory efficacy standard. Parameters are now frozen. Do not tune on these
20 tasks; the only acceptable next SPA experiment is the disjoint confirmatory
gate described in REPLACEMENT_DESIGN.md.

The original ISRA runtime, its router, deployment scripts, legacy runners and
their tests are removed from the working tree. Git history preserves the
negative result and prior exploratory code.

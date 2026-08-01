# Replacement design: quality without self-critique loops

Status: **development screen completed; no confirmatory efficacy result**
Date: 2026-08-01

## Decision

Legacy ISRA is retired as a runtime design. The controlled Llama 3.1 8B smoke
found no fixes from either grounded repair or an unguided retry, while adding
27.8% and 35.8% wall time respectively. More importantly, the literature does
not support treating a differently prompted copy of the same weak model as an
independent verifier.

The product objective remains live:

> Improve the pass@1 quality of a local, resource-constrained model while
> keeping mean and tail latency close to one ordinary generation.

The repository will therefore keep the negative result and the trustworthy
evaluation core, but replace the iterative runtime with independently testable
one-pass or compiled mechanisms.

## Evidence that changed the design

### What is rejected

- Generic `draft -> same-model critique -> rewrite` has no new evidence and can
  turn correct answers into wrong ones. Prompted intrinsic self-correction is
  not reliable across tasks, especially for models at or below 13B.
- Different role temperatures alter diversity, not knowledge or independence.
- Self-generated expected outputs are not a correctness oracle. On
  Qwen2.5-Coder-7B they have been shown to reduce HumanEval+/MBPP+ quality.
- Best-of-N is not a deployable quality result unless a selector that cannot
  see hidden tests actually recovers the correct candidate. `any(A, B)` is only
  a capability ceiling.
- Speculative decoding preserves the target distribution. It may accelerate a
  winner later, but cannot be credited with a quality gain.

### What remains plausible

| Rank | Mechanism | Online cost | Evidence and caveat |
|---|---|---:|---|
| 1 | Better clean one-pass backbone or code specialist | 1 generation | Directly serves the objective. Current vendor scores are not latency-comparable and must be reproduced with thinking disabled. Existing custom `abliterated` checkpoints are not acceptable evidence. |
| 2 | Selective Prompt Anchoring (SPAall) | 1 generation with a two-path logits batch | The final ICML 2025 paper reports `+5.4 pp` HumanEval+ for both DeepSeek-Coder-6.7B and CodeLlama-7B when SPA is enabled for every task. Decode throughput changed from 12.1 to 10.2 tok/s and 14.5 to 10.6 tok/s respectively. The larger headline result used test-triggered activation and is not deployable without trusted public tests. Apple/MLX performance is unknown. |
| 3 | Offline prompt/concept optimization with an input-only router | 1 generation, slightly longer prompt | Near-zero routing latency, but expected gains are modest and prompt optimization can degrade small models. Rules must be selected only on development data. |
| 4 | Execution-verified reflection/correction distilled by SFT/LoRA | 1 generation after training | ReflectionCoder reports Llama-3.1-8B HumanEval+ `62.2 -> 68.3` and MBPP+ `59.3 -> 63.0` with no reflection call at inference. Its full-training recipe is expensive; a Mac QLoRA pilot would be an approximation and needs an ordinary-SFT control. |
| 5 | Adaptive second candidate plus a small external code ranker | `1 + q` generations plus ranker | A deployable verifier exists, but published CodeScaler gains use Best-of-8. First measure the Best-of-2 oracle ceiling; do not download or deploy a ranker if there is little recoverable diversity. |

Primary sources:

- [Selective Prompt Anchoring, ICML 2025](https://proceedings.mlr.press/v267/tian25a.html)
- [Intrinsic self-correction review, TACL 2024](https://aclanthology.org/2024.tacl-1.78/)
- [Small Language Models Need Strong Verifiers, ACL Findings 2024](https://aclanthology.org/2024.findings-acl.924/)
- [Self-generated tests for self-correction, ACL 2025](https://aclanthology.org/2025.acl-long.881/)
- [ReflectionCoder, ACL 2025](https://aclanthology.org/2025.acl-long.494/)
- [Concept Distillation, NAACL Industry 2025](https://aclanthology.org/2025.naacl-industry.52/)
- [Adaptive inference-time compute, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/ff414825df833edb8b1839e3d5d495e9-Abstract-Conference.html)

## Selected first prototype: MLX SPAall

SPA changes decoding rather than asking the model to review itself. For each
token it evaluates the same model on a two-item batch:

1. the original prompt;
2. an identical token sequence in which the selected natural-language prompt
   tokens are replaced by a model-specific neutral/padding token.

For original logits `L` and masked-prompt logits `L_mask`, fixed-strength SPA
uses the paper's first-order approximation:

```text
L_spa = omega * L + (1 - omega) * L_mask
```

The generated token is appended to both cache branches. There is still one
answer and no hidden-test feedback, critic, retry, or answer selector. Batching
the branches preserves prompt positions and may amortize model-weight reads on
Apple unified memory; actual latency must be measured rather than inferred.

Frozen development configuration:

- model: existing `Meta-Llama-3.1-8B-Instruct-3bit`;
- decoding: greedy, beam 1;
- activation: SPA on every treatment task (`SPAall`);
- anchor: natural-language description in the last user message;
- mask token: Llama 3 `<|finetune_right_pad_id|>`;
- anchoring strength: `omega = 1.28`, the paper's cross-model preset;
- baseline and treatment use the same messages, token limit, evaluator and task
  order; only the logits intervention differs;
- evaluator tests never enter the generation process.

The implementation must make `omega = 1` an exact direct-decoding identity
before any quality run. Any token divergence at `omega = 1` is an implementation
failure, not a model result.

## Preregistered gates

### Stage 0: implementation integrity

- Unit tests cover prompt masking, token-position preservation and logits
  arithmetic.
- On five fixed prompts, `direct` and `SPA(omega=1)` must produce identical
  token IDs, finish reasons and extracted code.
- Canonical EvalPlus solutions pass and known-bad mutations fail in the pinned
  container.
- No incremental swap/pageouts or memory-pressure warning caused by the run on
  the 24 GB Mac.

### Stage 1: bounded development screen

Use HumanEval+ tasks 20--31 and 33--40, which were not used in the earlier ISRA
smoke. HumanEval/32 is excluded before any model request because the pinned
EvalPlus 0.3.1 `find_zero` special oracle executes `continue` before updating
its progress/details arrays; consequently it reports even the canonical
solution as `fail (0/0)`. HumanEval/40 is the preregistered replacement, keeping
the screen at 20 tasks. This is a mechanism screen, not a publishable benchmark.

Go only if all are true:

- at least two `direct wrong -> SPA right` flips;
- fixes exceed `direct right -> SPA wrong` regressions;
- mean wall time is at most `1.60x` direct and p95 at most `1.75x`;
- no increase in model/format/infrastructure errors.

Kill this SPA configuration immediately if net flips are non-positive, the
latency limit is exceeded, or memory pressure causes swap. Do not tune repeatedly
on the same 20 tasks.

### Stage 1 outcome (2026-08-01)

The frozen run `real-llama31-8b3bit-spa-dev-20-v2` produced two fixes, zero
regressions and a `1.226x` mean wall-time ratio (12/20 SPA versus 10/20
Direct). It passed this development gate but is not statistically confirmatory:
exact McNemar `p=0.5` and the task-paired bootstrap interval includes zero.
The full result, including the shared-host memory caveat, is recorded in
[`BENCHMARKS.md`](BENCHMARKS.md) and [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md).

### Stage 2: confirmatory gate after a positive screen

Freeze the implementation and hyperparameters, then use at least 100 disjoint
objective code tasks, preferably a date-filtered post-cutoff LiveCodeBench Code
Generation slice. Required result:

- at least `+2 pp` pass@1;
- lower bound of a task-paired 95% bootstrap interval above zero;
- exact McNemar test reported;
- mean latency at most `1.30x`, p95 at most `1.50x`;
- output token count no more than 10% above direct.

HumanEval+/MBPP+ remain regression suites, not the sole headline evidence.

## Independent next arms

These must not be mixed into the SPA attribution run:

1. **Clean backbone A/B.** Llama 3.1 8B versus clean Qwen3.5-9B
   non-thinking and Qwen2.5-Coder-7B, one call each. Gate: at least `+3 pp`,
   mean latency `<=1.20x`, p95 `<=1.50x`. Multi-gigabyte downloads require
   explicit approval under the session constraints.
2. **Correction distillation.** Generate only execution-verified `fail -> pass`
   training trajectories, train a QLoRA adapter, fuse it, and compare against
   both the frozen base and an ordinary-SFT adapter with matched data/tokens.
   Gate: `+3 pp` versus base, `+2 pp` versus ordinary SFT, and no more than 5%
   inference latency/token overhead.
3. **Best-of-2 ceiling.** Reuse fixed candidate pairs. Stop before adding a
   verifier unless `any(A, B)` offers at least a meaningful two-point ceiling
   on a sufficiently large set. A verifier must produce more recoveries than
   regressions on held-out data.

## Repository boundary

Keep:

- the pinned evaluator, paired result schema, retry/resume logic, immutable
  manifests, statistical analysis and negative-result logs.

Remove from the active runtime:

- four-phase orchestration, same-model critic, DEER, semantic compression,
  self-confidence stopping, authoritative self-generated tests, session state,
  and launchd configuration dedicated to the retired ISRA service.

Git history is the archive for deleted runtime code. Historical benchmark
records and documents remain explicitly labeled and are never rewritten.

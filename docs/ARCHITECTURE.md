# Active architecture: one-pass selective prompt anchoring

This document describes the active runtime. The previous multi-phase ISRA
orchestrator was removed because its same-model critique/rewrite loop was not
an independent source of correctness and produced no paired fixes in the
controlled 8B screen.

## Request flow

~~~text
OpenAI client
    │ POST /v1/chat/completions
    ▼
anchored_generation.py
    │
    ├─ direct mode: batch [visible prompt]
    │                  └─ logits L ────────────────┐
    │                                               ▼
    └─ SPA mode:   batch [visible prompt, masked prompt]
                       └─ logits L, L_mask ── L_spa = ωL + (1−ω)L_mask
                                                        │
                                                        ▼
                                             greedy next-token choice
                                                        │
                                                        ▼
                                            append same token to cache(s)
                                                        │
                                                        ▼
                                               OpenAI-compatible answer
~~~

SPA keeps generated positions aligned by appending the selected token to both
branches. Only selected natural-language tokens in the last user message are
masked; punctuation, role markers and code are preserved. No evaluator input,
public test output, critic, retry, candidate selector or self-generated test is
available to the decoder.

## Invariants

- Direct and SPA receive byte-identical visible messages, seed, token limit and
  decoding settings.
- A strength of omega = 1 is an exact identity: direct and SPA must emit the
  same token IDs. This is tested before any quality run.
- SPA uses a forward batch of two only; it returns one answer and reports one
  LLM call.
- Greedy decoding is fixed: temperature 0, top-p 1, beam 1.
- The MLX server serializes requests. MLX streaming state is thread-local, so
  generation remains on the model-loading thread.
- Response metadata records token-ID hashes, prompt/completion lengths, decoder
  timing, throughput, masking and memory allocator high-water mark.

## Research configuration

The initial development configuration is intentionally frozen:

| Field | Value |
|---|---|
| Model | Meta-Llama-3.1-8B-Instruct, MLX affine 3-bit group size 64 |
| Anchor | natural-language portion of the last user prompt |
| Mask token | Llama 3 finetune-right-pad token, id 128004 |
| Strength | omega = 1.28 |
| Modes | direct and SPAall |
| Evaluator | EvalPlus 0.3.1 / HumanEvalPlus v0.1.10 in a pinned Docker image |

The rationale, source evidence and the gates required before changing this
configuration are in [REPLACEMENT_DESIGN.md](REPLACEMENT_DESIGN.md).

## Benchmark isolation

The paired runner invokes the same evaluator after each answer, never before.
It stores an append-only result record per
(run, model, task, mode, seed), randomizes mode order deterministically per
task and resumes only exact matching manifests.

Generated code executes inside the evaluator container. Mount the repository
read-only and only the results directory writable. The development screen
excluded HumanEval/32 before inference because EvalPlus 0.3.1's special
find_zero oracle falsely rejects its canonical solution; details are recorded
in the experiment log.

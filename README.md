# ISRA — one-pass local-model quality experiments

> [!WARNING]
> The original iterative self-refinement runtime has been retired. It neither
> demonstrated a paired quality gain on the controlled Llama 3.1 8B smoke nor
> had a reliable theoretical basis for treating the same weak model as an
> independent critic.

This repository now tests compact, falsifiable mechanisms for improving a local
model's pass@1 quality without a second answer, hidden-test feedback, or an LLM
review loop.

The active prototype is **Selective Prompt Anchoring (SPAall)**: one model
forward pass is evaluated on the visible prompt and a position-preserving masked
copy, then the two logit vectors are combined before choosing the next token.
It remains one answer and one model call from the client's perspective.

~~~text
visible prompt ─┐
                ├─ same local model, batch size 2 ─ logit contrast ─ answer
masked prompt  ─┘
~~~

## Current evidence

| Experiment | Model / tasks | Result | Interpretation |
|---|---|---|---|
| Legacy iterative repair | Llama 3.1 8B, 20 HumanEval+ tasks | 0 fixes, +27.8% wall time | Rejected. |
| Legacy unguided retry | Llama 3.1 8B, same 20 tasks | 0 fixes, +35.8% wall time | Rejected. |
| SPAall development screen | Llama 3.1 8B 3-bit, 20 preregistered HumanEval+ tasks | 12/20 vs 10/20; 2 fixes, 0 regressions; mean wall time 1.226× | Promising mechanism screen only, **not** a publishable efficacy claim. |

The SPA screen is intentionally small. Its exact McNemar p-value is 0.5 and
the paired bootstrap interval includes zero. The next step is a frozen,
disjoint, post-cutoff code benchmark; do not tune SPA on the development tasks.
Full methods, limits and source-backed alternatives are in
[docs/REPLACEMENT_DESIGN.md](docs/REPLACEMENT_DESIGN.md). The full audit trail
is in [docs/EXPERIMENT_LOG.md](docs/EXPERIMENT_LOG.md).

## Run the prototype on Apple Silicon

Use an isolated Python environment with MLX, MLX-LM and aiohttp. The evaluated
remote setup used mlx-lm 0.31.3, mlx 0.32.0 and the existing
Meta-Llama-3.1-8B-Instruct-3bit checkpoint.

~~~bash
python -m pip install -r requirements-mlx.txt

python anchored_generation.py \
  --model-path /path/to/Meta-Llama-3.1-8B-Instruct-3bit \
  --host 127.0.0.1 \
  --port 8085
~~~

The server exposes OpenAI-compatible /v1/chat/completions, /v1/models and
/health. Select either llama31-8b3bit-direct or llama31-8b3bit-spa; the
benchmark runner also sends an explicit decoding_mode field.

~~~bash
curl http://127.0.0.1:8085/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "llama31-8b3bit-spa",
    "decoding_mode": "spa",
    "temperature": 0,
    "max_tokens": 800,
    "messages": [{"role": "user", "content": "Write a Python factorial function."}]
  }'
~~~

This research server serializes requests because MLX streams are thread-local.
It is not a multi-tenant production deployment.

## Evaluate without leaking hidden tests

Run generated code only in the pinned EvalPlus container. The runner first
proves that canonical solutions pass and deliberate mutations fail; it stores
an immutable manifest, response hashes, decoding metadata, order and evaluator
version.

~~~bash
docker build -f benchmarks/Dockerfile.evalplus -t isra-evalplus:0.3.1 .

docker run --rm \
  --add-host=host.docker.internal:host-gateway \
  -v "$PWD:/workspace:ro" \
  -v "$PWD/benchmark_runs:/results" \
  -w /workspace \
  isra-evalplus:0.3.1 \
  benchmarks/decoding_benchmark.py \
  --endpoint-url http://host.docker.internal:8085/v1/chat/completions \
  --model-id llama31-8b3bit \
  --run-root /results
~~~

For evaluation safety, keep the repository read-only in the container and
mount only the results directory writable. Benchmark runs are intentionally
ignored by Git.

## Repository layout

~~~text
anchored_generation.py             MLX SPAall OpenAI-compatible server
benchmarks/decoding_benchmark.py   paired direct-vs-SPA evaluator runner
benchmarks/paired_core.py          code extraction, evaluator and result storage
benchmarks/harness_common.py       provenance and HumanEval+ dataset helpers
docs/REPLACEMENT_DESIGN.md         research decision and preregistered gates
docs/EXPERIMENT_LOG.md             append-only experiment record
~~~

Historical ISRA claims and code were removed from the working tree; the
negative result remains documented and recoverable through Git history.

All rights reserved.

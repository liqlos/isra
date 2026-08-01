# ISRA — Iterative Self-Refinement Architecture

> [!WARNING]
> **Experimental research prototype — no demonstrated accuracy improvement.**
> ISRA explores whether a local model can improve its own answers through
> iterative critique, execution feedback and revision. A controlled 20-task
> HumanEval+ smoke on Llama 3.1 8B found **zero paired fixes** from the tested
> repair pipeline while adding roughly 28% wall time; an ordinary retry also
> produced zero fixes at still higher cost. Earlier 35B runs were confounded and
> likewise did not beat Direct. Do not present or deploy this repository as a
> proven quality or reliability layer.

The repository is retained as a negative-result experiment, a reproducible
evaluation harness, and a starting point for better-grounded work on external
verification or trained correction. See
[`docs/THEORY_REVIEW.md`](docs/THEORY_REVIEW.md) and
[`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md).

## What ISRA Does

ISRA sits between your client (Goose, curl, any OpenAI-compatible app) and your
LLM backend. Its legacy/default mode runs the following **4-phase experimental
pipeline** with optional early exits:

```
Client → ISRA (:8083)
  → Phase 0: Quick consensus check (2 parallel calls, ~1s for trivial queries)
  → Phase 1: Deep Thinking (reasoning + code/math generation)
  → DEER: Dynamic Early Exit (re-solve math, skip phases 2-4 if match)
  → Phase 2: Skeptical Review (critic finds errors, assigns confidence)
  → Phase 3: Essence Extraction (merge conclusions + critique)
  → [Decision Gate: confidence ≥ 75% OR stagnation OR max iters]
  → Phase 4: Final Synthesis (or direct code/math return)
  → OpenAI-format response
```

If the critic finds errors, legacy ISRA may re-run Phase 1 with critique
feedback, up to three times. This behavior has not demonstrated a net accuracy
gain. In particular, self-generated tests are not a trustworthy oracle and can
cause false corrections. New diagnostic variants therefore use only syntax and
task/user-provided examples as trusted triggers, with rollback when a repair
does not clear the complete available trusted suite.

## Current Result

The first trustworthy mechanism smoke used a pinned Linux EvalPlus evaluator,
HumanEval+ `v0.1.10`, exact byte-for-byte fan-out of the initial candidate, and
`Meta-Llama-3.1-8B-Instruct-3bit` on a 24 GB Mac.

| Variant | Pass | Eval fail | Error / skipped | Extra calls | Paired fixes | Added wall time |
|---|---:|---:|---:|---:|---:|---:|
| Phase 1 only | 12 | 7 | 1 / 0 | 0 | — | — |
| Grounded one-repair | 12 | 7 | 0 / 1 | 7 | 0 | +27.8% |
| Unguided retry | 12 | 7 | 0 / 1 | 7 | 0 | +35.8% |

This 20-task smoke is descriptive, not a statistically powered benchmark.
Nevertheless, it is sufficient to reject a claim that this tested variant
already improves the 8B model. A larger confirmatory run is not justified until
a different mechanism produces real fixes on a disjoint development set.

Older Qwen 35B, GSM8K and MMLU-Pro numbers are preserved in
[`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) as historical exploratory runs. They
were not fully paired, mixed sampling/pipeline settings, and must not be used as
evidence that ISRA improves quality.

## Key Features

- **Phase 0 — Quick Consensus**: Two parallel low-token calls. If answers match, return immediately (~1s for factual queries).
- **DEER — Dynamic Early Exit**: For math tasks, independently re-solve after Phase 1. If answers match, skip phases 2-4 (saves ~15s on 75% of math tasks).
- **Doctest Execution**: Runs `>>>` examples from docstrings in subprocess. If code fails, ISRA iterates with the error message.
- **Legacy Self-Generated Tests**: Preserved for historical ablation only; they are not a trusted correctness oracle.
- **FIX MODE**: On iteration 2+, passes the previous code + error to the model with "fix it, don't rewrite from scratch."
- **Same-Model Critic/Re-derivation**: An experimental diagnostic; different prompts or temperatures do not make it an independent verifier.
- **Stagnation Detection**: If the same error signature repeats across iterations, terminates early.
- **Session State**: Accumulates confirmed insights across conversation turns.
- **OpenAI-Compatible**: Drop-in replacement for any OpenAI API client.

## Quick Start

### Prerequisites

- Python 3.11+
- An OpenAI-compatible LLM backend (e.g., MLX server, vLLM, Ollama)
- `aiohttp` (`pip install aiohttp`)

### Run ISRA

```bash
# Point ISRA at your backend (default: http://127.0.0.1:8080)
export ISRA_ROUTER_URL=http://127.0.0.1:8080
export ISRA_BACKEND_MODEL=qwen3-a3b

# Start ISRA
python isra_orchestrator.py --host 0.0.0.0 --port 8083
```

### Test It

```bash
curl http://localhost:8083/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "isra-a3b",
    "messages": [{"role": "user", "content": "Write a function that returns the factorial of n"}],
    "max_tokens": 2048,
    "temperature": 0
  }'
```

### Use with Goose / Any OpenAI Client

Point your client at `http://localhost:8083` with model name `isra-a3b`.

### macOS Auto-Start (launchd)

```bash
# Clone and install
git clone <repo-url> isra
cd isra

# Install with: install.sh <venv-python> <model-path> [install-dir] [log-dir]
./deploy/install.sh /opt/homebrew/bin/python3 /path/to/your/model
```

## Configuration

All settings via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `ISRA_ROUTER_URL` | `http://127.0.0.1:8080` | Backend LLM URL |
| `ISRA_BACKEND_MODEL` | `qwen3-a3b` | Model name for backend |
| `ISRA_PORT` | `8083` | ISRA listen port |
| `ISRA_HOST` | `0.0.0.0` | ISRA bind address |
| `ISRA_MAX_ITERS` | `3` | Max pipeline iterations |
| `ISRA_CONFIDENCE_THRESHOLD` | `75` | Critic confidence to terminate |
| `ISRA_STAGNATION_THRESHOLD` | `0.75` | State similarity to detect stagnation |
| `ISRA_SESSION_TTL` | `3600` | Session expiry (seconds) |

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed pipeline design.

## Benchmarks

See [docs/BENCHMARKS.md](docs/BENCHMARKS.md) for full benchmark methodology and results.

## Project Structure

```
isra/
├── isra_orchestrator.py    # Main ISRA pipeline + HTTP server
├── model_router.py         # Optional: routes between multiple MLX models
├── requirements.txt        # Python dependencies (aiohttp)
├── deploy/
│   ├── install.sh          # macOS launchd installer
│   ├── mlx_watchdog.sh     # Process watchdog
│   └── *.plist.template    # launchd service templates
├── benchmarks/
│   ├── full_benchmark.py   # HumanEval + GSM8K benchmark
│   ├── bench_he_isra.py    # HumanEval-only quick test
│   └── bench_gsm_isra.py   # GSM8K-only quick test
└── docs/
    ├── ARCHITECTURE.md     # Pipeline design
    └── BENCHMARKS.md       # Results & methodology
```

## License

Private. All rights reserved.

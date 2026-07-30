# ISRA — Iterative Self-Refinement Architecture

A multi-phase LLM orchestrator that wraps any OpenAI-compatible backend with a critic-and-iterate pipeline, achieving significant accuracy improvements over direct model calls on code generation and math reasoning tasks.

## What ISRA Does

ISRA sits between your client (Goose, curl, any OpenAI-compatible app) and your LLM backend. Instead of a single forward pass, it runs a **4-phase pipeline** with optional early exits:

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

If the critic finds errors, ISRA **iterates** — re-running Phase 1 with the critique feedback, up to 3 times. For code tasks, it also runs **doctest execution feedback** and **self-generated test cases** in subprocess to catch logic errors.

## Benchmark Results

Tested with Qwen3.5-35B-A3B (3-bit MLX, local Mac mini M4) on full official benchmarks:

| Benchmark | Tasks | ISRA | Direct (vanilla) | Official Qwen3.5-35B-A3B |
|-----------|-------|------|-----------------|--------------------------|
| HumanEval | 164 | 143/164 = 87.2% | 149/164 = 90.9% | ~74.6% (LiveCodeBench v6)* |
| GSM8K | 300 | 267/300 = 89.0% | ~75% (partial) | ~95% (est) |

*LiveCodeBench v6 is a different (harder) code benchmark than HumanEval.

**Honest finding**: For a strong MoE model like Qwen3.5-35B-A3B, ISRA does NOT improve accuracy on standard benchmarks. The base model is already good enough that the orchestrator's overhead (self-test false negatives, iteration noise, timeout failures) outweighs the benefits. ISRA is more valuable for weaker models or specialized reasoning tasks.

See [docs/BENCHMARKS.md](docs/BENCHMARKS.md) for full methodology and analysis.

## Key Features

- **Phase 0 — Quick Consensus**: Two parallel low-token calls. If answers match, return immediately (~1s for factual queries).
- **DEER — Dynamic Early Exit**: For math tasks, independently re-solve after Phase 1. If answers match, skip phases 2-4 (saves ~15s on 75% of math tasks).
- **Doctest Execution**: Runs `>>>` examples from docstrings in subprocess. If code fails, ISRA iterates with the error message.
- **Self-Generated Tests**: For code tasks without doctests, generates test cases from the docstring and executes them.
- **FIX MODE**: On iteration 2+, passes the previous code + error to the model with "fix it, don't rewrite from scratch."
- **Critic with Independent Re-derivation**: For math, the critic independently solves the problem before reviewing, catching arithmetic errors.
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

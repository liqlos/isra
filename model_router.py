#!/usr/bin/env python3
"""
Smart model router for Mac mini MLX servers.
Runs ON the Mac mini, listens on port 8080, routes OpenAI-format requests
to the correct backend based on the `model` field. When switching models,
stops the current model and starts the requested one via launchd.

Models:
  - "qwen3-a3b"       → port 8081 (mlx_lm, 3-bit, launchd: local.qwen3-a3b)
  - "qwen3-27b-4bit"  → port 8081 (mlx_lm, 4-bit, launchd: local.qwen3-27b-4bit)
  - "qwen3-27b-5bit"  → port 8085 (mlx_lm, 5-bit)

Only ONE model runs at a time (24GB RAM constraint).
"""
import argparse
import asyncio
import hashlib
import json
import logging
import os
import subprocess
import time

import aiohttp
from aiohttp import web

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("router")

# ── Model registry ──────────────────────────────────────────────────────────
MODELS = {
    "qwen3-a3b": {
        "port": 8081,
        "label": "local.qwen3-a3b",
        "plist": "local.qwen3-a3b.plist",
        "display": "Qwen3.6-35B-A3B abliterated 3-bit (MoE 3B active, 80 tok/s)",
        "ctx": 65536,
        "startup_wait": 40,
        "backend_model": os.environ.get("MLX_MODEL_PATH_A3B", "/opt/mlx-models/Qwen3.6-35B-A3B-abliterated-mixed36"),
    },
    "qwen3-27b-4bit": {
        "port": 8081,
        "label": "local.qwen3-27b-4bit",
        "plist": "local.qwen3-27b-4bit.plist",
        "display": "Qwen3.6-27B abliterated 4-bit (high quality, ~20 tok/s, 16GB)",
        "ctx": 16384,
        "startup_wait": 40,
        "backend_model": os.environ.get("MLX_MODEL_PATH_27B_4BIT", "/opt/mlx-models/Qwen3.6-27B-abliterated-4bit"),
    },
    "qwen3-27b-5bit": {
        "port": 8085,
        "label": "local.qwen3-27b-5bit",
        "plist": "local.qwen3-27b-5bit.plist",
        "display": "Qwen3.6-27B abliterated 5-bit MLX (high quality, ~9 tok/s, 19GB)",
        "ctx": 8192,
        "startup_wait": 40,
        "backend_model": os.environ.get("MLX_MODEL_PATH_27B_5BIT", "/opt/mlx-models/Qwen3.6-27B-abliterated-5bit-MLX"),
    },
}

DEFAULT_MODEL = "qwen3-a3b"
LAUNCHAGENTS_DIR = os.path.expanduser("~/Library/LaunchAgents")

# State
_current_model: str | None = None
_switch_lock = asyncio.Lock()


def run_cmd(cmd: str, timeout: int = 30) -> tuple[int, str]:
    """Run a local shell command."""
    try:
        r = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
        return r.returncode, r.stdout + r.stderr
    except subprocess.TimeoutExpired:
        return -1, "timeout"
    except Exception as e:
        return -1, str(e)


async def check_port_open(port: int, timeout: float = 1.5) -> bool:
    """Check if a port is accepting connections locally."""
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection("127.0.0.1", port), timeout=timeout
        )
        writer.close()
        await writer.wait_closed()
        return True
    except (OSError, asyncio.TimeoutError):
        return False


async def stop_all_models():
    """Stop all running model servers."""
    log.info("Stopping all models...")
    # Unload all launchd agents
    for name, info in MODELS.items():
        plist_path = os.path.join(LAUNCHAGENTS_DIR, info["plist"])
        if os.path.exists(plist_path):
            run_cmd(f"launchctl unload {plist_path} 2>/dev/null", timeout=10)
    # Also kill any stragglers
    run_cmd("pkill -f 'infer.*--serve' 2>/dev/null; pkill -f 'mlx_.*server' 2>/dev/null; pkill -f 'llama-server' 2>/dev/null", timeout=10)
    await asyncio.sleep(3)
    log.info("All models stopped")


async def start_model(model_name: str) -> bool:
    """Start a model via launchd."""
    info = MODELS[model_name]
    plist_path = os.path.join(LAUNCHAGENTS_DIR, info["plist"])

    if not os.path.exists(plist_path):
        log.error("plist not found: %s", plist_path)
        return False

    log.info("Starting model '%s' (plist=%s)...", model_name, info["plist"])
    rc, out = run_cmd(f"launchctl load {plist_path} 2>&1", timeout=10)
    if rc != 0 and "already loaded" not in out.lower():
        log.warning("launchctl load: %s", out.strip()[-200:])

    # Wait for port to be ready
    max_wait = info["startup_wait"]
    port = info["port"]
    start = time.time()
    while time.time() - start < max_wait:
        if await check_port_open(port):
            elapsed = time.time() - start
            log.info("Model '%s' ready on port %d (%.1fs)", model_name, port, elapsed)
            await warmup_model(model_name)
            return True
        await asyncio.sleep(2)

    log.error("Model '%s' failed to start within %ds", model_name, max_wait)
    return False


async def warmup_model(model_name: str):
    """Send a tiny request to warm up the model."""
    info = MODELS[model_name]
    port = info["port"]
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    body = {"model": info["backend_model"], "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 1}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=body, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                await resp.read()
                log.info("Warmup '%s' done (status %d)", model_name, resp.status)
    except Exception as e:
        log.warning("Warmup '%s': %s", model_name, e)


async def ensure_model_loaded(model_name: str) -> bool:
    """Ensure the requested model is running. Switch if needed."""
    global _current_model

    if model_name not in MODELS:
        log.warning("Unknown model '%s', using default '%s'", model_name, DEFAULT_MODEL)
        model_name = DEFAULT_MODEL

    if _current_model == model_name:
        info = MODELS[model_name]
        if await check_port_open(info["port"], timeout=1.0):
            return True
        log.warning("Model '%s' port not responding, restarting...", model_name)
        _current_model = None

    async with _switch_lock:
        if _current_model == model_name:
            info = MODELS[model_name]
            if await check_port_open(info["port"], timeout=1.0):
                return True

        log.info("Switching: %s → %s", _current_model, model_name)
        await stop_all_models()
        ok = await start_model(model_name)
        if ok:
            _current_model = model_name
        return ok


def resolve_model_name(request_model: str) -> str:
    """Map various model name aliases to canonical names."""
    if not request_model:
        return DEFAULT_MODEL
    m = request_model.lower().strip()
    if m in MODELS:
        return m
    if "122" in m or "flash" in m:
        return "122b-moe"
    if "gpt-oss" in m and ("uncen" in m or "ablit" in m or "derest" in m):
        return "gpt-oss-20b-uncensored"
    if "gpt-oss" in m or "gptoss" in m or "oss-20" in m:
        return "gpt-oss-20b"
    if "9b" in m or "dense" in m or "huihui" in m:
        return "9b-dense"
    if "27b" in m and ("8bit" in m or "8-bit" in m or "q8" in m or "gguf" in m):
        return "qwen3-27b-8bit"
    return DEFAULT_MODEL


# ── Proxy handlers ───────────────────────────────────────────────────────────

# Session tracking for 122B (avoids re-prefilling entire conversation)
# Key: hash of first message (system prompt) → session_id
# When Goose sends the same system prompt, we reuse the session
_session_map: dict[str, str] = {}  # prompt_hash → session_id


def _compute_prompt_hash(messages: list) -> str:
    """Compute a hash of the first 1-2 messages (system + first user) to identify a session."""
    if not messages:
        return ""
    # Use first message (usually system) + second message (first user turn)
    key_parts = []
    for msg in messages[:2]:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if isinstance(content, list):
            content = str(content[:100])
        key_parts.append(f"{role}:{content[:200]}")
    return hashlib.md5("|".join(key_parts).encode()).hexdigest()[:16]


async def proxy_request(request: web.Request, method: str) -> web.StreamResponse:
    """Proxy a request to the backend, ensuring the model is loaded first."""
    body = await request.read()
    body_json = {}
    try:
        body_json = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        log.warning("Failed to parse body as JSON: %s (len=%d, ct=%s)",
                    e, len(body), request.headers.get("Content-Type", "?"))

    requested_model = body_json.get("model", "")
    canonical = resolve_model_name(requested_model)

    # Rewrite model field to the backend's expected model name/path.
    if body_json:
        body_json["model"] = MODELS[canonical]["backend_model"]

        # For 122B: add session_id to enable continuation (skip re-prefill)
        # The 122B server supports session_id in the request body.
        # If the same system prompt is sent again, we reuse the session
        # so the server skips prefilling already-processed tokens.
        if canonical == "122b-moe":
            messages = body_json.get("messages", [])
            prompt_hash = _compute_prompt_hash(messages)
            if prompt_hash:
                if prompt_hash not in _session_map:
                    _session_map[prompt_hash] = f"goose-{prompt_hash}"
                body_json["session_id"] = _session_map[prompt_hash]
                log.info("  session_id=%s (hash=%s, msgs=%d)",
                         body_json["session_id"], prompt_hash, len(messages))
            else:
                log.warning("  no prompt_hash! messages=%d", len(messages))

        body = json.dumps(body_json).encode()

    ok = await ensure_model_loaded(canonical)
    if not ok:
        return web.json_response(
            {"error": {"message": f"Failed to load model '{canonical}'", "type": "server_error"}},
            status=503,
        )

    info = MODELS[canonical]
    backend_port = info["port"]
    backend_url = f"http://127.0.0.1:{backend_port}{request.path_qs}"

    headers = {k: v for k, v in request.headers.items()
               if k.lower() not in ("host", "content-length", "transfer-encoding")}
    # Set correct Content-Length for rewritten body
    headers["Content-Length"] = str(len(body))
    is_stream = body_json.get("stream", False)

    log.info("→ %s %s (model=%s, stream=%s)", method, request.path, canonical, is_stream)

    try:
        timeout = aiohttp.ClientTimeout(total=3600, sock_read=600)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.request(method, backend_url, data=body, headers=headers) as backend_resp:
                if is_stream:
                    resp = web.StreamResponse(
                        status=backend_resp.status,
                        headers={
                            "Content-Type": "text/event-stream",
                            "Cache-Control": "no-cache",
                            "Connection": "keep-alive",
                        },
                    )
                    await resp.prepare(request)
                    async for chunk in backend_resp.content.iter_any():
                        await resp.write(chunk)
                    await resp.write_eof()
                    return resp
                else:
                    data = await backend_resp.read()
                    return web.Response(
                        status=backend_resp.status,
                        body=data,
                        content_type=backend_resp.content_type,
                    )
    except aiohttp.ClientError as e:
        log.error("Proxy error for '%s': %s", canonical, e)
        return web.json_response(
            {"error": {"message": f"Backend error: {e}", "type": "server_error"}},
            status=502,
        )


async def handle_chat_completions(request: web.Request):
    return await proxy_request(request, "POST")


async def handle_completions(request: web.Request):
    return await proxy_request(request, "POST")


async def handle_models(request: web.Request):
    """Return list of available models (OpenAI /v1/models format)."""
    models_list = []
    for name, info in MODELS.items():
        models_list.append({
            "id": name,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "local",
        })
    return web.json_response({"object": "list", "data": models_list})


async def handle_health(request: web.Request):
    """Health check."""
    return web.json_response({
        "status": "ok",
        "current_model": _current_model,
        "models": {k: {"port": v["port"], "display": v["display"]} for k, v in MODELS.items()},
    })


async def handle_switch(request: web.Request):
    """Force switch to a model. GET /switch?model=9b-dense"""
    model = request.query.get("model", "")
    canonical = resolve_model_name(model)
    ok = await ensure_model_loaded(canonical)
    return web.json_response({
        "ok": ok,
        "model": canonical,
        "display": MODELS[canonical]["display"],
    })


def create_app() -> web.Application:
    app = web.Application(client_max_size=50 * 1024 * 1024)
    app.router.add_post("/v1/chat/completions", handle_chat_completions)
    app.router.add_post("/v1/completions", handle_completions)
    app.router.add_get("/v1/models", handle_models)
    app.router.add_get("/health", handle_health)
    app.router.add_get("/switch", handle_switch)
    return app


async def on_startup(app):
    """Detect which model is currently running on startup."""
    global _current_model
    for name, info in MODELS.items():
        if await check_port_open(info["port"], timeout=1.0):
            _current_model = name
            log.info("Detected running model: %s (port %d)", name, info["port"])
            return
    log.info("No model currently running. Will load on first request.")


def main():
    parser = argparse.ArgumentParser(description="Smart model router for Mac mini MLX")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    log.info("=== Smart Model Router ===")
    log.info("Listening on %s:%d", args.host, args.port)
    log.info("Models:")
    for name, info in MODELS.items():
        log.info("  %-15s → port %d (%s)", name, info["port"], info["display"])

    app = create_app()
    app.on_startup.append(on_startup)
    web.run_app(app, host=args.host, port=args.port, access_log=None)


if __name__ == "__main__":
    main()

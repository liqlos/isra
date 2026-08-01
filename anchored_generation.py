#!/usr/bin/env python3
"""OpenAI-compatible MLX server for direct and prompt-anchored decoding.

The anchored path implements the first-order logits intervention from
"Selective Prompt Anchoring for Code Generation" (ICML 2025). It produces one
answer in one decoding loop; it does not run a critic, retry, hidden tests, or
an answer selector.

MLX imports are intentionally lazy so prompt preparation and manifest logic can
be unit-tested on machines without a usable model checkpoint.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import difflib
import hashlib
import json
import math
import os
import re
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from aiohttp import web


PAPER_URL = "https://proceedings.mlr.press/v267/tian25a.html"
DEFAULT_STRENGTH = 1.28
DEFAULT_ANCHOR_MODE = "natural_language"
DEFAULT_MASK_TOKEN = "<|finetune_right_pad_id|>"
SUPPORTED_ANCHOR_MODES = ("natural_language", "last_user")
SUPPORTED_DECODING_MODES = ("direct", "spa")


@dataclass(frozen=True)
class AnchoringSpec:
    enabled: bool
    strength: float
    anchor_mode: str
    mask_token: str

    def __post_init__(self) -> None:
        if not math.isfinite(self.strength):
            raise ValueError("anchoring strength must be finite")
        if self.anchor_mode not in SUPPORTED_ANCHOR_MODES:
            raise ValueError(
                f"unsupported anchor mode {self.anchor_mode!r}; "
                f"expected one of {SUPPORTED_ANCHOR_MODES}"
            )


@dataclass
class GenerationResult:
    text: str
    token_ids: list[int]
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    metadata: dict[str, Any]


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if not isinstance(content, str):
        raise ValueError("only text chat messages are supported")
    return content


def _last_user_index(messages: Sequence[dict[str, Any]]) -> int:
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") == "user":
            return index
    raise ValueError("at least one user message is required")


def remove_natural_language_description(content: str) -> str:
    """Return a reference prompt with the code-task description removed.

    HumanEval-style prompts keep their function signature, quotes, and public
    doctest examples. The text before the first doctest inside the docstring is
    removed. Token differences between this reference and the original identify
    the natural-language tokens to mask. For an unrecognized prompt, returning
    an empty string conservatively anchors the complete user message.
    """

    match = re.search(
        r"(?P<quote>\"\"\"|''')(?P<body>.*?)(?P=quote)",
        content,
        flags=re.DOTALL,
    )
    if not match:
        return ""

    body = match.group("body")
    lines = body.splitlines(keepends=True)
    doctest_start = next(
        (index for index, line in enumerate(lines) if line.lstrip().startswith(">>>")),
        len(lines),
    )
    description = "".join(lines[:doctest_start])
    if not description.strip():
        return ""

    retained = "".join(lines[doctest_start:])
    return content[: match.start("body")] + retained + content[match.end("body") :]


def reference_messages(
    messages: Sequence[dict[str, Any]], anchor_mode: str
) -> list[dict[str, Any]]:
    """Build a prompt reference with the selected user semantics removed."""

    if anchor_mode not in SUPPORTED_ANCHOR_MODES:
        raise ValueError(f"unsupported anchor mode: {anchor_mode}")
    result = copy.deepcopy(list(messages))
    user_index = _last_user_index(result)
    content = _message_text(result[user_index])
    result[user_index]["content"] = (
        remove_natural_language_description(content)
        if anchor_mode == "natural_language"
        else ""
    )
    return result


def normalize_token_ids(value: Any) -> list[int]:
    """Normalize tokenizer list/array/tensor output to one flat integer list."""

    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        raise TypeError(f"unsupported tokenizer output: {type(value).__name__}")
    if value and isinstance(value[0], list):
        if len(value) != 1:
            raise ValueError("only a single prompt is supported")
        value = value[0]
    if not all(isinstance(item, int) for item in value):
        raise TypeError("tokenizer output must contain integer token IDs")
    return list(value)


def changed_token_positions(
    original_ids: Sequence[int], reference_ids: Sequence[int]
) -> list[int]:
    """Locate original-token spans removed or replaced in a reference prompt."""

    matcher = difflib.SequenceMatcher(
        a=list(original_ids), b=list(reference_ids), autojunk=False
    )
    positions: list[int] = []
    for tag, original_start, original_end, _, _ in matcher.get_opcodes():
        if tag in {"delete", "replace"}:
            positions.extend(range(original_start, original_end))
    return sorted(set(positions))


def masked_prompt_ids(
    original_ids: Sequence[int], reference_ids: Sequence[int], mask_token_id: int
) -> tuple[list[int], list[int]]:
    """Replace changed original tokens while preserving prompt length exactly."""

    positions = changed_token_positions(original_ids, reference_ids)
    if not positions:
        raise ValueError("anchor selection did not identify any prompt tokens")
    masked = list(original_ids)
    for index in positions:
        masked[index] = int(mask_token_id)
    if len(masked) != len(original_ids):
        raise AssertionError("masked prompt length changed")
    return masked, positions


def combine_logits(main_logits: Any, masked_logits: Any, strength: float) -> Any:
    """Apply SPA's first-order logit approximation.

    This deliberately has no probability modulation or model confidence gate;
    the preregistered first run uses the fixed-strength equation in the paper.
    """

    if not math.isfinite(strength):
        raise ValueError("anchoring strength must be finite")
    return strength * main_logits + (1.0 - strength) * masked_logits


def _tokenize_messages(
    tokenizer: Any,
    messages: Sequence[dict[str, Any]],
    chat_template_kwargs: dict[str, Any] | None = None,
) -> list[int]:
    kwargs = dict(chat_template_kwargs or {})
    tokenized = tokenizer.apply_chat_template(
        list(messages),
        tokenize=True,
        add_generation_prompt=True,
        **kwargs,
    )
    return normalize_token_ids(tokenized)


def resolve_mask_token_id(tokenizer: Any, mask_token: str) -> int:
    """Resolve a neutral token, preferring the explicitly configured token."""

    if mask_token:
        encoded = normalize_token_ids(
            tokenizer.encode(mask_token, add_special_tokens=False)
        )
        if len(encoded) == 1:
            return encoded[0]

    for attribute in ("pad_token_id", "unk_token_id"):
        token_id = getattr(tokenizer, attribute, None)
        if isinstance(token_id, int):
            return token_id
    raise ValueError(
        f"mask token {mask_token!r} is not one token and tokenizer has no pad/unk token"
    )


def prepare_prompt_pair(
    tokenizer: Any,
    messages: Sequence[dict[str, Any]],
    *,
    anchor_mode: str,
    mask_token_id: int,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> tuple[list[int], list[int], list[int]]:
    """Tokenize the original prompt and construct its position-preserving mask."""

    main_ids = _tokenize_messages(tokenizer, messages, chat_template_kwargs)
    reference_ids = _tokenize_messages(
        tokenizer,
        reference_messages(messages, anchor_mode),
        chat_template_kwargs,
    )
    auxiliary_ids, positions = masked_prompt_ids(
        main_ids, reference_ids, mask_token_id
    )
    return main_ids, auxiliary_ids, positions


def _eos_token_ids(tokenizer: Any) -> set[int]:
    values: set[int] = set()
    plural = getattr(tokenizer, "eos_token_ids", None)
    if plural is not None:
        values.update(int(item) for item in plural)
    singular = getattr(tokenizer, "eos_token_id", None)
    if isinstance(singular, int):
        values.add(singular)
    return values


class MLXAnchoredGenerator:
    """One-checkpoint direct/SPA generator with a shared batched KV cache."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        mask_token: str = DEFAULT_MASK_TOKEN,
        prefill_step_size: int = 2048,
    ) -> None:
        import mlx.core as mx
        from mlx_lm.models import cache as cache_module
        from mlx_lm.sample_utils import make_sampler
        from mlx_lm.utils import load

        self.mx = mx
        self.cache_module = cache_module
        self.make_sampler = make_sampler
        self.model_path = str(model_path)
        self.model, self.tokenizer = load(self.model_path)
        self.mask_token = mask_token
        self.mask_token_id = resolve_mask_token_id(self.tokenizer, mask_token)
        self.prefill_step_size = int(prefill_step_size)
        if self.prefill_step_size <= 0:
            raise ValueError("prefill step size must be positive")

    def _model_logits(self, token_batch: Any, prompt_cache: Any) -> Any:
        logits = self.model(token_batch, cache=prompt_cache)
        return logits[:, -1, :]

    def generate(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        mode: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        seed: int,
        strength: float = DEFAULT_STRENGTH,
        anchor_mode: str = DEFAULT_ANCHOR_MODE,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> GenerationResult:
        if mode not in SUPPORTED_DECODING_MODES:
            raise ValueError(f"unsupported decoding mode: {mode}")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if temperature < 0:
            raise ValueError("temperature cannot be negative")
        if not 0 < top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")

        spec = AnchoringSpec(
            enabled=mode == "spa",
            strength=float(strength),
            anchor_mode=anchor_mode,
            mask_token=self.mask_token,
        )
        if spec.enabled:
            main_ids, auxiliary_ids, positions = prepare_prompt_pair(
                self.tokenizer,
                messages,
                anchor_mode=spec.anchor_mode,
                mask_token_id=self.mask_token_id,
                chat_template_kwargs=chat_template_kwargs,
            )
            prompt_rows = [main_ids, auxiliary_ids]
        else:
            main_ids = _tokenize_messages(
                self.tokenizer, messages, chat_template_kwargs
            )
            positions = []
            prompt_rows = [main_ids]

        mx = self.mx
        mx.random.seed(int(seed))
        mx.reset_peak_memory()
        prompt_batch = mx.array(prompt_rows, dtype=mx.int32)
        prompt_cache = self.cache_module.make_prompt_cache(self.model)
        sampler = self.make_sampler(temp=float(temperature), top_p=float(top_p))
        eos_ids = _eos_token_ids(self.tokenizer)

        started = time.perf_counter()
        prompt_started = started
        offset = 0
        while prompt_batch.shape[1] - offset > 1:
            remaining = prompt_batch.shape[1] - offset - 1
            width = min(self.prefill_step_size, remaining)
            self._model_logits(
                prompt_batch[:, offset : offset + width], prompt_cache
            )
            mx.eval([entry.state for entry in prompt_cache])
            offset += width
            mx.clear_cache()

        logits = self._model_logits(prompt_batch[:, offset:], prompt_cache)
        mx.eval(logits)
        prompt_elapsed = time.perf_counter() - prompt_started
        generation_started = time.perf_counter()

        output_ids: list[int] = []
        finish_reason = "length"
        for _ in range(max_tokens):
            selected_logits = (
                combine_logits(logits[0], logits[1], spec.strength)
                if spec.enabled
                else logits[0]
            )
            logprobs = selected_logits - mx.logsumexp(
                selected_logits, keepdims=True
            )
            sampled = sampler(logprobs[None, :])
            mx.eval(sampled)
            token_id = int(sampled.item())
            if token_id in eos_ids:
                finish_reason = "stop"
                break
            output_ids.append(token_id)
            next_batch = mx.array(
                [[token_id] for _ in range(len(prompt_rows))], dtype=mx.int32
            )
            logits = self._model_logits(next_batch, prompt_cache)
            mx.eval(logits)

        generation_elapsed = time.perf_counter() - generation_started
        total_elapsed = time.perf_counter() - started
        text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        metadata = {
            "algorithm": "selective_prompt_anchoring" if spec.enabled else "direct",
            "paper": PAPER_URL if spec.enabled else None,
            "anchoring": asdict(spec),
            "mask_token_id": self.mask_token_id if spec.enabled else None,
            "masked_prompt_tokens": len(positions),
            "forward_batch_size": len(prompt_rows),
            "visible_prompt_tokens": len(main_ids),
            "visible_prompt_token_ids_sha256": hashlib.sha256(
                ",".join(map(str, main_ids)).encode("ascii")
            ).hexdigest(),
            "forward_prompt_tokens": len(main_ids) * len(prompt_rows),
            "prompt_tps": round(len(main_ids) / max(prompt_elapsed, 1e-9), 3),
            "generation_tps": round(
                len(output_ids) / max(generation_elapsed, 1e-9), 3
            ),
            "decoder_latency_ms": round(total_elapsed * 1000, 3),
            "peak_memory_gb": round(mx.get_peak_memory() / (1024**3), 4),
            "completion_token_ids_sha256": hashlib.sha256(
                ",".join(map(str, output_ids)).encode("ascii")
            ).hexdigest(),
            "llm_calls": 1,
        }
        mx.clear_cache()
        return GenerationResult(
            text=text,
            token_ids=output_ids,
            finish_reason=finish_reason,
            prompt_tokens=len(main_ids),
            completion_tokens=len(output_ids),
            metadata=metadata,
        )


def _request_mode(body: dict[str, Any]) -> str:
    explicit = body.get("decoding_mode")
    if explicit is not None:
        if explicit not in SUPPORTED_DECODING_MODES:
            raise ValueError(f"unsupported decoding_mode: {explicit!r}")
        return explicit
    model = str(body.get("model", ""))
    return "spa" if model.endswith("-spa") else "direct"


def create_app(
    decoder: MLXAnchoredGenerator,
    *,
    model_id: str,
    default_strength: float = DEFAULT_STRENGTH,
    default_anchor_mode: str = DEFAULT_ANCHOR_MODE,
) -> web.Application:
    lock = asyncio.Lock()
    direct_id = f"{model_id}-direct"
    spa_id = f"{model_id}-spa"

    async def models(_: web.Request) -> web.Response:
        now = int(time.time())
        return web.json_response(
            {
                "object": "list",
                "data": [
                    {"id": direct_id, "object": "model", "created": now},
                    {"id": spa_id, "object": "model", "created": now},
                ],
            }
        )

    async def health(_: web.Request) -> web.Response:
        return web.json_response(
            {
                "status": "ok",
                "service": "mlx-anchored-generation",
                "model_id": model_id,
                "model_path": decoder.model_path,
                "config": {
                    "decoding_modes": list(SUPPORTED_DECODING_MODES),
                    "anchor_modes": list(SUPPORTED_ANCHOR_MODES),
                    "default_strength": default_strength,
                    "default_anchor_mode": default_anchor_mode,
                    "mask_token": decoder.mask_token,
                    "mask_token_id": decoder.mask_token_id,
                    "hidden_tests_exposed": False,
                    "answer_candidates": 1,
                    "llm_calls": 1,
                },
            }
        )

    async def completions(request: web.Request) -> web.Response:
        try:
            body = await request.json()
            if body.get("stream", False):
                raise ValueError("streaming is not implemented in the research prototype")
            messages = body.get("messages")
            if not isinstance(messages, list) or not messages:
                raise ValueError("messages must be a non-empty list")
            if not all(isinstance(message, dict) for message in messages):
                raise ValueError("each message must be an object")
            mode = _request_mode(body)
            max_tokens = int(body.get("max_tokens", 1024))
            temperature = float(body.get("temperature", 0.0))
            top_p = float(body.get("top_p", 1.0))
            seed = int(body.get("seed", 0))
            strength = float(body.get("anchoring_strength", default_strength))
            anchor_mode = str(body.get("anchor_mode", default_anchor_mode))
            chat_template_kwargs = body.get("chat_template_kwargs")
            if chat_template_kwargs is not None and not isinstance(
                chat_template_kwargs, dict
            ):
                raise ValueError("chat_template_kwargs must be an object")
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            return web.json_response(
                {"error": {"message": str(exc), "type": "invalid_request_error"}},
                status=400,
            )

        try:
            async with lock:
                # MLX streams are thread-local. The research server serializes
                # requests, so generation must stay on the thread that loaded
                # the model instead of being handed to asyncio.to_thread().
                result = decoder.generate(
                    messages,
                    mode=mode,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    seed=seed,
                    strength=strength,
                    anchor_mode=anchor_mode,
                    chat_template_kwargs=chat_template_kwargs,
                )
        except Exception as exc:
            return web.json_response(
                {
                    "error": {
                        "message": f"{type(exc).__name__}: {exc}",
                        "type": "generation_error",
                    }
                },
                status=500,
            )

        response_model = spa_id if mode == "spa" else direct_id
        return web.json_response(
            {
                "id": f"anchored-{uuid.uuid4().hex}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": response_model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": result.text},
                        "finish_reason": result.finish_reason,
                    }
                ],
                "usage": {
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                    "total_tokens": result.prompt_tokens + result.completion_tokens,
                },
                "decoding_metadata": result.metadata,
            }
        )

    app = web.Application(client_max_size=4 * 1024 * 1024)
    app.router.add_get("/health", health)
    app.router.add_get("/v1/models", models)
    app.router.add_post("/v1/chat/completions", completions)
    return app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        default=os.environ.get("ANCHORED_MODEL_PATH", ""),
        help="local MLX checkpoint path",
    )
    parser.add_argument(
        "--model-id",
        default=os.environ.get("ANCHORED_MODEL_ID", "llama31-8b3bit"),
    )
    parser.add_argument("--host", default=os.environ.get("ANCHORED_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("ANCHORED_PORT", "8085")))
    parser.add_argument("--mask-token", default=DEFAULT_MASK_TOKEN)
    parser.add_argument("--strength", type=float, default=DEFAULT_STRENGTH)
    parser.add_argument(
        "--anchor-mode",
        choices=SUPPORTED_ANCHOR_MODES,
        default=DEFAULT_ANCHOR_MODE,
    )
    parser.add_argument("--prefill-step-size", type=int, default=2048)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.model_path:
        raise SystemExit("--model-path or ANCHORED_MODEL_PATH is required")
    decoder = MLXAnchoredGenerator(
        args.model_path,
        mask_token=args.mask_token,
        prefill_step_size=args.prefill_step_size,
    )
    app = create_app(
        decoder,
        model_id=args.model_id,
        default_strength=args.strength,
        default_anchor_mode=args.anchor_mode,
    )
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

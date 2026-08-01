from __future__ import annotations

import json
from argparse import Namespace

import pytest

from benchmarks.livecodebench_benchmark import (
    SAFE_TASK_KEYS,
    load_decoder_manifest,
    make_body,
    variant_order,
)
from benchmarks.paired_core import sha256_json


def _task(task_id: str = "2024-01-01-a") -> dict[str, object]:
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "public problem text"},
    ]
    return {
        "task_id": task_id,
        "contest_date": "2024-01-01",
        "platform": "leetcode",
        "difficulty": "easy",
        "prompt_messages": messages,
        "prompt_sha256": sha256_json(messages),
    }


def _manifest(task: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "decoder_input_contract": {"contains_hidden_tests": False},
        "tasks": [task],
    }


def test_livecodebench_decoder_manifest_rejects_extra_test_fields(tmp_path):
    task = _task()
    task["private_test_cases"] = []
    path = tmp_path / "unsafe.json"
    path.write_text(json.dumps(_manifest(task)), encoding="utf-8")

    with pytest.raises(ValueError, match="unexpected task fields"):
        load_decoder_manifest(path, 0)


def test_livecodebench_direct_and_spa_share_exact_public_messages(tmp_path):
    task = _task()
    path = tmp_path / "safe.json"
    path.write_text(json.dumps(_manifest(task)), encoding="utf-8")
    tasks, _ = load_decoder_manifest(path, 0)
    args = Namespace(
        model_id="llama31-8b3bit",
        strength=1.28,
        anchor_mode="natural_language",
        max_tokens=2000,
    )

    direct = make_body(args, tasks[0], "direct_greedy", 0)
    spa = make_body(args, tasks[0], "spa_greedy", 0)

    assert set(tasks[0]) == SAFE_TASK_KEYS
    assert direct["messages"] == spa["messages"]
    assert direct["decoding_mode"] == "direct"
    assert spa["decoding_mode"] == "spa"
    assert variant_order(str(tasks[0]["task_id"]), 0) == variant_order(
        str(tasks[0]["task_id"]), 0
    )

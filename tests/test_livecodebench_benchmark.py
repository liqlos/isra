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
from benchmarks.livecodebench_data import iter_filtered_records
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


def test_livecodebench_streaming_filter_uses_all_release_v6_source_files(tmp_path):
    for name, date in {
        "test.jsonl": "2023-12-31T00:00:00",
        "test2.jsonl": "2024-01-01T00:00:00",
        "test3.jsonl": "2024-02-01T00:00:00",
        "test4.jsonl": "2024-03-01T00:00:00",
        "test5.jsonl": "2024-04-01T00:00:00",
        "test6.jsonl": "2025-04-30T00:00:00",
    }.items():
        (tmp_path / name).write_text(
            json.dumps({"question_id": name, "contest_date": date}) + "\n",
            encoding="utf-8",
        )

    rows = list(
        iter_filtered_records(
            release_version="release_v6",
            start_date="2024-01-01",
            end_date="2025-04-30",
            snapshot_dir=tmp_path,
        )
    )

    assert [row["question_id"] for row in rows] == [
        "test2.jsonl",
        "test3.jsonl",
        "test4.jsonl",
        "test5.jsonl",
        "test6.jsonl",
    ]

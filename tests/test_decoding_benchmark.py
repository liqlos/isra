from __future__ import annotations

from argparse import Namespace

from benchmarks.decoding_benchmark import VARIANTS, make_body, variant_order


def test_decoding_variant_order_is_stable_and_complete():
    first = variant_order("HumanEval/20", 0)
    assert first == variant_order("HumanEval/20", 0)
    assert set(first) == set(VARIANTS)
    assert len(first) == len(VARIANTS)


def test_direct_and_spa_payloads_differ_only_in_declared_intervention():
    args = Namespace(
        model_id="llama31-8b3bit",
        strength=1.28,
        anchor_mode="natural_language",
        max_tokens=512,
    )
    problem = {"prompt": "def f(x):\n    pass\n"}
    direct = make_body(args, problem, "direct_greedy", 7)
    spa = make_body(args, problem, "spa_greedy", 7)

    assert direct["messages"] == spa["messages"]
    assert direct["temperature"] == spa["temperature"] == 0.0
    assert direct["top_p"] == spa["top_p"] == 1.0
    assert direct["max_tokens"] == spa["max_tokens"] == 512
    assert direct["seed"] == spa["seed"] == 7
    assert direct["decoding_mode"] == "direct"
    assert spa["decoding_mode"] == "spa"
    assert spa["anchoring_strength"] == 1.28

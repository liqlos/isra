from __future__ import annotations

import numpy as np
import pytest

from anchored_generation import (
    DEFAULT_MASK_TOKEN,
    changed_token_positions,
    combine_logits,
    masked_prompt_ids,
    prepare_prompt_pair,
    reference_messages,
    remove_natural_language_description,
    resolve_mask_token_id,
)


class CharacterTokenizer:
    """Tiny deterministic tokenizer with a stable chat framing for unit tests."""

    pad_token_id = None
    unk_token_id = None

    def encode(self, text, add_special_tokens=False):
        if text == DEFAULT_MASK_TOKEN:
            return [999]
        return [1000 + ord(character) for character in text]

    def apply_chat_template(
        self, messages, tokenize=True, add_generation_prompt=True, **kwargs
    ):
        tokens = [1]
        for message in messages:
            role = {"system": 2, "user": 3, "assistant": 4}[message["role"]]
            tokens.extend([role, 5])
            tokens.extend(self.encode(message["content"]))
            tokens.append(6)
        if add_generation_prompt:
            tokens.extend([4, 5])
        return tokens


def test_natural_language_reference_keeps_signature_and_public_examples():
    content = '''from typing import List

def f(xs: List[int]) -> int:
    """Return the largest even item, or zero when none exists.

    >>> f([1, 4, 3])
    4
    """
'''
    reference = remove_natural_language_description(content)
    assert "def f" in reference
    assert ">>> f([1, 4, 3])" in reference
    assert "largest even item" not in reference


def test_unstructured_prompt_falls_back_to_anchoring_complete_user_message():
    messages = [
        {"role": "system", "content": "Return code only."},
        {"role": "user", "content": "Write a robust parser."},
    ]
    reference = reference_messages(messages, "natural_language")
    assert reference[-1]["content"] == ""
    assert messages[-1]["content"] == "Write a robust parser."


def test_changed_positions_and_masking_preserve_exact_length():
    original = [1, 2, 10, 11, 12, 3, 4]
    reference = [1, 2, 3, 4]
    assert changed_token_positions(original, reference) == [2, 3, 4]
    masked, positions = masked_prompt_ids(original, reference, 99)
    assert masked == [1, 2, 99, 99, 99, 3, 4]
    assert positions == [2, 3, 4]
    assert len(masked) == len(original)


def test_masking_rejects_noop_anchor_selection():
    with pytest.raises(ValueError, match="did not identify"):
        masked_prompt_ids([1, 2, 3], [1, 2, 3], 99)


def test_prepare_prompt_pair_masks_only_diff_and_keeps_two_equal_rows():
    tokenizer = CharacterTokenizer()
    messages = [
        {"role": "system", "content": "Code only."},
        {
            "role": "user",
            "content": 'def f(x):\n    """Increment x.\n    >>> f(1)\n    2\n    """\n',
        },
    ]
    main, auxiliary, positions = prepare_prompt_pair(
        tokenizer,
        messages,
        anchor_mode="natural_language",
        mask_token_id=999,
    )
    assert len(main) == len(auxiliary)
    assert positions
    assert all(auxiliary[index] == 999 for index in positions)
    assert all(
        auxiliary[index] == token
        for index, token in enumerate(main)
        if index not in positions
    )

def test_mask_token_must_resolve_to_one_token():
    tokenizer = CharacterTokenizer()
    assert resolve_mask_token_id(tokenizer, DEFAULT_MASK_TOKEN) == 999
    with pytest.raises(ValueError, match="not one token"):
        resolve_mask_token_id(tokenizer, "not-special")


def test_spa_logits_equation_and_identity():
    main = np.array([2.0, -1.0, 0.5])
    masked = np.array([-2.0, 3.0, 0.25])
    np.testing.assert_array_equal(combine_logits(main, masked, 1.0), main)
    np.testing.assert_allclose(
        combine_logits(main, masked, 1.25),
        1.25 * main - 0.25 * masked,
    )

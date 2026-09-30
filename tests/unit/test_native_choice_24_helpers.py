"""Offline checks for the helpers of the 24-option live test (#288, #295).

The live test reads each option's control from the rendered prompt and the
vocabulary size from the router ``/v1/models`` entry. These tests run the
same helpers on a prompt rendered offline and on bodies shaped like the
router's ``/v1/models`` reply.
"""

from __future__ import annotations

import pytest

from tests.live.test_native_choice_24_live import (
    _CRITERIA,
    _QUESTION,
    _TARGET,
    _models_n_vocab,
    _prompt_controls,
)
from typevet.domain.field_instructions import render_field_instructions
from typevet.domain.judgment_normalize import normalize_choice

pytestmark = pytest.mark.unit

_LABELS = tuple(_CRITERIA)


def _rendered() -> str:
    decision = normalize_choice(_QUESTION, field_name="category")
    block = render_field_instructions(
        decision, choice_criteria=_CRITERIA, original_labels=_LABELS
    )
    return f"<|im_start|>user\n{block}<|im_end|>\n<|im_start|>assistant\n"


def test_prompt_controls_reads_each_control_from_the_rendered_prompt() -> None:
    controls = _prompt_controls(_rendered(), _LABELS)

    assert list(controls) == list(_LABELS)
    assert controls[_TARGET] == "K"
    assert controls["groceries"] == "0"
    assert controls["stationery_cards"] == "N"


def test_prompt_controls_follows_the_prompt_not_a_fixed_list() -> None:
    swapped = _rendered().replace("K → camping_gear", "Z → camping_gear")

    assert _prompt_controls(swapped, _LABELS)[_TARGET] == "Z"


def test_prompt_controls_reads_the_control_prefix_form() -> None:
    prompt = "Options:\nControl 0 → yes: agree\nControl 1 → no\n"

    assert _prompt_controls(prompt, ("yes", "no")) == {"yes": "0", "no": "1"}


@pytest.mark.parametrize(
    "prompt",
    ["Options:\n0 → yes\n", "Options:\n0 → yes\n1 → no\n2 → no\n"],
    ids=["missing-label", "label-twice"],
)
def test_prompt_controls_rejects_a_label_without_exactly_one_line(
    prompt: str,
) -> None:
    with pytest.raises(AssertionError, match="no"):
        _prompt_controls(prompt, ("yes", "no"))


_MODEL = "gemma-4-31b-24gib-kv11-decoder"


def _models(model: str = _MODEL, **meta: object) -> dict[str, object]:
    entry = {"vocab_type": 2, "n_ctx": 8192, "ftype": "Q2_K - Medium", **meta}
    return {
        "object": "list",
        "data": [
            {"id": "other-model", "meta": {"n_vocab": 32000}},
            {"id": model, "object": "model", "meta": entry},
        ],
    }


@pytest.mark.parametrize("n_vocab", [262144, 32000])
def test_models_n_vocab_reads_meta_of_the_matching_entry(n_vocab: int) -> None:
    assert _models_n_vocab(_models(n_vocab=n_vocab), _MODEL) == n_vocab


@pytest.mark.parametrize(
    "models",
    [
        _models(),
        _models(n_vocab=0),
        _models(n_vocab=True),
        _models(model="gemma-other", n_vocab=262144),
        {"data": "not-a-list"},
    ],
    ids=["absent", "zero", "bool", "no-entry", "bad-data"],
)
def test_models_n_vocab_fails_clearly_without_a_vocabulary_size(
    models: dict[str, object],
) -> None:
    with pytest.raises(AssertionError, match=r"n_vocab|entries"):
        _models_n_vocab(models, _MODEL)

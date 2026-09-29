"""Offline Gemma template / binding contract fixtures (#118)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

from typevet.adapters.outbound.gemma.served_template import (
    CHATML_ASSISTANT_HEADER,
    CHATML_IM_END,
    CHATML_IM_START,
    GEMMA4_MODEL_TURN_HEADER,
    GEMMA4_TURN_CLOSE,
    GEMMA4_TURN_OPEN,
    ServedTemplateClass,
)
from typevet.domain.errors import GemmaTemplateError


def degraded_chatml_rendered(user_text: str) -> str:
    return (
        f"{CHATML_IM_START}user\n{user_text}{CHATML_IM_END}\n{CHATML_ASSISTANT_HEADER}"
    )


def native_gemma4_rendered(user_text: str) -> str:
    return (
        f"{GEMMA4_TURN_OPEN}user\n{user_text}{GEMMA4_TURN_CLOSE}\n"
        f"{GEMMA4_MODEL_TURN_HEADER}"
    )


GO_EMOTIONS_USER: Final[str] = (
    "Given the user message, which single emotion label best applies?"
)
PINNED_DEGRADED_PROMPT: Final[str] = degraded_chatml_rendered(GO_EMOTIONS_USER)

PINNED_LABEL_TOKEN_IDS: Final[dict[str, tuple[int, ...]]] = {
    "anger": (4751,),
    "joy": (3672,),
    "neutral": (45258,),
    "admiration": (553, 18570, 567),
}


def pinned_tokenize_with_special(text: str) -> list[int]:
    body = [ord(c) for c in text if c != "\n"]
    return [2, *body]


def pinned_tokenize_content(text: str) -> list[int]:
    if text in PINNED_LABEL_TOKEN_IDS:
        return list(PINNED_LABEL_TOKEN_IDS[text])
    return [len(text)]


def get_fixtures() -> list[dict[str, Any]]:
    degraded = PINNED_DEGRADED_PROMPT
    native = native_gemma4_rendered("Pick one label.")
    return [
        {
            "name": "degraded_anchor_byte_and_token_boundary",
            "rendered": degraded,
            "tokenize_special": pinned_tokenize_with_special,
            "tokenize_content": pinned_tokenize_content,
            "expect": {
                "kind": "anchor",
                "template_class": ServedTemplateClass.DEGRADED_CHATML.value,
                "prefix_token_count": len(pinned_tokenize_with_special(degraded)),
                "byte_length": len(degraded.encode("utf-8")),
            },
        },
        {
            "name": "native_anchor_no_thinking",
            "rendered": native,
            "tokenize_special": pinned_tokenize_with_special,
            "tokenize_content": pinned_tokenize_content,
            "expect": {
                "kind": "anchor",
                "template_class": ServedTemplateClass.NATIVE_GEMMA4_TURN.value,
            },
        },
        {
            "name": "bind_three_pinned_enum_strings",
            "labels": ("anger", "joy", "neutral"),
            "tokenize_content": pinned_tokenize_content,
            "expect": {
                "kind": "bind",
                "token_ids": {
                    "anger": PINNED_LABEL_TOKEN_IDS["anger"],
                    "joy": PINNED_LABEL_TOKEN_IDS["joy"],
                    "neutral": PINNED_LABEL_TOKEN_IDS["neutral"],
                },
            },
        },
        {
            "name": "whitespace_enum_negative",
            "labels": (" anger",),
            "tokenize_content": pinned_tokenize_content,
            "expect": {"kind": "error", "exc_type": "GemmaTemplateError"},
        },
        {
            "name": "split_control_label_negative",
            "labels": (f"bad{CHATML_IM_START}assistant",),
            "tokenize_content": pinned_tokenize_content,
            "expect": {"kind": "error", "exc_type": "GemmaTemplateError"},
        },
        {
            "name": "duplicated_bos_negative",
            "rendered": degraded,
            "tokenize_special": lambda _text: [2, 2, 9],
            "expect": {"kind": "error", "exc_type": "GemmaTemplateError"},
        },
        {
            "name": "missing_assistant_boundary",
            "rendered": f"{CHATML_IM_START}user\nHi{CHATML_IM_END}\n",
            "tokenize_special": pinned_tokenize_with_special,
            "expect": {"kind": "error", "exc_type": "GemmaTemplateError"},
        },
        {
            "name": "wrong_template_mixed_markers",
            "rendered": degraded + GEMMA4_TURN_OPEN,
            "tokenize_special": pinned_tokenize_with_special,
            "expect": {"kind": "error", "exc_type": "GemmaTemplateError"},
        },
        {
            "name": "premature_termination_partial_label",
            "completion": "admi",
            "expected_label": "admiration",
            "template_class": ServedTemplateClass.DEGRADED_CHATML.value,
            "tokenize_content": pinned_tokenize_content,
            "expect": {"kind": "termination", "result": "premature"},
        },
        {
            "name": "complete_enum_termination",
            "completion": "anger",
            "expected_label": "anger",
            "template_class": ServedTemplateClass.DEGRADED_CHATML.value,
            "tokenize_content": pinned_tokenize_content,
            "expect": {"kind": "termination", "result": "complete"},
        },
    ]


def exc_type_from_name(name: str) -> type[Exception]:
    mapping: Mapping[str, type[Exception]] = {
        "GemmaTemplateError": GemmaTemplateError,
    }
    try:
        return mapping[name]
    except KeyError as exc:
        raise ValueError(f"unknown exc_type: {name}") from exc

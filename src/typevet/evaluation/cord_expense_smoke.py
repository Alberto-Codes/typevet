"""CORD expense live-smoke helpers: attachment floors and Gemma 4 capability ([#185][i185]).

The direct three-label smoke gates on prompt token growth, not on semantic
quality. Measured image cost differs between Gemma 3 and Gemma 4 multimodal
ids on the same router, so the attachment floor is model-specific. A pinned
Gemma 4 receipt records vision, the native turn family and combined outcomes
for offline regression.

Examples:
    ```python
    from typevet.evaluation.cord_expense_smoke import (
        assert_cord_expense_attachment,
        cord_combined_attachment_floor,
    )

    floor = cord_combined_attachment_floor("gemma-4-31b-kv9-q4km-mm")
    assert floor > 200
    ```

See Also:
    - [typevet.evaluation.cord_semantic_acceptance][]: #161 revision 1 floors
    - [tests.live.test_cord_expense_smoke_live][]: opt-in live smoke

[i185]: https://github.com/Alberto-Codes/typevet/issues/185
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

# One attached receipt image on the CORD smoke router, measured on live runs.
_GEMMA3_IMAGE_PROMPT_TOKENS: Final[int] = 256
_GEMMA4_IMAGE_PROMPT_TOKENS: Final[int] = 245
# A silently dropped image adds marker text only (#155 recipe).
_MARKER_ONLY_PROMPT_TOKENS: Final[int] = 31

GEMMA4_DIRECT_RECEIPT_MODEL: Final[str] = "gemma-4-31b-kv9-q4km-mm"
GEMMA4_NATIVE_TURN: Final[str] = "native_gemma4_turn"


def measured_image_prompt_tokens(model_id: str) -> int:
    """Return the measured single-image prompt cost for one multimodal model id.

    Args:
        model_id: Router model id from ``TYPEVET_LLAMA__MULTIMODAL_MODEL``.

    Returns:
        Prompt tokens one CORD receipt image adds on a native-turn prefix.

    Raises:
        ValueError: When ``model_id`` is not a supported Gemma 3 or Gemma 4 id.
    """
    if model_id.startswith("gemma-4"):
        return _GEMMA4_IMAGE_PROMPT_TOKENS
    if model_id.startswith("gemma-3"):
        return _GEMMA3_IMAGE_PROMPT_TOKENS
    msg = (
        "CORD expense attachment floors are defined only for gemma-3 and "
        f"gemma-4 multimodal ids, not {model_id!r}"
    )
    raise ValueError(msg)


def cord_combined_attachment_floor(model_id: str) -> int:
    """Minimum ``combined`` minus ``text_only`` token gap that proves attachment.

    Args:
        model_id: Router model id from ``TYPEVET_LLAMA__MULTIMODAL_MODEL``.

    Returns:
        Token gap floor for the same claim across the two modalities.
    """
    return measured_image_prompt_tokens(model_id) - _MARKER_ONLY_PROMPT_TOKENS


def cord_image_only_attachment_floor(model_id: str) -> int:
    """Minimum absolute prompt tokens for an ``image_only`` row.

    Args:
        model_id: Router model id from ``TYPEVET_LLAMA__MULTIMODAL_MODEL``.

    Returns:
        Absolute ``tokens_evaluated`` floor for a receipt-only request.
    """
    return measured_image_prompt_tokens(model_id) - _MARKER_ONLY_PROMPT_TOKENS


def validate_gemma4_smoke_capability(receipt: Mapping[str, Any]) -> None:
    """Assert one saved receipt records Gemma 4 vision on the native turn path.

    Args:
        receipt: Live or vendored CORD expense smoke receipt mapping.

    Raises:
        ValueError: When capability fields disagree with the Gemma 4 contract.
    """
    if receipt.get("vision") is not True:
        msg = "receipt must declare vision=true for Gemma 4 capability"
        raise ValueError(msg)
    served = receipt.get("served_template")
    if served != GEMMA4_NATIVE_TURN:
        msg = f"receipt served_template must be {GEMMA4_NATIVE_TURN!r}, got {served!r}"
        raise ValueError(msg)
    model = receipt.get("model")
    if not isinstance(model, str) or not model.startswith("gemma-4"):
        msg = "receipt model must be a gemma-4 multimodal id"
        raise ValueError(msg)


def assert_cord_expense_attachment(
    *,
    model_id: str,
    text_only: Mapping[str, Mapping[str, object]],
    image_only: Mapping[str, Mapping[str, object]],
    combined: Mapping[str, Mapping[str, object]],
    claim_ids: tuple[str, ...],
    receipt_ids: tuple[str, ...],
) -> None:
    """Fail loud when prompt token counts show a silently dropped receipt image.

    Args:
        model_id: Router model id used for the smoke run.
        text_only: ``text_only`` rows keyed by claim id.
        image_only: ``image_only`` rows keyed by receipt id.
        combined: ``combined`` rows keyed by claim id.
        claim_ids: Claim ids to compare across ``text_only`` and ``combined``.
        receipt_ids: Receipt ids that must meet the ``image_only`` floor.

    Raises:
        TypeError: When the router omits integer token counts.
        ValueError: When any row falls below the model-specific floor.
    """
    combined_floor = cord_combined_attachment_floor(model_id)
    image_floor = cord_image_only_attachment_floor(model_id)
    for claim_id in claim_ids:
        text_row = text_only[claim_id]
        combined_row = combined[claim_id]
        text_tokens = text_row["tokens_evaluated"]
        image_tokens = combined_row["tokens_evaluated"]
        if not isinstance(text_tokens, int) or not isinstance(image_tokens, int):
            msg = "router did not report integer prompt tokens"
            raise TypeError(msg)
        gap = image_tokens - text_tokens
        if gap < combined_floor:
            msg = (
                f"{claim_id} combined evaluated {image_tokens} prompt tokens "
                f"against a {text_tokens}-token text baseline (gap {gap}); "
                f"the image was not attached"
            )
            raise ValueError(msg)
    for receipt_id in receipt_ids:
        row = image_only[receipt_id]
        tokens = row["tokens_evaluated"]
        if not isinstance(tokens, int):
            msg = "router did not report integer prompt tokens"
            raise TypeError(msg)
        if tokens < image_floor:
            msg = (
                f"{receipt_id} image_only evaluated {tokens} prompt tokens; "
                "the image was not attached"
            )
            raise ValueError(msg)

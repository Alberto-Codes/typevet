"""CORD expense live-smoke helpers: attachment floors and Gemma 4 capability ([#185][i185]).

The direct three-label smoke gates on prompt token growth, not on semantic
quality. Measured image cost differs between Gemma 3 and Gemma 4 multimodal
ids on the same router, so the attachment floor is model-specific. A pinned
Gemma 4 receipt records vision, the native turn family and combined outcomes
for offline regression.

Examples:
    ```python
    from typevet_evals.cord.expense_smoke import (
        assert_cord_expense_attachment,
        assert_cord_expense_live_smoke_gate,
        cord_combined_attachment_floor,
    )

    floor = cord_combined_attachment_floor("gemma-4-31b-kv9-q4km-mm")
    assert floor > 200
    ```

See Also:
    - [typevet_evals.cord.semantic_acceptance][]: #161 revision 1 floors
    - [evals.tests.live.test_cord_expense_smoke_live][]: opt-in live smoke

[i185]: https://github.com/Alberto-Codes/typevet/issues/185
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

# One attached receipt image on the CORD smoke router, measured on live runs.
_GEMMA3_IMAGE_PROMPT_TOKENS: Final[int] = 256
_GEMMA4_IMAGE_PROMPT_TOKENS: Final[int] = 245
# A silently dropped image adds marker text only (#155 recipe).
_MARKER_ONLY_PROMPT_TOKENS: Final[int] = 31

GEMMA3_DIRECT_RECEIPT_MODEL: Final[str] = "gemma-3-4b-it-q4km-mm"
GEMMA4_DIRECT_RECEIPT_MODEL: Final[str] = "gemma-4-31b-kv9-q4km-mm"
GEMMA4_NATIVE_TURN: Final[str] = "native_gemma4_turn"
GEMMA3_NATIVE_TURN: Final[str] = "native_gemma3_turn"


@dataclass(frozen=True, slots=True)
class CordExpenseAttachmentProfile:
    """Verified attachment calibration for one router model and served family.

    Attributes:
        model_id (str): Router multimodal model id.
        served_template (str): Native turn family from ``/apply-template``.
        measured_image_prompt_tokens (int): Prompt tokens one receipt image adds
            on the combined arm (text baseline subtracted).
        image_only_state_omission_tokens (int): Prompt tokens for the fixed
            ``image_only`` claim text with no image (same prompt, omission).

    Examples:
        ```python
        from typevet_evals.cord.expense_smoke import (
            GEMMA4_DIRECT_RECEIPT_MODEL,
            resolve_cord_expense_attachment_profile,
        )

        profile = resolve_cord_expense_attachment_profile(
            GEMMA4_DIRECT_RECEIPT_MODEL,
            "native_gemma4_turn",
        )
        assert profile.measured_image_prompt_tokens == 245
        ```
    """

    model_id: str
    served_template: str
    measured_image_prompt_tokens: int
    image_only_state_omission_tokens: int


_VERIFIED_ATTACHMENT_PROFILES: Final[
    dict[tuple[str, str], CordExpenseAttachmentProfile]
] = {
    (
        GEMMA3_DIRECT_RECEIPT_MODEL,
        GEMMA3_NATIVE_TURN,
    ): CordExpenseAttachmentProfile(
        model_id=GEMMA3_DIRECT_RECEIPT_MODEL,
        served_template=GEMMA3_NATIVE_TURN,
        measured_image_prompt_tokens=_GEMMA3_IMAGE_PROMPT_TOKENS,
        image_only_state_omission_tokens=218,
    ),
    (
        GEMMA4_DIRECT_RECEIPT_MODEL,
        GEMMA4_NATIVE_TURN,
    ): CordExpenseAttachmentProfile(
        model_id=GEMMA4_DIRECT_RECEIPT_MODEL,
        served_template=GEMMA4_NATIVE_TURN,
        measured_image_prompt_tokens=_GEMMA4_IMAGE_PROMPT_TOKENS,
        image_only_state_omission_tokens=218,
    ),
}


class CordExpenseLiveSmokeGateError(ValueError):
    """The live CORD smoke harness must fail before scoring.

    Examples:
        ```python
        from typevet_evals.cord.expense_smoke import (
            CordExpenseLiveSmokeGateError,
        )

        assert issubclass(CordExpenseLiveSmokeGateError, ValueError)
        ```
    """


def resolve_cord_expense_attachment_profile(
    model_id: str,
    served_template: str,
) -> CordExpenseAttachmentProfile:
    """Return the verified attachment profile for one model and served family.

    Args:
        model_id: Router model id from ``TYPEVET_LLAMA__MULTIMODAL_MODEL``.
        served_template: ``ServedTemplateClass`` value from ``/apply-template``.

    Returns:
        Measured attachment calibration bound to that configuration.

    Raises:
        ValueError: When the pair is not a verified CORD smoke configuration.
    """
    profile = _VERIFIED_ATTACHMENT_PROFILES.get((model_id, served_template))
    if profile is None:
        msg = (
            "CORD expense attachment evidence is defined only for verified "
            f"gemma-3 or gemma-4 native-turn configurations, not "
            f"({model_id!r}, {served_template!r})"
        )
        raise ValueError(msg)
    return profile


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


def cord_image_only_attachment_gap_floor(model_id: str) -> int:
    """Minimum ``image_only`` minus omission token gap that proves attachment.

    Args:
        model_id: Router model id from ``TYPEVET_LLAMA__MULTIMODAL_MODEL``.

    Returns:
        Token gap floor for the fixed ``image_only`` prompt with media present.
    """
    return cord_combined_attachment_floor(model_id)


def assert_cord_expense_live_smoke_gate(
    *,
    vision: bool,
    served_template: str,
    model_id: str,
) -> CordExpenseAttachmentProfile:
    """Fail before scoring when the router cannot run the CORD smoke contract.

    Args:
        vision: Whether ``GET /props`` reports image input for the model.
        served_template: Served template family from ``/apply-template``.
        model_id: Multimodal model id under test.

    Returns:
        Verified attachment profile when native Gemma 3 or Gemma 4 may score.

    Raises:
        CordExpenseLiveSmokeGateError: When vision or the served family blocks
            the smoke before any scoring call.
        ValueError: When ``model_id`` and ``served_template`` are not verified.
    """
    if not vision:
        msg = (
            f"{model_id} reports text-only input modalities; "
            "a text-only router cannot prove receipt reading"
        )
        raise CordExpenseLiveSmokeGateError(msg)
    if served_template not in (GEMMA3_NATIVE_TURN, GEMMA4_NATIVE_TURN):
        msg = (
            "CORD expense smoke requires a native Gemma 3 or Gemma 4 turn, "
            f"not {served_template!r}"
        )
        raise CordExpenseLiveSmokeGateError(msg)
    if served_template == GEMMA4_NATIVE_TURN and not model_id.startswith("gemma-4"):
        msg = "native Gemma 4 turn requires a gemma-4 multimodal model id"
        raise CordExpenseLiveSmokeGateError(msg)
    if served_template == GEMMA3_NATIVE_TURN and not model_id.startswith("gemma-3"):
        msg = "native Gemma 3 turn requires a gemma-3 multimodal model id"
        raise CordExpenseLiveSmokeGateError(msg)
    return resolve_cord_expense_attachment_profile(model_id, served_template)


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
    profile: CordExpenseAttachmentProfile,
    text_only: Mapping[str, Mapping[str, object]],
    image_only: Mapping[str, Mapping[str, object]],
    combined: Mapping[str, Mapping[str, object]],
    claim_ids: tuple[str, ...],
    receipt_ids: tuple[str, ...],
    image_only_omission_tokens: int,
) -> None:
    """Fail loud when prompt token counts show a silently dropped receipt image.

    Args:
        profile: Verified model and served-template attachment calibration.
        text_only: ``text_only`` rows keyed by claim id.
        image_only: ``image_only`` rows keyed by receipt id.
        combined: ``combined`` rows keyed by claim id.
        claim_ids: Claim ids to compare across ``text_only`` and ``combined``.
        receipt_ids: Receipt ids that must meet the ``image_only`` gap floor.
        image_only_omission_tokens: Prompt tokens for the ``image_only`` claim
            text with no image on the same native turn family.

    Raises:
        TypeError: When the router omits integer token counts.
        ValueError: When any row falls below the model-specific floor or the
            profile does not match the receipt configuration.
    """
    if profile.model_id.startswith("gemma-4"):
        if profile.served_template != GEMMA4_NATIVE_TURN:
            msg = "attachment profile served_template does not match Gemma 4"
            raise ValueError(msg)
    elif profile.model_id.startswith("gemma-3"):
        if profile.served_template != GEMMA3_NATIVE_TURN:
            msg = "attachment profile served_template does not match Gemma 3"
            raise ValueError(msg)
    else:
        msg = f"unsupported attachment profile model id {profile.model_id!r}"
        raise ValueError(msg)

    combined_floor = profile.measured_image_prompt_tokens - _MARKER_ONLY_PROMPT_TOKENS
    image_gap_floor = combined_floor
    if not isinstance(image_only_omission_tokens, int):
        msg = "image_only omission control must report integer prompt tokens"
        raise TypeError(msg)
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
        gap = tokens - image_only_omission_tokens
        if gap < image_gap_floor:
            msg = (
                f"{receipt_id} image_only evaluated {tokens} prompt tokens "
                f"against a {image_only_omission_tokens}-token omission baseline "
                f"(gap {gap}); the image was not attached"
            )
            raise ValueError(msg)

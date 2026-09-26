"""Gemma answer-prefix anchors and enum token binding (no HTTP).

Examples:
    ```python
    from typevet.gemma_answer_binding import ThinkingDisposition

    assert ThinkingDisposition.NO_THINKING.value == "no_thinking"
    ```

See Also:
    - [typevet.gemma_served_template][]: Template gate classification
    - [typevet.domain.candidate_scoring_request][]: Scoring request types
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from typevet.domain.candidate_scoring_request import CandidateTokenSpec
from typevet.domain.errors import GemmaTemplateError
from typevet.gemma_served_template import (
    CHATML_ASSISTANT_HEADER,
    GEMMA4_MODEL_TURN_HEADER,
    GEMMA4_NO_THINKING_PREFILL,
    GEMMA4_THINK_TRIGGER,
    ServedTemplateClass,
    classify_served_template,
    label_embeds_control_fragment,
    stop_markers_for,
)

DEFAULT_BOS_TOKEN_ID: Final[int] = 2
TokenizeFn = Callable[[str], Sequence[int]]


class ThinkingDisposition(StrEnum):
    """Whether thinking-channel prefill is required at the answer anchor.

    Attributes:
        NO_THINKING (ThinkingDisposition): M1 default; no thinking-on path.

    Examples:
        ```python
        assert ThinkingDisposition.NO_THINKING.value == "no_thinking"
        ```
    """

    NO_THINKING = "no_thinking"


@dataclass(frozen=True, slots=True)
class AnswerAnchor:
    """Byte and token boundary immediately before the first candidate token.

    Attributes:
        template_class (ServedTemplateClass): Gate classification for the prefix.
        prefix (str): Rendered prompt up to the answer boundary.
        prefix_token_count (int): Token count of ``prefix`` with specials.
        byte_length (int): UTF-8 byte length of ``prefix``.

    Examples:
        ```python
        from typevet.gemma_served_template import ServedTemplateClass

        AnswerAnchor(
            template_class=ServedTemplateClass.DEGRADED_CHATML,
            prefix="x",
            prefix_token_count=1,
            byte_length=1,
        )
        ```
    """

    template_class: ServedTemplateClass
    prefix: str
    prefix_token_count: int
    byte_length: int


def resolve_answer_anchor(
    rendered_prompt: str,
    *,
    tokenize_with_special: TokenizeFn,
    thinking: ThinkingDisposition = ThinkingDisposition.NO_THINKING,
    bos_token_id: int = DEFAULT_BOS_TOKEN_ID,
) -> AnswerAnchor:
    """Validate and locate the pre-candidate answer anchor.

    Args:
        rendered_prompt: Full ``/apply-template`` output.
        tokenize_with_special: Tokenizer that preserves special tokens.
        thinking: Thinking disposition (M1: ``NO_THINKING`` only).
        bos_token_id: Expected single BOS id when present.

    Returns:
        Validated ``AnswerAnchor``.

    Raises:
        GemmaTemplateError: Unsupported template, bad boundary, or thinking mismatch.
    """
    template_class = classify_served_template(rendered_prompt)
    if template_class is ServedTemplateClass.UNSUPPORTED:
        msg = (
            "unsupported served template family; gate on /apply-template "
            f"(class={template_class.value})"
        )
        raise GemmaTemplateError(msg)

    prefix = _validated_prefix(rendered_prompt, template_class)
    _assert_answer_boundary(prefix, template_class)
    _assert_no_thinking_path(prefix, template_class, thinking)
    token_ids = list(tokenize_with_special(prefix))
    _assert_single_bos(
        token_ids, bos_token_id=bos_token_id, template_class=template_class
    )
    return AnswerAnchor(
        template_class=template_class,
        prefix=prefix,
        prefix_token_count=len(token_ids),
        byte_length=len(prefix.encode("utf-8")),
    )


def bind_enum_label(
    label: str,
    *,
    tokenize_content: TokenizeFn,
) -> tuple[int, ...]:
    """Bind one enum string to exact token ids (no surrounding whitespace).

    Args:
        label: Exact enum string.
        tokenize_content: Tokenizer for content tokens.

    Returns:
        Non-empty token-id tuple for ``label``.

    Raises:
        GemmaTemplateError: Whitespace, empty, control fragment, or empty tokenize.
    """
    if label != label.strip():
        msg = f"enum label must not have leading or trailing whitespace: {label!r}"
        raise GemmaTemplateError(msg)
    if not label:
        msg = "enum label must be non-empty"
        raise GemmaTemplateError(msg)
    if label_embeds_control_fragment(label):
        msg = f"enum label embeds a reserved control fragment: {label!r}"
        raise GemmaTemplateError(msg)
    token_ids = tuple(int(t) for t in tokenize_content(label))
    if not token_ids:
        msg = f"enum label produced no tokens: {label!r}"
        raise GemmaTemplateError(msg)
    return token_ids


def bind_enum_labels(
    labels: Sequence[str],
    *,
    tokenize_content: TokenizeFn,
) -> tuple[CandidateTokenSpec, ...]:
    """Bind ordered enum labels to ``CandidateTokenSpec`` rows.

    Args:
        labels: Ordered candidate enum strings.
        tokenize_content: Tokenizer for content tokens.

    Returns:
        One ``CandidateTokenSpec`` per label, in order.

    Raises:
        GemmaTemplateError: Propagated from ``bind_enum_label``.
    """
    specs: list[CandidateTokenSpec] = []
    for label in labels:
        token_ids = bind_enum_label(label, tokenize_content=tokenize_content)
        specs.append(CandidateTokenSpec(label=label, token_ids=token_ids))
    return tuple(specs)


def termination_kind(
    assistant_completion: str,
    *,
    template_class: ServedTemplateClass,
    expected_label: str,
    tokenize_content: TokenizeFn,
) -> str:
    """Classify assistant completion termination for contract fixtures.

    Args:
        assistant_completion: Model text after the answer anchor.
        template_class: Served template family.
        expected_label: Gold enum string.
        tokenize_content: Tokenizer for content tokens.

    Returns:
        ``complete``, ``stopped_at_marker``, or ``premature``.
    """
    if assistant_completion == expected_label:
        return "complete"
    markers = stop_markers_for(template_class)
    for marker in markers:
        if assistant_completion.endswith(marker):
            return "stopped_at_marker"
    bound_len = len(bind_enum_label(expected_label, tokenize_content=tokenize_content))
    if (
        bound_len > 1
        and assistant_completion
        and expected_label.startswith(assistant_completion)
    ):
        return "premature"
    if assistant_completion and assistant_completion not in expected_label:
        return "premature"
    return "premature"


def _validated_prefix(rendered: str, template_class: ServedTemplateClass) -> str:
    if template_class is ServedTemplateClass.DEGRADED_CHATML:
        if not rendered.endswith(CHATML_ASSISTANT_HEADER):
            msg = (
                "missing ChatML assistant answer boundary "
                f"(expected suffix {CHATML_ASSISTANT_HEADER!r})"
            )
            raise GemmaTemplateError(msg)
        return rendered
    if not rendered.endswith(GEMMA4_MODEL_TURN_HEADER):
        msg = (
            "missing Gemma4 model answer boundary "
            f"(expected suffix {GEMMA4_MODEL_TURN_HEADER!r})"
        )
        raise GemmaTemplateError(msg)
    return rendered


def _assert_answer_boundary(prefix: str, template_class: ServedTemplateClass) -> None:
    if template_class is ServedTemplateClass.DEGRADED_CHATML:
        if CHATML_ASSISTANT_HEADER not in prefix:
            raise GemmaTemplateError("missing ChatML assistant answer boundary")
        return
    if GEMMA4_MODEL_TURN_HEADER not in prefix:
        raise GemmaTemplateError("missing Gemma4 model answer boundary")


def _assert_no_thinking_path(
    prefix: str,
    template_class: ServedTemplateClass,
    thinking: ThinkingDisposition,
) -> None:
    if thinking is not ThinkingDisposition.NO_THINKING:
        msg = "only no_thinking disposition is implemented for M1"
        raise GemmaTemplateError(msg)
    if GEMMA4_THINK_TRIGGER in prefix:
        raise GemmaTemplateError("no_thinking path rejects <|think|> in prefix")
    if template_class is ServedTemplateClass.NATIVE_GEMMA4_TURN:
        tail = prefix.split(GEMMA4_MODEL_TURN_HEADER, maxsplit=1)[-1]
        if tail and tail != GEMMA4_NO_THINKING_PREFILL:
            raise GemmaTemplateError(
                "no_thinking native prefix has unexpected content after model turn"
            )


def _assert_single_bos(
    token_ids: Sequence[int],
    *,
    bos_token_id: int,
    template_class: ServedTemplateClass,
) -> None:
    count = sum(1 for tid in token_ids if tid == bos_token_id)
    if count > 1:
        msg = (
            f"duplicated BOS token id {bos_token_id} "
            f"(count={count}, template={template_class.value})"
        )
        raise GemmaTemplateError(msg)

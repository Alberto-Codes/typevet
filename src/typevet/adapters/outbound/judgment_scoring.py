"""Sync JudgmentPort adapter over CandidateScoringPort (#125).

A native served family wraps every prefix, with or without media; Gemma 3 and
Gemma 4 turns both qualify (#171, #179). Without one, text-only prefixes use
degraded ChatML and media fails closed (#157). An injected ``ModelFramingPort``
replaces that served-template choice for every prefix (#174); it receives the
user text that the ``context_template`` rendered (#373). A framing prefix
that ends in a Gemma 4 model turn without the no-thinking prefill is refused
before scoring (#354). An optional ``off_option_threshold`` flags each answer
whose off-option mass is above it and keeps one ``OffOptionReceipt`` per
answer on the response (#353). Optional ``option_block`` and
``context_template`` templates change the words around the control lines and
the context; the control mapping and the prefill stay the same, and
``text_parts`` pins both (#364). A media marker in the state, the
instructions or the criteria is neutralized before the adapter adds one real
marker per image, so caller text cannot bind or demand an image (#433).

Examples:
    ```python
    from typevet.domain.judgment_questions import Noul
    from typevet.adapters.outbound.judgment_scoring import ScoringJudgmentAdapter
    from typevet.testing import ScriptedScoringFake

    fake = ScriptedScoringFake(logprobs={"True": -0.2, "False": -1.0})
    port = ScoringJudgmentAdapter(fake, tokenize_content=lambda s: (ord(s[0]),))
    port.judge("text", {"q": Noul()}, "fake")
    ```

See Also:
    - [typevet.ports.judgment][]: JudgmentPort protocol
    - [typevet.domain.judgment_normalize][]: Normalize and control bind
    - [typevet.domain.media][]: Images the keyword-only ``media`` argument takes
    - [typevet.ports.framing][]: ModelFramingPort protocol
    - [typevet.domain.text_parts][]: Option block and context template rules
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any, TypedDict, Unpack

from typevet.adapters.outbound.gemma import (
    ServedTemplateClass,
    require_no_thinking_prefill,
)
from typevet.adapters.outbound.gemma.scoring_prefix import compose_served_prefix
from typevet.domain.candidate_scoring_request import CandidateTokenSpec
from typevet.domain.decision_execute import (
    CategoricalExecutionResult,
    apply_off_option_threshold,
    check_off_option_threshold,
    execute_categorical_decision,
)
from typevet.domain.decisions import Decision
from typevet.domain.errors import DecisionExecutionError, JudgmentValidationError
from typevet.domain.field_instructions import render_field_instructions
from typevet.domain.judgment_answers import (
    Answer,
    ChoiceAnswer,
    NoulAnswer,
    ScoreAnswer,
)
from typevet.domain.judgment_normalize import (
    bind_control_candidates,
    judgment_original_labels,
    normalize_question,
)
from typevet.domain.judgment_questions import Choice, Noul, Question, Score
from typevet.domain.judgment_response import (
    JudgmentResponse,
    OffOptionReceipt,
    TokenUsage,
)
from typevet.domain.media import (
    MEDIA_MARKER,
    ImageInput,
    count_media_markers,
    neutralize_media_markers,
)
from typevet.domain.text_parts import TextParts, render_context
from typevet.ports.framing import ModelFramingPort
from typevet.ports.scoring import CandidateScoringPort


def _execute_label(choice: object) -> str:
    if isinstance(choice, bool):
        return "True" if choice else "False"
    return str(choice)


def bind_candidates_for_execute(
    decision: Decision,
    question: Question,
    tokenize_content: Callable[[str], Sequence[int]],
) -> tuple[CandidateTokenSpec, ...]:
    """Bind controls, then relabel for ``execute_categorical_decision`` alignment.

    ``bind_control_candidates`` keeps Noul originals ``true``/``false`` on specs;
    execute expects ``True``/``False`` matching ``Decision.choices`` bools.

    Args:
        decision: Normalized categorical field.
        question: Native question supplying original label order.
        tokenize_content: Single-token control encoder hook.

    Returns:
        Candidate specs whose ``label`` values match executor choice strings.
    """
    originals = judgment_original_labels(question)
    bound = bind_control_candidates(originals, tokenize_content)
    return tuple(
        CandidateTokenSpec(_execute_label(choice), spec.token_ids)
        for spec, choice in zip(bound, decision.choices, strict=True)
    )


def answer_from_execution(
    question: Question, result: CategoricalExecutionResult
) -> Answer:
    """Map executor output to native judgment answers (#122 rev2).

    Args:
        question: Native question that produced ``result.decision``.
        result: Softmax outcome from ``execute_categorical_decision``.

    Returns:
        ``NoulAnswer``, ``ChoiceAnswer``, or ``ScoreAnswer`` per question type,
        with ``off_option_flag`` copied from ``result.off_option``.

    Raises:
        JudgmentValidationError: Unsupported question type.
    """
    flag = result.off_option.off_option_flag
    if isinstance(question, Noul):
        noul = next(p for choice, p in result.probabilities if choice is True)
        return NoulAnswer(noul=noul, off_option_flag=flag)
    if isinstance(question, Choice):
        originals = judgment_original_labels(question)
        probs = {originals[i]: p for i, (_, p) in enumerate(result.probabilities)}
        idx = result.decision.choices.index(result.value)
        selected = originals[idx]
        return ChoiceAnswer(
            choice=selected,
            confidence=probs[selected],
            probabilities=probs,
            off_option_flag=flag,
        )
    if isinstance(question, Score):
        probs = {int(choice): p for choice, p in result.probabilities}
        legend = {i: str(question.criteria[i]) for i in range(len(question.criteria))}
        score = sum(level * p for level, p in probs.items())
        return ScoreAnswer(
            score=score,
            confidence=max(probs.values()),
            legend=legend,
            probabilities=probs,
            off_option_flag=flag,
        )
    msg = f"unsupported question type: {type(question)!r}"
    raise JudgmentValidationError(msg)


def _check_threshold(threshold: float | None) -> None:
    """Reject an invalid off-option threshold before any scoring IO.

    Raises:
        JudgmentValidationError: ``threshold`` is not ``None`` or a number
            in ``[0, 1]``.
    """
    try:
        check_off_option_threshold(threshold)
    except DecisionExecutionError as exc:
        raise JudgmentValidationError(str(exc)) from exc


def _field_criteria(question: Question) -> Mapping[str, str] | None:
    if isinstance(question, Noul) and question.criteria:
        return {str(k): str(v) for k, v in question.criteria.items()}
    if isinstance(question, Choice):
        return {
            str(k): str(v) if v is not None else ""
            for k, v in question.criteria.items()
        }
    if isinstance(question, Score):
        return {str(i): str(desc) for i, desc in enumerate(question.criteria)}
    return None


def _state_context(state: str | dict[str, Any] | list[Any]) -> str:
    """Render the state as text with every caller media marker neutralized.

    Args:
        state: Content under evaluation.

    Returns:
        State text that holds no ``MEDIA_MARKER`` (#433).
    """
    text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
    return neutralize_media_markers(text)


def _media_context(
    state: str | dict[str, Any] | list[Any],
    media: tuple[ImageInput, ...],
) -> str:
    """Prepend one media marker per image to the rendered state context.

    Args:
        state: Content under evaluation.
        media: Images the scoring prefix must mark.

    Returns:
        Context text whose marker count matches ``len(media)``.
    """
    context = _state_context(state)
    if not media:
        return context
    return "\n".join([MEDIA_MARKER] * len(media) + [context])


class ScoringJudgmentAdapter:
    """JudgmentPort implementation using injected candidate scoring.

    Examples:
        ```python
        adapter = ScoringJudgmentAdapter(scoring_port, tokenize_content=tokenize)
        adapter.judge("state", {"q": Noul()}, "model-id")
        ```

    Attributes:
        _port (CandidateScoringPort): Injected scorer; not closed by this adapter.
        _tokenize (Callable[[str], Sequence[int]]): Control-string tokenizer hook.
        _temperature (float): Softmax temperature forwarded to execute.
        _served_template (ServedTemplateClass | None): Served family for
            every prefix; ``None`` when unknown.
        _pinned_model (str | None): When set, ``judge`` rejects other model ids
            before tokenization or scoring IO.
        _framing (ModelFramingPort | None): When set, composes every prefix in
            place of the served-template choice.
        _text_parts (TextParts): Validated option block and context templates.
    """

    def __init__(
        self,
        scoring_port: CandidateScoringPort,
        *,
        tokenize_content: Callable[[str], Sequence[int]],
        temperature: float = 1.0,
        served_template: ServedTemplateClass | None = None,
        pinned_model: str | None = None,
        framing: ModelFramingPort | None = None,
        text_parts: TextParts | None = None,
    ) -> None:
        """Wire scoring port, tokenizer, family, pin, framing and text parts.

        ``framing`` is exclusive with ``served_template``: a framing composes
        every prefix, so the family would not be read. Both text parts work
        with every path; a framing receives the user text that
        ``context_template`` rendered (#373). ``TextParts`` validates both
        templates when it is built.

        Raises:
            ValueError: ``framing`` is set with ``served_template``.
        """
        if framing is not None and served_template is not None:
            msg = "pass framing or served_template, not both"
            raise ValueError(msg)
        self._text_parts = text_parts or TextParts()
        self._port = scoring_port
        self._tokenize = tokenize_content
        self._temperature = temperature
        self._served_template = served_template
        self._pinned_model = pinned_model
        self._framing = framing

    @property
    def text_parts(self) -> TextParts:
        """Return the text parts; ``text_parts.receipt()`` pins them by digest.

        Returns:
            Validated option block and context templates.
        """
        return self._text_parts

    def _field_block(self, decision: Decision, question: Question) -> str:
        """Render one field block with the ``option_block`` text part.

        Args:
            decision: Normalized categorical field.
            question: Native question supplying criteria and label order.

        Returns:
            Field instructions that map each control to its label, with every
            caller media marker neutralized (#433).
        """
        block = render_field_instructions(
            decision,
            choice_criteria=_field_criteria(question),
            original_labels=judgment_original_labels(question),
            option_block=self._text_parts.option_block,
        )
        return neutralize_media_markers(block)

    def _field_prefix(
        self, context: str, field_block: str, media: tuple[ImageInput, ...]
    ) -> str:
        """Compose one field prefix with the framing, else the served family.

        With a framing, ``served_template`` is not read. The framing receives
        the user text that ``context_template`` rendered. The framing prefix
        must keep one media marker per image. A prefix whose last turn is a
        Gemma 4 model turn must end with the no-thinking prefill (#354).
        Either failure is raised here, before any scoring IO.

        Args:
            context: Rendered state context, media markers included.
            field_block: Rendered field instructions.
            media: Images the prefix marks.

        Returns:
            Scoring prefix ending at the answer boundary.

        Raises:
            JudgmentValidationError: Framing prefix media-marker count differs
                from ``len(media)``, or a ``compose_served_prefix`` rejection.
            GemmaTemplateError: Framing prefix is a Gemma 4 model turn without
                the no-thinking prefill; the message names the framing class.
        """
        if self._framing is None:
            return compose_served_prefix(
                context=context,
                field_block=field_block,
                media=media,
                served_template=self._served_template,
                context_template=self._text_parts.context_template,
            )
        user_text = render_context(
            self._text_parts.context_template,
            context=context,
            field_block=field_block,
        )
        prefix = self._framing.compose_prefix(user_text=user_text, media=media)
        markers = count_media_markers(prefix)
        if markers != len(media):
            msg = (
                f"framing prefix holds {markers} {MEDIA_MARKER} media marker(s) "
                f"but {len(media)} image(s) were supplied"
            )
            raise JudgmentValidationError(msg)
        require_no_thinking_prefill(
            prefix,
            field_block=field_block,
            framing_class=type(self._framing).__qualname__,
        )
        return prefix

    def judge(
        self,
        state: str | dict[str, Any] | list[Any],
        questions: Mapping[str, Question | Mapping[str, Any]],
        model: str,
        *,
        media: tuple[ImageInput, ...] | None = None,
        off_option_threshold: float | None = None,
    ) -> JudgmentResponse:
        """Validate all questions, score sequentially, return typed answers.

        Builds a scoring prefix whose field block maps each ordinal control
        string to the public answer label before calling the scorer. A native
        served family wraps every field prefix whether or not ``media`` is
        empty. When ``media`` is non-empty, every prefix carries one
        ``MEDIA_MARKER`` per image and every scoring request carries the same
        image tuple. Without a native family, text-only prefixes use ChatML.
        An injected framing composes every prefix instead. The
        ``text_parts`` templates render the field block and the user text.
        A set ``off_option_threshold`` flags, and does not raise for, each answer
        whose off-option mass is above it; a ``None`` mass never flags.

        Args:
            state: Content under evaluation (text or JSON-serializable value).
            questions: Named native questions.
            model: Backend model id forwarded to the scorer.
            media: Images to condition every scored field on, in order.
            off_option_threshold: Off-option mass limit in ``[0, 1]``, or
                ``None`` (the default) to turn the guard off.

        Returns:
            ``JudgmentResponse`` with one typed answer and one
            ``OffOptionReceipt`` per question id.

        Raises:
            JudgmentValidationError: Invalid threshold, model, wire shape,
                question payload, unsupported served template, media without
                a native Gemma 3 or Gemma 4 served template, or a framing prefix
                with the wrong media-marker count, before any scoring IO.
            GemmaTemplateError: A framing prefix ends in a Gemma 4 model turn
                without the no-thinking prefill, before any scoring IO.
        """
        if not model.strip():
            raise JudgmentValidationError("model must be non-empty")
        _check_threshold(off_option_threshold)
        if self._pinned_model is not None and model != self._pinned_model:
            msg = f"model {model!r} does not match pinned model {self._pinned_model!r}"
            raise JudgmentValidationError(msg)
        images = media or ()
        context = _media_context(state, images)
        prepared: list[
            tuple[str, Question, Decision, tuple[CandidateTokenSpec, ...], str]
        ] = []
        for name, raw in questions.items():
            if not isinstance(raw, (Noul, Choice, Score)):
                msg = f"unsupported wire question for {name!r}"
                raise JudgmentValidationError(msg)
            decision = normalize_question(raw, field_name=name)
            candidates = bind_candidates_for_execute(decision, raw, self._tokenize)
            field_block = self._field_block(decision, raw)
            prefix = self._field_prefix(context, field_block, images)
            prepared.append((name, raw, decision, candidates, prefix))

        answers: dict[str, Answer] = {}
        receipts: dict[str, OffOptionReceipt] = {}
        usage = TokenUsage()
        result_model = model
        for name, raw, decision, candidates, prefix in prepared:
            executed = execute_categorical_decision(
                decision,
                prefix=prefix,
                candidates=candidates,
                port=self._port,
                model=model,
                temperature=self._temperature,
                media=images,
            )
            executed = apply_off_option_threshold(executed, off_option_threshold)
            answers[name] = answer_from_execution(raw, executed)
            receipts[name] = executed.off_option
            result_model = executed.model
            usage = _merge_usage(usage, executed.usage)
        return JudgmentResponse(
            model=result_model,
            usage=usage,
            answers=answers,
            off_option=receipts,
            text_parts=self._text_parts.receipt(),
        )


def _merge_usage(left: TokenUsage, right: TokenUsage) -> TokenUsage:
    """Sum token counts when both sides report usage.

    Returns:
        Combined usage; ``None`` counts stay ``None`` only when both inputs omit them.
    """
    in_left, in_right = left.input_tokens, right.input_tokens
    out_left, out_right = left.output_tokens, right.output_tokens
    input_tokens = (
        None
        if in_left is None and in_right is None
        else (in_left or 0) + (in_right or 0)
    )
    output_tokens = (
        None
        if out_left is None and out_right is None
        else (out_left or 0) + (out_right or 0)
    )
    return TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens)


class ScoringAdapterSettings(TypedDict, total=False):
    """Optional keywords that ``judge_with_scoring`` forwards to the adapter.

    Attributes:
        temperature (float): Softmax temperature for execute; default ``1.0``.
        served_template (ServedTemplateClass | None): Served family for media
            prefixes; default ``None`` (unknown).
        off_option_threshold (float | None): Forwarded to ``judge``, not to
            the adapter; default ``None`` (guard off).
    """

    temperature: float
    served_template: ServedTemplateClass | None
    off_option_threshold: float | None


def judge_with_scoring(
    state: str | dict[str, Any] | list[Any],
    questions: Mapping[str, Question | Mapping[str, Any]],
    model: str,
    *,
    scoring_port: CandidateScoringPort,
    tokenize_content: Callable[[str], Sequence[int]],
    media: tuple[ImageInput, ...] | None = None,
    framing: ModelFramingPort | None = None,
    **settings: Unpack[ScoringAdapterSettings],
) -> JudgmentResponse:
    """One-shot judgment via ``ScoringJudgmentAdapter``.

    Non-empty ``media`` needs ``served_template=NATIVE_GEMMA3_TURN`` or
    ``NATIVE_GEMMA4_TURN``; an omitted, unknown or unsupported family fails
    closed before any scoring IO. A ``framing`` composes every prefix
    instead; ``None`` keeps the served-family path.

    Args:
        state: Content under evaluation.
        questions: Named native questions.
        model: Backend model id.
        scoring_port: Injected candidate scorer.
        tokenize_content: Control-string tokenizer hook.
        media: Images to condition every scored field on, in order.
        framing: Model framing that composes every prefix, or ``None``.

    Other Parameters:
        temperature (float): Softmax temperature for execute.
        served_template (ServedTemplateClass | None): Served family for media
            prefixes; ``None`` means unknown.
        off_option_threshold (float | None): Off-option mass limit that
            ``judge`` applies; ``None`` turns the guard off.

    Returns:
        ``JudgmentResponse`` from a fresh adapter instance.

    Raises:
        ValueError: Both ``framing`` and ``served_template`` are set.
    """
    adapter = ScoringJudgmentAdapter(
        scoring_port,
        tokenize_content=tokenize_content,
        framing=framing,
        temperature=settings.get("temperature", 1.0),
        served_template=settings.get("served_template"),
    )
    threshold = settings.get("off_option_threshold")
    return adapter.judge(
        state, questions, model, media=media, off_option_threshold=threshold
    )

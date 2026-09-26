"""Sync JudgmentPort adapter over CandidateScoringPort (#125).

Text-only prefixes use degraded ChatML. Media prefixes use the served native
turn family and fail closed when that family is unknown (#157).

Examples:
    ```python
    from typevet.judge import ScoringJudgmentAdapter
    from typevet.domain.judgment_questions import Noul
    from tests.fixtures.scoring_contract import ContractScoringFake

    fake = ContractScoringFake(logprobs={"True": -0.2, "False": -1.0})
    port = ScoringJudgmentAdapter(fake, tokenize_content=lambda s: (ord(s[0]),))
    port.judge("text", {"q": Noul()}, "fake")
    ```

See Also:
    - [typevet.ports.judgment][]: JudgmentPort protocol
    - [typevet.domain.judgment_normalize][]: Normalize and control bind
    - [typevet.domain.media][]: Images the keyword-only ``media`` argument takes
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from typevet.adapters.outbound.gemma import (
    ServedTemplateClass,
    compose_media_scoring_prefix,
)
from typevet.domain.candidate_scoring_request import CandidateTokenSpec
from typevet.domain.decision_execute import (
    CategoricalExecutionResult,
    execute_categorical_decision,
)
from typevet.domain.decisions import Decision
from typevet.domain.errors import JudgmentValidationError
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
from typevet.domain.judgment_response import JudgmentResponse, TokenUsage
from typevet.domain.media import MEDIA_MARKER, ImageInput
from typevet.field_prompt import compose_scoring_prefix, render_field_instructions
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
        ``NoulAnswer``, ``ChoiceAnswer``, or ``ScoreAnswer`` per question type.

    Raises:
        JudgmentValidationError: Unsupported question type.
    """
    if isinstance(question, Noul):
        noul = next(p for choice, p in result.probabilities if choice is True)
        return NoulAnswer(noul=noul)
    if isinstance(question, Choice):
        originals = judgment_original_labels(question)
        probs = {originals[i]: p for i, (_, p) in enumerate(result.probabilities)}
        idx = result.decision.choices.index(result.value)
        selected = originals[idx]
        return ChoiceAnswer(
            choice=selected,
            confidence=probs[selected],
            probabilities=probs,
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
        )
    msg = f"unsupported question type: {type(question)!r}"
    raise JudgmentValidationError(msg)


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
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False)


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


def _compose_prefix(
    *,
    context: str,
    field_block: str,
    media: tuple[ImageInput, ...],
    served_template: ServedTemplateClass | None,
) -> str:
    """Pick ChatML for text and the served native turn family for media.

    Args:
        context: Rendered state context, media markers included.
        field_block: Rendered field instructions.
        media: Images the prefix marks.
        served_template: Served-template family, or ``None`` when unknown.

    Returns:
        Scoring prefix ending at the answer boundary.

    Raises:
        JudgmentValidationError: Media with an unknown or unsupported family.
    """
    if not media:
        return compose_scoring_prefix(context=context, field_block=field_block)
    if served_template is not ServedTemplateClass.NATIVE_GEMMA3_TURN:
        family = "unknown" if served_template is None else served_template.value
        msg = (
            "media scoring needs a native Gemma 3 served template "
            f"(served template={family})"
        )
        raise JudgmentValidationError(msg)
    return compose_media_scoring_prefix(
        context=context,
        field_block=field_block,
        template_class=served_template,
    )


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
        _served_template (ServedTemplateClass | None): Served family for media
            prefixes; ``None`` when unknown.
    """

    def __init__(
        self,
        scoring_port: CandidateScoringPort,
        *,
        tokenize_content: Callable[[str], Sequence[int]],
        temperature: float = 1.0,
        served_template: ServedTemplateClass | None = None,
    ) -> None:
        """Wire scoring port, tokenizer hook, temperature and served family."""
        self._port = scoring_port
        self._tokenize = tokenize_content
        self._temperature = temperature
        self._served_template = served_template

    def judge(
        self,
        state: str | dict[str, Any] | list[Any],
        questions: Mapping[str, Question | Mapping[str, Any]],
        model: str,
        *,
        media: tuple[ImageInput, ...] | None = None,
    ) -> JudgmentResponse:
        """Validate all questions, score sequentially, return typed answers.

        Builds a scoring prefix whose field block maps each ordinal control
        string to the public answer label before calling the scorer. When
        ``media`` is non-empty, every field prefix uses the served native turn
        family, carries one ``MEDIA_MARKER`` per image, and every scoring
        request carries the same image tuple. Text-only prefixes use ChatML.

        Args:
            state: Content under evaluation (text or JSON-serializable value).
            questions: Named native questions.
            model: Backend model id forwarded to the scorer.
            media: Images to condition every scored field on, in order.

        Returns:
            ``JudgmentResponse`` with one typed answer per question id.

        Raises:
            JudgmentValidationError: Invalid model, wire shape, question payload,
                or media without a native Gemma 3 served template, before any
                scoring IO.
        """
        if not model.strip():
            raise JudgmentValidationError("model must be non-empty")
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
            criteria = _field_criteria(raw)
            originals = judgment_original_labels(raw)
            field_block = render_field_instructions(
                decision,
                choice_criteria=criteria,
                original_labels=originals,
            )
            prefix = _compose_prefix(
                context=context,
                field_block=field_block,
                media=images,
                served_template=self._served_template,
            )
            prepared.append((name, raw, decision, candidates, prefix))

        answers: dict[str, Answer] = {}
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
            answers[name] = answer_from_execution(raw, executed)
            result_model = executed.model
            usage = _merge_usage(usage, executed.usage)
        return JudgmentResponse(model=result_model, usage=usage, answers=answers)


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


def judge_with_scoring(
    state: str | dict[str, Any] | list[Any],
    questions: Mapping[str, Question | Mapping[str, Any]],
    model: str,
    *,
    scoring_port: CandidateScoringPort,
    tokenize_content: Callable[[str], Sequence[int]],
    temperature: float = 1.0,
    media: tuple[ImageInput, ...] | None = None,
) -> JudgmentResponse:
    """One-shot judgment via ``ScoringJudgmentAdapter``.

    The served template is unknown here, so non-empty ``media`` fails closed;
    construct ``ScoringJudgmentAdapter`` with ``served_template`` instead.

    Args:
        state: Content under evaluation.
        questions: Named native questions.
        model: Backend model id.
        scoring_port: Injected candidate scorer.
        tokenize_content: Control-string tokenizer hook.
        temperature: Softmax temperature for execute.
        media: Images to condition every scored field on, in order.

    Returns:
        ``JudgmentResponse`` from a fresh adapter instance.
    """
    return ScoringJudgmentAdapter(
        scoring_port,
        tokenize_content=tokenize_content,
        temperature=temperature,
    ).judge(state, questions, model, media=media)

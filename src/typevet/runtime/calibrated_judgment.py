"""Judgment port wrapper that applies calibration maps to Noul and Score (#352).

``CalibratedJudgment`` wraps any ``JudgmentPort``. The caller declares the
task id and the serving backend once. The wrapper refuses a map fitted for
another task or backend at construction. It refuses a response from another
model at judgment time, because a map does not transfer between tasks,
models or backends (#343).

The model check reads ``JudgmentResponse.model``: the model id that the inner
port reports for the answers. A response carries no backend name, so the
``backend`` argument is the backend that the caller declares for the inner
port. The model check runs for every map on every response, also when the
call does not ask the mapped question. A mapped question that the call does
not ask is skipped.

A Noul map changes the Noul probability. A Score map (a map with ``levels``)
maps each level probability, rescales the levels to sum to 1, and then sets
``score`` to the expected level and ``confidence`` to the largest level
probability. A Choice question is refused.

Examples:
    ```python
    from typevet.adapters.inbound import load_calibration_map
    from typevet.runtime import CalibratedJudgment

    maps = {"fraud": load_calibration_map("maps/fraud.json", sha256=digest)}
    port = CalibratedJudgment(inner, maps, task_id="fraud", backend="vllm")
    response = port.judge(text, {"fraud": Noul()}, "gemma")
    response.calibration["fraud"].raw
    ```

See Also:
    - [typevet.domain.calibration][]: Map types and the validator
    - [typevet.adapters.inbound.calibration_map][]: The file reader
    - [typevet.ports.judgment][]: JudgmentPort protocol
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

from typevet.domain.calibration import CalibrationMap, CalibrationRecord
from typevet.domain.errors import (
    CalibrationMapError,
    CalibrationModelMismatchError,
    CalibrationTargetError,
    CalibrationTaskMismatchError,
)
from typevet.domain.judgment_answers import Answer, NoulAnswer, ScoreAnswer
from typevet.domain.judgment_questions import Question, Score, question_types
from typevet.domain.judgment_response import JudgmentResponse
from typevet.domain.media import ImageInput
from typevet.ports.judgment import JudgmentPort

_TARGET_MESSAGE = "calibration map kind does not match the question"


def _checked_maps(
    maps: Mapping[str, CalibrationMap], *, task_id: str, backend: str
) -> dict[str, CalibrationMap]:
    """Validate the map collection against the declared task and backend.

    Args:
        maps: Maps keyed by question name.
        task_id: The caller-defined task id.
        backend: The serving backend of the inner port.

    Returns:
        A copy of ``maps``.

    Raises:
        CalibrationMapError: ``maps`` is empty, or holds a bad key or value.
        CalibrationTaskMismatchError: A map was fitted for another task.
        CalibrationModelMismatchError: A map was fitted on another backend.
    """
    if not maps:
        raise CalibrationMapError("calibration maps must name at least one question")
    for name, cmap in maps.items():
        if not isinstance(name, str) or not name:
            raise CalibrationMapError("calibration map keys must be question names")
        if not isinstance(cmap, CalibrationMap):
            raise CalibrationMapError("calibration map values must be CalibrationMap")
        if cmap.fitted_on.task_id != task_id:
            msg = "calibration map task does not match the wrapper task_id"
            raise CalibrationTaskMismatchError(msg)
        if cmap.fitted_on.backend != backend:
            msg = "calibration map backend does not match the wrapper backend"
            raise CalibrationModelMismatchError(msg)
    return dict(maps)


def _question_levels(question: object) -> int | None:
    """Return the level count of a typed or wire Score question, when known.

    Args:
        question: A typed question or a wire dictionary.

    Returns:
        The number of criteria, or ``None`` when the question does not list
        them.
    """
    if isinstance(question, Score):
        return len(question.criteria)
    criteria = question.get("criteria") if isinstance(question, Mapping) else None
    return len(criteria) if isinstance(criteria, (list, tuple)) else None


def _check_question(cmap: CalibrationMap, kind: str | None, question: object) -> None:
    """Refuse a mapped question whose kind or level count the map cannot take.

    An unknown wire type passes; the answer check after the call decides.

    Args:
        cmap: The map for the question.
        kind: The wire type name from ``question_types``.
        question: The typed question or wire dictionary.

    Raises:
        CalibrationTargetError: The question is a Choice, the map kind does
            not match the question kind, or the level count differs.
    """
    if kind == "choice":
        raise CalibrationTargetError(_TARGET_MESSAGE)
    wanted = "noul" if cmap.levels is None else "score"
    if kind in {"noul", "score"} and kind != wanted:
        raise CalibrationTargetError(_TARGET_MESSAGE)
    levels = _question_levels(question) if kind == "score" else None
    if levels is not None and levels != cmap.levels:
        raise CalibrationTargetError(_TARGET_MESSAGE)


def _calibrated_noul(
    answer: Answer, cmap: CalibrationMap
) -> tuple[NoulAnswer, CalibrationRecord]:
    """Apply a Noul map to one answer.

    Args:
        answer: The inner answer.
        cmap: A Noul map.

    Returns:
        The calibrated answer and its record.

    Raises:
        CalibrationTargetError: The answer is not a Noul.
    """
    if not isinstance(answer, NoulAnswer):
        raise CalibrationTargetError(_TARGET_MESSAGE)
    calibrated = cmap.apply(answer.noul)
    record = CalibrationRecord(
        raw=answer.noul,
        calibrated=calibrated,
        method=cmap.method,
        map_sha256=cmap.sha256,
    )
    return NoulAnswer(noul=calibrated, off_option_flag=answer.off_option_flag), record


def _calibrated_score(
    answer: Answer, cmap: CalibrationMap
) -> tuple[ScoreAnswer, CalibrationRecord]:
    """Apply a pooled Score map to each level and recompute the score.

    Args:
        answer: The inner answer.
        cmap: A Score map.

    Returns:
        The calibrated answer and its record.

    Raises:
        CalibrationTargetError: The answer is not a Score, or its level
            count differs from the map.
    """
    if not isinstance(answer, ScoreAnswer) or len(answer.probabilities) != cmap.levels:
        raise CalibrationTargetError(_TARGET_MESSAGE)
    probabilities = cmap.apply_levels(answer.probabilities)
    low, high = min(probabilities), max(probabilities)
    score = sum(level * p for level, p in probabilities.items())
    calibrated = ScoreAnswer(
        score=min(max(score, low), high),
        confidence=max(probabilities.values()),
        legend=dict(answer.legend),
        probabilities=probabilities,
        off_option_flag=answer.off_option_flag,
    )
    record = CalibrationRecord(
        raw=answer.score,
        calibrated=calibrated.score,
        method=cmap.method,
        map_sha256=cmap.sha256,
        raw_levels=dict(answer.probabilities),
        calibrated_levels=dict(probabilities),
    )
    return calibrated, record


class CalibratedJudgment:
    """``JudgmentPort`` that calibrates mapped answers of an inner port.

    The answer carries the calibrated value. ``JudgmentResponse.calibration``
    records the raw value, the calibrated value, the method and the map
    digest for each mapped question. A Score record also holds the raw and
    calibrated level probabilities. Unmapped questions stay raw.

    Attributes:
        _inner (JudgmentPort): The wrapped port; not closed by this wrapper.
        _maps (dict[str, CalibrationMap]): Maps keyed by question name.
        _task_id (str): The caller-defined task id.
        _backend (str): The declared serving backend of the inner port.

    Examples:
        ```python
        port = CalibratedJudgment(inner, maps, task_id="fraud", backend="vllm")
        ```
    """

    def __init__(
        self,
        inner: JudgmentPort,
        maps: Mapping[str, CalibrationMap],
        *,
        task_id: str,
        backend: str,
    ) -> None:
        """Bind the inner port and refuse maps for another task or backend.

        Args:
            inner: The judgment port to wrap.
            maps: Validated maps keyed by question name.
            task_id: The caller-defined task id each map must declare.
            backend: The serving backend of ``inner``, for example
                ``llama_cpp`` or ``vllm``. Each map must declare it.
        """
        self._maps = _checked_maps(maps, task_id=task_id, backend=backend)
        self._inner = inner
        self._task_id = task_id
        self._backend = backend

    def judge(
        self,
        state: str | dict[str, Any] | list[Any],
        questions: Mapping[str, Question | Mapping[str, Any]],
        model: str,
        *,
        media: tuple[ImageInput, ...] | None = None,
        off_option_threshold: float | None = None,
    ) -> JudgmentResponse:
        """Judge with the inner port, then calibrate each mapped answer.

        Args:
            state: Content under evaluation.
            questions: Question names to typed questions or wire dictionaries.
            model: Backend model id or alias, forwarded to the inner port.
            media: Images forwarded to the inner port.
            off_option_threshold: Off-option guard forwarded to the inner port.

        Returns:
            The inner response with calibrated answers and records.

        Raises:
            CalibrationTargetError: Before the inner call, a mapped question
                is a Choice, its kind does not match the map, or its level
                count differs. After the call, a mapped answer does not
                match the map.
            CalibrationModelMismatchError: The response model differs from
                the model of any map.
            CalibrationMapError: A mapped answer already carries a record.
        """
        for name, kind in question_types(questions).items():
            if name in self._maps:
                _check_question(self._maps[name], kind, questions[name])
        response = self._inner.judge(
            state,
            questions,
            model,
            media=media,
            off_option_threshold=off_option_threshold,
        )
        if any(m.fitted_on.model != response.model for m in self._maps.values()):
            msg = "calibration map model does not match the judgment model"
            raise CalibrationModelMismatchError(msg)
        answers: dict[str, Answer] = dict(response.answers)
        records = dict(response.calibration)
        for name, cmap in self._maps.items():
            if name not in answers:
                continue
            calibrate = _calibrated_noul if cmap.levels is None else _calibrated_score
            answer, record = calibrate(answers[name], cmap)
            if name in records:
                msg = "judgment answer already carries a calibration record"
                raise CalibrationMapError(msg)
            answers[name], records[name] = answer, record
        return dataclasses.replace(response, answers=answers, calibration=records)

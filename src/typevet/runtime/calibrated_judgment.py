"""Judgment port wrapper that applies calibration maps to Noul answers (#352).

``CalibratedJudgment`` wraps any ``JudgmentPort``. The caller declares the
task id and the serving backend once. The wrapper refuses a map fitted for
another task or backend at construction. It refuses a response from another
model at judgment time, because a map does not transfer between tasks,
models or backends (#343).

The model check reads ``JudgmentResponse.model``: the model id that the inner
port reports for the answers. A response carries no backend name, so the
``backend`` argument is the backend that the caller declares for the inner
port.

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
from typevet.domain.judgment_answers import Answer, NoulAnswer
from typevet.domain.judgment_questions import Choice, Question, Score
from typevet.domain.judgment_response import JudgmentResponse
from typevet.domain.media import ImageInput
from typevet.ports.judgment import JudgmentPort

_TARGET_MESSAGE = "calibration maps apply to Noul questions only"


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


class CalibratedJudgment:
    """``JudgmentPort`` that calibrates mapped Noul answers of an inner port.

    The answer carries the calibrated value. ``JudgmentResponse.calibration``
    records the raw value, the calibrated value, the method and the map
    digest for each mapped question. Unmapped questions stay raw.

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
        """Judge with the inner port, then calibrate each mapped Noul answer.

        Args:
            state: Content under evaluation.
            questions: Question names to typed questions or wire dictionaries.
            model: Backend model id or alias, forwarded to the inner port.
            media: Images forwarded to the inner port.
            off_option_threshold: Off-option guard forwarded to the inner port.

        Returns:
            The inner response with calibrated answers and records.

        Raises:
            CalibrationTargetError: A mapped question is a Choice or Score,
                before the inner call, or a mapped answer is not a Noul.
            CalibrationModelMismatchError: The response model differs from
                the model of a map that applies.
            CalibrationMapError: A mapped answer already carries a record.
        """
        for name, question in questions.items():
            if name in self._maps and isinstance(question, (Choice, Score)):
                raise CalibrationTargetError(_TARGET_MESSAGE)
        response = self._inner.judge(
            state,
            questions,
            model,
            media=media,
            off_option_threshold=off_option_threshold,
        )
        answers: dict[str, Answer] = dict(response.answers)
        records = dict(response.calibration)
        for name, cmap in self._maps.items():
            if name not in answers:
                continue
            answer = answers[name]
            if not isinstance(answer, NoulAnswer):
                raise CalibrationTargetError(_TARGET_MESSAGE)
            if cmap.fitted_on.model != response.model:
                msg = "calibration map model does not match the judgment model"
                raise CalibrationModelMismatchError(msg)
            if name in records:
                msg = "judgment answer already carries a calibration record"
                raise CalibrationMapError(msg)
            calibrated = cmap.apply(answer.noul)
            answers[name] = NoulAnswer(noul=calibrated)
            records[name] = CalibrationRecord(
                raw=answer.noul,
                calibrated=calibrated,
                method=cmap.method,
                map_sha256=cmap.sha256,
            )
        return dataclasses.replace(response, answers=answers, calibration=records)

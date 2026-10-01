"""Caller-supplied calibration maps for Noul probabilities (#352).

A calibration map is a small fitted function. It changes the probability that
the caller reads. It does not change the model. The ``typevet_evals`` study in
#343 fits the maps; this module validates a parsed map and applies it.

The module performs no IO. The file reader is
``typevet.adapters.inbound.calibration_map.load_calibration_map``. It checks
the file digest, parses the JSON and calls ``calibration_map_from_mapping``.

Each map clips its input and its output to
``[CALIBRATION_CLIP, 1 - CALIBRATION_CLIP]``. The isotonic map returns the
first knot value below the first knot, the nearest lower knot value between
knots and the last knot value above the last knot.

Every error message names fields and rules only, never a value.

Attributes:
    CALIBRATION_MAP_SCHEMA (str): Schema id that each map document declares.
    CALIBRATION_CLIP (float): Clip bound for map inputs and outputs.
    CALIBRATION_METHODS (frozenset[str]): Supported map methods.

Examples:
    ```python
    from typevet.domain.calibration import calibration_map_from_mapping

    cmap = calibration_map_from_mapping(document, sha256=digest)
    calibrated = cmap.apply(0.6)
    ```

See Also:
    - [typevet.adapters.inbound.calibration_map][]: The file reader
    - [typevet.runtime.calibrated_judgment][]: The judgment port wrapper
    - [typevet.domain.errors][]: ``CalibrationMapError`` and its subclasses
"""

from __future__ import annotations

import bisect
import itertools
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

from typevet.domain.errors import CalibrationDigestError, CalibrationMapError

CALIBRATION_MAP_SCHEMA: Final = "typevet.calibration_map/1"
CALIBRATION_CLIP: Final = 1e-6
CALIBRATION_METHODS: Final = frozenset({"temperature", "platt", "isotonic"})

_HEX: Final = frozenset("0123456789abcdef")
_SHA256_HEX_LENGTH: Final = 64
_METRICS: Final = ("ece", "brier", "accuracy")
_PARAMETER_KEYS: Final = {
    "temperature": frozenset({"temperature"}),
    "platt": frozenset({"slope", "intercept"}),
    "isotonic": frozenset({"knots", "values"}),
}


def _bad(field: str) -> CalibrationMapError:
    return CalibrationMapError(f"calibration map field {field!r} is invalid")


def _clip(p: float) -> float:
    return min(max(p, CALIBRATION_CLIP), 1.0 - CALIBRATION_CLIP)


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _probability(value: object) -> float | None:
    number = _finite(value)
    return number if number is not None and 0.0 <= number <= 1.0 else None


def _number(container: Mapping[str, object], key: str, field: str) -> float:
    number = _finite(container.get(key))
    if number is None:
        raise _bad(field)
    return number


def _count(container: Mapping[str, object], key: str, field: str) -> int:
    value = container.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise _bad(field)
    return value


def _text(container: Mapping[str, object], key: str, field: str) -> str:
    value = container.get(key)
    if not isinstance(value, str) or not value.strip():
        raise _bad(field)
    return value


def _section(container: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = container.get(key)
    if not isinstance(value, Mapping):
        raise _bad(key)
    return value


def normalize_sha256(digest: object) -> str:
    """Return a sha256 hex digest in lower case, or refuse it.

    Args:
        digest: The caller-supplied digest.

    Returns:
        The digest as 64 lower-case hex characters.

    Raises:
        CalibrationDigestError: The digest is missing or is not 64 hex
            characters.
    """
    if not isinstance(digest, str) or not digest:
        raise CalibrationDigestError("calibration map sha256 is missing")
    lowered = digest.lower()
    if len(lowered) != _SHA256_HEX_LENGTH or not set(lowered) <= _HEX:
        raise CalibrationDigestError("calibration map sha256 is not 64 hex characters")
    return lowered


@dataclass(frozen=True, slots=True)
class FittedOn:
    """Where a calibration map was fitted.

    Attributes:
        task_id (str): Caller-defined task the map belongs to.
        receipt (str): Path or name of the source receipt.
        receipt_sha256 (str): Digest of the source receipt, for traceability.
        n_calibration (int): Rows in the calibration half.
        model (str): Model id in the source receipt.
        backend (str): Serving backend in the source receipt.

    Examples:
        ```python
        assert cmap.fitted_on.task_id == "fraud-message"
        ```
    """

    task_id: str
    receipt: str
    receipt_sha256: str
    n_calibration: int
    model: str
    backend: str


@dataclass(frozen=True, slots=True)
class CalibrationEvaluation:
    """Held-out metrics that the producer recorded for a map.

    A map whose ``rule_met`` is ``False`` is still valid. The wrapper records
    the evaluation and never refuses on it.

    Attributes:
        n_evaluation (int): Rows in the evaluation half.
        before (dict[str, float]): ``ece``, ``brier`` and ``accuracy`` before.
        after (dict[str, float]): ``ece``, ``brier`` and ``accuracy`` after.
        rule_met (bool | None): The #343 rule outcome, when recorded.

    Examples:
        ```python
        assert cmap.evaluation.after["ece"] <= 1.0
        ```
    """

    n_evaluation: int
    before: dict[str, float]
    after: dict[str, float]
    rule_met: bool | None = None


@dataclass(frozen=True, slots=True)
class CalibrationRecord:
    """The raw and calibrated value of one Noul answer.

    Attributes:
        raw (float): The probability that the inner judgment returned.
        calibrated (float): The probability after the map.
        method (str): The map method.
        map_sha256 (str): The sha256 of the map file bytes.

    Examples:
        ```python
        record = CalibrationRecord(
            raw=0.6, calibrated=0.45, method="platt", map_sha256="0" * 64
        )
        assert record.raw == 0.6
        ```
    """

    raw: float
    calibrated: float
    method: str
    map_sha256: str


@dataclass(frozen=True, slots=True)
class TemperatureParameters:
    """Temperature map ``sigmoid(logit(p) / temperature)``.

    Attributes:
        temperature (float): The fitted temperature, above zero.

    Examples:
        ```python
        assert TemperatureParameters(1.0).map(0.7) == pytest.approx(0.7)
        ```
    """

    temperature: float

    def map(self, x: float) -> float:
        """Return the scaled value of a clipped probability.

        Args:
            x: A probability inside the clip bounds.

        Returns:
            The unclipped mapped probability.
        """
        return _sigmoid(_logit(x) / self.temperature)


@dataclass(frozen=True, slots=True)
class PlattParameters:
    """Platt map ``sigmoid(slope * logit(p) + intercept)``.

    Attributes:
        slope (float): The fitted slope.
        intercept (float): The fitted intercept.

    Examples:
        ```python
        assert PlattParameters(1.0, 0.0).map(0.7) == pytest.approx(0.7)
        ```
    """

    slope: float
    intercept: float

    def map(self, x: float) -> float:
        """Return the scaled value of a clipped probability.

        Args:
            x: A probability inside the clip bounds.

        Returns:
            The unclipped mapped probability.
        """
        return _sigmoid(self.slope * _logit(x) + self.intercept)


@dataclass(frozen=True, slots=True)
class IsotonicParameters:
    """Non-decreasing step map over ascending knots.

    Attributes:
        knots (tuple[float, ...]): Strictly ascending knots in ``[0, 1]``.
        values (tuple[float, ...]): Non-decreasing value at each knot.

    Examples:
        ```python
        assert IsotonicParameters((0.2, 0.8), (0.1, 0.9)).map(0.5) == 0.1
        ```
    """

    knots: tuple[float, ...]
    values: tuple[float, ...]

    def map(self, x: float) -> float:
        """Return the value of the nearest knot at or below ``x``.

        Args:
            x: A probability inside the clip bounds.

        Returns:
            The step value; the first knot value below the first knot.
        """
        index = bisect.bisect_right(self.knots, x) - 1
        return self.values[max(index, 0)]


CalibrationParameters = TemperatureParameters | PlattParameters | IsotonicParameters
"""Union of the fitted parameter types, one per method."""


@dataclass(frozen=True, slots=True)
class CalibrationMap:
    """One validated calibration map for one task and one question.

    Attributes:
        method (str): ``temperature``, ``platt`` or ``isotonic``.
        parameters (CalibrationParameters): Fitted parameters for ``method``.
        fitted_on (FittedOn): Task, receipt, model and backend of the fit.
        evaluation (CalibrationEvaluation): Held-out metrics, recorded only.
        producer (str): The ``typevet_evals`` version that wrote the map.
        sha256 (str): The sha256 of the map file bytes.

    Examples:
        ```python
        cmap = calibration_map_from_mapping(document, sha256=digest)
        assert 0.0 < cmap.apply(0.6) < 1.0
        ```
    """

    method: str
    parameters: CalibrationParameters
    fitted_on: FittedOn
    evaluation: CalibrationEvaluation
    producer: str
    sha256: str

    def apply(self, p: float) -> float:
        """Map one probability through the fitted function.

        Args:
            p: A probability in ``[0, 1]``.

        Returns:
            The calibrated probability, clipped to ``[CALIBRATION_CLIP,
            1 - CALIBRATION_CLIP]``.

        Raises:
            CalibrationMapError: ``p`` is not a finite number in ``[0, 1]``.
        """
        x = _probability(p)
        if x is None:
            raise CalibrationMapError("calibration input is not a probability")
        return _clip(self.parameters.map(_clip(x)))


def _sequence(parameters: Mapping[str, object], key: str) -> tuple[float, ...]:
    value = parameters.get(key)
    if not isinstance(value, (list, tuple)) or not value:
        raise _bad(f"parameters.{key}")
    numbers = tuple(_probability(v) for v in value)
    if any(n is None for n in numbers):
        raise _bad(f"parameters.{key}")
    return tuple(n for n in numbers if n is not None)


def _parameters(method: str, parameters: Mapping[str, object]) -> CalibrationParameters:
    if set(parameters) != _PARAMETER_KEYS[method]:
        raise _bad("parameters")
    if method == "temperature":
        temperature = _number(parameters, "temperature", "parameters.temperature")
        if temperature <= 0.0:
            raise _bad("parameters.temperature")
        return TemperatureParameters(temperature)
    if method == "platt":
        return PlattParameters(
            slope=_number(parameters, "slope", "parameters.slope"),
            intercept=_number(parameters, "intercept", "parameters.intercept"),
        )
    knots = _sequence(parameters, "knots")
    values = _sequence(parameters, "values")
    if len(knots) != len(values):
        raise _bad("parameters.values")
    if any(b <= a for a, b in itertools.pairwise(knots)):
        raise _bad("parameters.knots")
    if any(b < a for a, b in itertools.pairwise(values)):
        raise _bad("parameters.values")
    return IsotonicParameters(knots, values)


def _fitted_on(data: Mapping[str, object]) -> FittedOn:
    section = _section(data, "fitted_on")
    receipt_sha256 = _text(section, "receipt_sha256", "fitted_on.receipt_sha256")
    try:
        normalize_sha256(receipt_sha256)
    except CalibrationDigestError:
        raise _bad("fitted_on.receipt_sha256") from None
    return FittedOn(
        task_id=_text(section, "task_id", "fitted_on.task_id"),
        receipt=_text(section, "receipt", "fitted_on.receipt"),
        receipt_sha256=receipt_sha256,
        n_calibration=_count(section, "n_calibration", "fitted_on.n_calibration"),
        model=_text(section, "model", "fitted_on.model"),
        backend=_text(section, "backend", "fitted_on.backend"),
    )


def _metrics(section: Mapping[str, object], key: str) -> dict[str, float]:
    block = _section(section, key)
    return {m: _number(block, m, f"evaluation.{key}.{m}") for m in _METRICS}


def _evaluation(data: Mapping[str, object]) -> CalibrationEvaluation:
    section = _section(data, "evaluation")
    rule_met = section.get("rule_met")
    if rule_met is not None and not isinstance(rule_met, bool):
        raise _bad("evaluation.rule_met")
    return CalibrationEvaluation(
        n_evaluation=_count(section, "n_evaluation", "evaluation.n_evaluation"),
        before=_metrics(section, "before"),
        after=_metrics(section, "after"),
        rule_met=rule_met,
    )


def calibration_map_from_mapping(
    data: Mapping[str, object], *, sha256: str
) -> CalibrationMap:
    """Validate a parsed map document and build a ``CalibrationMap``.

    Args:
        data: The parsed JSON document.
        sha256: The verified sha256 of the file bytes.

    Returns:
        The validated map.

    Raises:
        CalibrationDigestError: ``sha256`` is missing or malformed.
        CalibrationMapError: A required field is missing or invalid.
    """
    digest = normalize_sha256(sha256)
    if not isinstance(data, Mapping):
        raise CalibrationMapError("calibration map document is not an object")
    if data.get("schema") != CALIBRATION_MAP_SCHEMA:
        raise _bad("schema")
    method = data.get("method")
    if not isinstance(method, str) or method not in CALIBRATION_METHODS:
        raise _bad("method")
    return CalibrationMap(
        method=method,
        parameters=_parameters(method, _section(data, "parameters")),
        fitted_on=_fitted_on(data),
        evaluation=_evaluation(data),
        producer=_text(_section(data, "producer"), "typevet_evals", "producer"),
        sha256=digest,
    )

"""Write calibration map artifacts from labelled receipt series (#352).

The library reads a calibration map with
``typevet.adapters.inbound.load_calibration_map``. This module writes one. It
fits one method on the calibration half of a series, measures the evaluation
half before and after, and returns the ``typevet.calibration_map/1`` document.

A Noul map comes from one binary series. A Score map is one pooled
one-vs-rest map over every level probability. ``score_level_series`` builds
that series. It repeats each row id once per level, so ``split_indices`` puts
every level of a row in the same half. A Score artifact carries a top-level
``levels`` field; a Noul artifact has none.

No Score map has a held-out result yet. The #343 study fitted binary series
only. The Score path is a tested mechanism, not a measured improvement.

``write_calibration_map`` writes stable JSON and returns the sha256 of the
written bytes. The caller pins that digest when it loads the map.

Attributes:
    PRODUCER_DISTRIBUTION (str): The distribution whose version the artifact
        records as its producer.

Examples:
    ```python
    from typevet_evals.calibration_artifact import (
        calibration_map_artifact,
        write_calibration_map,
    )

    artifact = calibration_map_artifact(
        series,
        method="platt",
        task_id="signature-same-writer",
        model="google/gemma-4-31B-it",
        backend="vllm",
        receipt=receipt_path,
    )
    digest = write_calibration_map("map.json", artifact)
    ```

See Also:
    - [typevet_evals.calibration][]: The fitters, the split and the rule
    - [typevet.adapters.inbound.calibration_map][]: The file reader
    - [typevet.domain.calibration][]: The map validator and schema
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from importlib.metadata import version
from pathlib import Path
from typing import Any, Final

from typevet.domain.calibration import CALIBRATION_MAP_SCHEMA
from typevet_evals.calibration import (
    Calibrator,
    IsotonicMap,
    Series,
    accuracy_at_half,
    brier_score,
    fit_isotonic,
    fit_platt,
    fit_temperature,
    rule_met,
    split_indices,
)
from typevet_evals.face_match.metrics import expected_calibration_error

PRODUCER_DISTRIBUTION: Final = "typevet-evals"
_MIN_LEVELS: Final = 2
_FITTERS: Final = {
    "temperature": fit_temperature,
    "platt": fit_platt,
    "isotonic": fit_isotonic,
}


def score_level_series(
    name: str,
    source: str,
    ids: Sequence[str],
    probabilities: Sequence[Mapping[int, float]],
    gold: Sequence[int],
) -> Series:
    """Build the pooled one-vs-rest series of Score rows.

    Each row gives one entry per level, in ascending level order. The entry
    for level ``k`` holds the probability of ``k`` and the label
    ``gold == k``. The row id repeats for each level, so one row never
    spans both halves of the split.

    Args:
        name: Series name, ``family/receipt[/column]``.
        source: Receipt path recorded as ``fitted_on.receipt``.
        ids: Row ids in receipt order.
        probabilities: The level probabilities of each row.
        gold: The gold level of each row.

    Returns:
        The pooled binary series.

    Raises:
        ValueError: The inputs differ in length, a row has another level set,
            fewer than two levels exist, or a gold level is not a level.
    """
    if not len(ids) == len(probabilities) == len(gold):
        raise ValueError("score rows need one level map and one gold level each")
    levels = sorted(probabilities[0]) if probabilities else []
    if len(levels) < _MIN_LEVELS:
        raise ValueError("score rows need at least two levels")
    if any(sorted(row) != levels for row in probabilities):
        raise ValueError("every score row needs the same level set")
    if any(level not in levels for level in gold):
        raise ValueError("every gold level must be one of the score levels")
    return Series(
        name,
        source,
        tuple(item for item in ids for _ in levels),
        tuple(float(row[k]) for row in probabilities for k in levels),
        tuple(k == g for g in gold for k in levels),
    )


def _metrics(
    probabilities: Sequence[float], labels: Sequence[bool]
) -> dict[str, float]:
    return {
        "ece": expected_calibration_error(probabilities, labels),
        "brier": brier_score(probabilities, labels),
        "accuracy": accuracy_at_half(probabilities, labels),
    }


def _parameters(calibrator: Calibrator) -> dict[str, Any]:
    if isinstance(calibrator, IsotonicMap):
        return {"knots": list(calibrator.knots), "values": list(calibrator.values)}
    return calibrator.params()


def _checked_levels(levels: int | None) -> int | None:
    if levels is None:
        return None
    if isinstance(levels, bool) or not isinstance(levels, int):
        raise TypeError("levels must be an integer of two or more")
    if levels < _MIN_LEVELS:
        raise ValueError("levels must be an integer of two or more")
    return levels


def calibration_map_artifact(
    series: Series,
    *,
    method: str,
    task_id: str,
    model: str,
    backend: str,
    receipt: str | Path,
    levels: int | None = None,
) -> dict[str, Any]:
    """Fit one method on a series and return its calibration map document.

    The fit sees the calibration half only. The evaluation block holds the
    evaluation-half ECE, Brier score and accuracy before and after the fit,
    and the #343 rule result. ``fitted_on.receipt`` is ``series.source``.
    ``fitted_on.receipt_sha256`` is the lower-case hex sha256 of the bytes
    of the file at ``receipt``.

    The caller owns the check that ``levels`` matches the series. This
    function does not compare them. The wrapper refuses a mismatch at
    judgment time.

    Args:
        series: The binary series, or the pooled series of a Score.
        method: ``temperature``, ``platt`` or ``isotonic``.
        task_id: Caller-defined task the map belongs to.
        model: Model id in the source receipt.
        backend: Serving backend in the source receipt.
        receipt: The source receipt file.
        levels: The Score level count; ``None`` for a Noul map.

    Returns:
        The ``typevet.calibration_map/1`` document, with ``levels`` only for
        a Score map.

    Raises:
        TypeError: ``levels`` is not an integer.
        ValueError: The method is unknown, ``levels`` is below two, or a
            split half is empty.
    """
    if method not in _FITTERS:
        raise ValueError("method must be temperature, platt or isotonic")
    level_count = _checked_levels(levels)
    cal, ev = split_indices(series.ids)
    if not cal or not ev:
        raise ValueError("each half of the split needs at least one row")
    p, y = series.probabilities, series.labels
    ev_p, ev_y = [p[i] for i in ev], [y[i] for i in ev]
    calibrator = _FITTERS[method]([p[i] for i in cal], [y[i] for i in cal])
    before = _metrics(ev_p, ev_y)
    after = _metrics([calibrator(x) for x in ev_p], ev_y)
    artifact: dict[str, Any] = {
        "schema": CALIBRATION_MAP_SCHEMA,
        "method": method,
        "parameters": _parameters(calibrator),
        "fitted_on": {
            "task_id": task_id,
            "receipt": series.source,
            "receipt_sha256": hashlib.sha256(Path(receipt).read_bytes()).hexdigest(),
            "n_calibration": len(cal),
            "model": model,
            "backend": backend,
        },
        "evaluation": {
            "n_evaluation": len(ev),
            "before": before,
            "after": after,
            "rule_met": rule_met(before, after),
        },
        "producer": {"typevet_evals": version(PRODUCER_DISTRIBUTION)},
    }
    if level_count is not None:
        artifact["levels"] = level_count
    return artifact


def write_calibration_map(path: str | Path, artifact: Mapping[str, Any]) -> str:
    """Write a map document as stable JSON and return its sha256.

    Keys are sorted and floats keep full precision, so the same document
    gives the same bytes. A non-finite number is refused.

    Args:
        path: The file to write.
        artifact: The document from ``calibration_map_artifact``.

    Returns:
        The lower-case hex sha256 of the written bytes.
    """
    text = json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n"
    data = text.encode("utf-8")
    Path(path).write_bytes(data)
    return hashlib.sha256(data).hexdigest()

"""Duel scores, reliability bins, answer parsing and the duel receipt (#412).

A duel replays a doodle receipt round by round. In each round a person picks
a category and says how sure they are, from 50 to 100: the percent chance
that the pick is right, where 50 is a coin flip. Both players get the
same score: the pick Brier, ``(1 - c) ** 2`` when the pick is right and
``c ** 2`` when it is wrong. The model's ``c`` is its probability on its own
pick. Lower Brier is better; always saying 50% scores 0.25. The functions
here do no I/O.

Attributes:
    CONFIDENCE_MIN (int): Lowest confidence a person can give, in percent.
    CONFIDENCE_MAX (int): Highest confidence a person can give, in percent.
    BASELINE_BRIER (float): Brier of a player who always says 50%.
    DuelAnswer (type): One person's pick and confidence for one round.
    pick_brier (callable): Brier of one pick at one confidence.
    multiclass_brier (callable): Brier over every category for one row.
    reliability_rows (callable): Six reliability bins, 0-50 merged.
    parse_label (callable): Number or name to a category.
    parse_confidence (callable): Text to a confidence in percent.
    parse_answers (callable): Answers file body to answers.
    score_round (callable): One source row and one answer to a round record.
    duel_scores (callable): Accuracy, mean Brier and mean confidence.
    build_duel_receipt (callable): Round records to a duel receipt body.

Examples:
    ```python
    from typevet_evals.doodle_duel import pick_brier

    assert round(pick_brier(0.8, correct=False), 2) == 0.64
    ```

See Also:
    - [typevet_evals.doodle_duel.duel_session][]: round loop and I/O
    - [typevet_evals.face_match.metrics][]: the reliability table
    - [typevet_evals.calibration][]: the mean Brier score
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from typevet_evals.calibration import brier_score
from typevet_evals.datasets.quickdraw import DATASET_CREDIT, DATASET_LICENSE
from typevet_evals.doodle_duel.runner import RECEIPT_ISSUE
from typevet_evals.face_match.metrics import reliability_table

CONFIDENCE_MIN: Final[int] = 50
CONFIDENCE_MAX: Final[int] = 100
BASELINE_BRIER: Final[float] = 0.25

_BINS: Final[int] = 10
_LOW_BINS: Final[int] = 5
_PERCENT: Final[float] = 100.0
_CONFIDENCE_TEXT: Final = re.compile(r"\d{1,3}")
_ANSWER_KEYS: Final[tuple[str, ...]] = ("key_id", "label", "confidence")


@dataclass(frozen=True, slots=True)
class DuelAnswer:
    """One person's pick and confidence for one round.

    Attributes:
        key_id (str): Drawing id of the round.
        label (str): The picked category.
        confidence (int): How sure the person is, in percent, 50 to 100.

    Examples:
        ```python
        DuelAnswer("4858658959654912", "car", 85)
        ```
    """

    key_id: str
    label: str
    confidence: int


def pick_brier(confidence: float, *, correct: bool) -> float:
    """Return the Brier score of one pick.

    Args:
        confidence: Probability in [0, 1] that the pick is right.
        correct: Whether the pick is right.

    Returns:
        ``(1 - confidence) ** 2`` when right, else ``confidence ** 2``.
    """
    return (1.0 - confidence) ** 2 if correct else confidence**2


def multiclass_brier(
    probabilities: Mapping[str, float], true_label: str, categories: Sequence[str]
) -> float:
    """Return the Brier score summed over every category for one row.

    Args:
        probabilities: Probability of each category.
        true_label: The right category.
        categories: Every category of the question.

    Returns:
        The sum of ``(p - onehot) ** 2`` over ``categories``, from 0 to 2.

    Raises:
        ValueError: When ``true_label`` is not in ``categories``.
    """
    if true_label not in categories:
        msg = f"true label {true_label} is not a category"
        raise ValueError(msg)
    return sum(
        (float(probabilities[name]) - (1.0 if name == true_label else 0.0)) ** 2
        for name in categories
    )


def _row(
    name: str, lower: float, upper: float, count: int, conf: float, hits: float
) -> dict[str, Any]:
    mean = conf / count if count else None
    accuracy = hits / count if count else None
    gap = mean - accuracy if mean is not None and accuracy is not None else None
    return {
        "bin": name,
        "lower": lower,
        "upper": upper,
        "count": count,
        "mean_confidence": mean,
        "accuracy": accuracy,
        "gap": gap,
    }


def reliability_rows(
    confidences: Sequence[float], correct: Sequence[bool]
) -> list[dict[str, Any]]:
    """Bin confidences against right picks in six rows.

    The rows come from ten equal-width bins; the lower five merge into one
    ``0-50`` row. A value on an edge goes to the upper bin and 1.0 goes to
    ``90-100``. ``gap`` is mean confidence minus accuracy; positive means
    overconfident.

    Args:
        confidences: Confidence in [0, 1] for each pick.
        correct: Whether each pick is right.

    Returns:
        Rows ``0-50``, ``50-60`` ... ``90-100``. Empty rows hold ``None`` in
        ``mean_confidence``, ``accuracy`` and ``gap``.
    """
    table = reliability_table(confidences, correct, n_bins=_BINS)
    sums = [
        (
            b.count,
            (b.mean_confidence or 0.0) * b.count,
            round((b.fraction_same_person or 0.0) * b.count),
        )
        for b in table
    ]
    low = [sum(part) for part in zip(*sums[:_LOW_BINS], strict=True)]
    rows = [_row("0-50", 0.0, table[_LOW_BINS].lower, low[0], low[1], low[2])]
    for b, (count, conf, hits) in zip(table[_LOW_BINS:], sums[_LOW_BINS:], strict=True):
        name = f"{round(b.lower * _PERCENT)}-{round(b.upper * _PERCENT)}"
        rows.append(_row(name, b.lower, b.upper, count, conf, hits))
    return rows


def parse_label(text: str, categories: Sequence[str]) -> str:
    """Read a pick as a 1-based number or a category name.

    Args:
        text: The typed pick; spaces and letter case are ignored.
        categories: Categories in the order shown.

    Returns:
        The picked category.

    Raises:
        ValueError: When the text is no number in range and no category.
    """
    value = text.strip()
    if value.isdigit() and 1 <= int(value) <= len(categories):
        return categories[int(value) - 1]
    for name in categories:
        if name.lower() == value.lower():
            return name
    msg = f"pick a number from 1 to {len(categories)} or a category name"
    raise ValueError(msg)


def parse_confidence(text: str) -> int:
    """Read a confidence as a whole number of percent.

    Args:
        text: The typed confidence; a trailing ``%`` is allowed.

    Returns:
        The confidence, from 50 to 100.

    Raises:
        ValueError: When the text is not a whole number from 50 to 100.
    """
    value = text.strip().removesuffix("%").strip()
    if (
        _CONFIDENCE_TEXT.fullmatch(value)
        and CONFIDENCE_MIN <= int(value) <= CONFIDENCE_MAX
    ):
        return int(value)
    msg = f"type a whole number from {CONFIDENCE_MIN} to {CONFIDENCE_MAX}"
    raise ValueError(msg)


def _answer(
    entry: object, index: int, key_id: str, categories: Sequence[str]
) -> DuelAnswer:
    if not isinstance(entry, dict) or any(k not in entry for k in _ANSWER_KEYS):
        msg = f"answers[{index}] must be an object with {list(_ANSWER_KEYS)}"
        raise ValueError(msg)
    if entry["key_id"] != key_id:
        msg = f"answers[{index}] key_id {entry['key_id']} is not round key_id {key_id}"
        raise ValueError(msg)
    label = entry["label"]
    if not isinstance(label, str) or label not in categories:
        msg = f"answers[{index}] label {label!r} is not a category"
        raise ValueError(msg)
    confidence = entry["confidence"]
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, int)
        or not CONFIDENCE_MIN <= confidence <= CONFIDENCE_MAX
    ):
        msg = (
            f"answers[{index}] confidence {confidence!r} is not an int "
            f"{CONFIDENCE_MIN} to {CONFIDENCE_MAX}"
        )
        raise ValueError(msg)
    return DuelAnswer(key_id, label, confidence)


def parse_answers(
    raw: object, key_ids: Sequence[str], categories: Sequence[str]
) -> tuple[DuelAnswer, ...]:
    """Check an answers file body against the rounds to play.

    Args:
        raw: The parsed JSON, a list of ``{"key_id", "label", "confidence"}``.
        key_ids: Drawing id of each round, in round order.
        categories: Every category of the question.

    Returns:
        One answer per round, in round order.

    Raises:
        ValueError: When ``raw`` is not a list, its length is not the round
            count, or an entry has another key id, an unknown label or a
            confidence that is not an int from 50 to 100.
    """
    if not isinstance(raw, list) or len(raw) != len(key_ids):
        found = len(raw) if isinstance(raw, list) else type(raw).__name__
        msg = f"the answers file needs a list of {len(key_ids)} round(s): {found}"
        raise ValueError(msg)
    return tuple(
        _answer(entry, i, key_id, categories)
        for i, (entry, key_id) in enumerate(zip(raw, key_ids, strict=True))
    )


def score_round(
    number: int,
    row: Mapping[str, Any],
    answer: DuelAnswer,
    categories: Sequence[str],
) -> dict[str, Any]:
    """Score one round for both players.

    Args:
        number: 1-based round number.
        row: The source receipt row of the round.
        answer: The person's answer.
        categories: Every category of the question.

    Returns:
        The round record, with confidences as fractions in [0, 1].

    Raises:
        ValueError: When the answer label is not a category.
    """
    if answer.label not in categories:
        msg = f"pick {answer.label} is not a category"
        raise ValueError(msg)
    truth = row["true_label"]
    model_label = row["chosen_label"]
    model_confidence = float(row["probabilities"][model_label])
    player_confidence = answer.confidence / _PERCENT
    player_correct = answer.label == truth
    model_correct = model_label == truth
    return {
        "round": number,
        "key_id": row["key_id"],
        "true_label": truth,
        "player_label": answer.label,
        "player_confidence": player_confidence,
        "player_correct": player_correct,
        "player_brier": pick_brier(player_confidence, correct=player_correct),
        "model_label": model_label,
        "model_confidence": model_confidence,
        "model_correct": model_correct,
        "model_brier": pick_brier(model_confidence, correct=model_correct),
    }


def _player_scores(records: Sequence[Mapping[str, Any]], who: str) -> dict[str, float]:
    confidences = [float(r[f"{who}_confidence"]) for r in records]
    correct = [bool(r[f"{who}_correct"]) for r in records]
    return {
        "accuracy": sum(correct) / len(correct),
        "mean_brier": brier_score(confidences, correct),
        "mean_confidence": sum(confidences) / len(confidences),
    }


def duel_scores(
    records: Sequence[Mapping[str, Any]],
    source_rows: Sequence[Mapping[str, Any]],
    categories: Sequence[str],
) -> dict[str, Any]:
    """Return accuracy, mean Brier and mean confidence for both players.

    Args:
        records: Round records from ``score_round``; at least one.
        source_rows: The source receipt rows of the same rounds.
        categories: Every category of the question.

    Returns:
        ``{"player", "model", "baseline_brier"}``; the model entry also holds
        the mean multi-class Brier over the same rounds.
    """
    model = _player_scores(records, "model")
    model["multiclass_brier"] = sum(
        multiclass_brier(row["probabilities"], row["true_label"], categories)
        for row in source_rows
    ) / len(source_rows)
    return {
        "player": _player_scores(records, "player"),
        "model": model,
        "baseline_brier": BASELINE_BRIER,
    }


def build_duel_receipt(
    records: Sequence[Mapping[str, Any]],
    *,
    source_rows: Sequence[Mapping[str, Any]],
    categories: Sequence[str],
    input_mode: str,
    source_receipt: Mapping[str, Any],
    per_category: int,
) -> dict[str, Any]:
    """Build the duel receipt body; it holds no image bytes and no paths.

    Args:
        records: Round records from ``score_round``, in round order.
        source_rows: The source receipt rows of the same rounds.
        categories: Every category of the question.
        input_mode: ``"interactive"`` or ``"answers_file"``.
        source_receipt: ``{"path", "sha256", "backend", "model"}``.
        per_category: Doodles per category in the source run.

    Returns:
        The receipt body with scores, reliability rows and round records.
    """
    reliability = {
        who: reliability_rows(
            [float(r[f"{who}_confidence"]) for r in records],
            [bool(r[f"{who}_correct"]) for r in records],
        )
        for who in ("player", "model")
    }
    return {
        "issue": RECEIPT_ISSUE,
        "kind": "doodle_duel_play",
        "input_mode": input_mode,
        "source_receipt": dict(source_receipt),
        "pins": {
            "categories": list(categories),
            "per_category": per_category,
            "dataset_license": DATASET_LICENSE,
            "credit": DATASET_CREDIT,
        },
        "rounds": len(records),
        "scores": duel_scores(records, source_rows, categories),
        "reliability": reliability,
        "rows": [dict(r) for r in records],
    }

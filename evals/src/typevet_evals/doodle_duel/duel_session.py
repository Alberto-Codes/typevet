"""Duel rounds: source receipt, drawings, prompts and the round loop (#412).

A duel reads a doodle receipt and plays its rows in order. Each round
renders the drawing to a PNG in the session directory, asks the person for a
pick and a confidence, then shows the answer, the person's pick and the
model's pick. The drawings come from the Quick, Draw! cache, or from Google's
public bucket on a cache miss. No round calls a model.

Attributes:
    SOURCE_ROW_KEYS (tuple[str, ...]): Fields every source receipt row holds.
    PICK_PROMPT (str): Prompt for the pick.
    CONFIDENCE_PROMPT (str): Prompt for the confidence.
    load_source_receipt (callable): Read and check a doodle receipt.
    doodles_for_rows (callable): Drawings of the rounds, by key id.
    play_duel (callable): Play the rounds and return the round records.
    interactive_answers (callable): Answers typed by a person.
    file_answers (callable): Answers read from an answers file.

Examples:
    ```python
    from typevet_evals.doodle_duel import file_answers, play_duel

    records = play_duel(
        rows,
        doodles,
        categories,
        file_answers(answers),
        session_dir=Path("duel-rounds"),
        write=print,
    )
    ```

See Also:
    - [typevet_evals.doodle_duel.duel][]: scores, bins and the receipt
    - [typevet_evals.datasets.quickdraw][]: Quick, Draw! loader and cache
    - [typevet_evals.cli.doodle_duel_play][]: the duel command
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import httpx

from typevet_evals.datasets.quickdraw import Doodle, fetch_doodles
from typevet_evals.doodle_duel.duel import (
    CONFIDENCE_MAX,
    CONFIDENCE_MIN,
    DuelAnswer,
    parse_confidence,
    parse_label,
    score_round,
)
from typevet_evals.doodle_duel.render import render_strokes
from typevet_evals.doodle_duel.runner import RECEIPT_ISSUE

SOURCE_ROW_KEYS: Final[tuple[str, ...]] = (
    "key_id",
    "true_label",
    "chosen_label",
    "probabilities",
    "correct",
)
PICK_PROMPT: Final[str] = "Your pick (number or name): "
CONFIDENCE_PROMPT: Final[str] = (
    f"How sure are you, {CONFIDENCE_MIN} to {CONFIDENCE_MAX}? "
)
_PERCENT: Final[int] = 100

Answer = Callable[[int, str], DuelAnswer]


def _check_pins(receipt: Mapping[str, Any]) -> list[str]:
    pins = receipt.get("pins")
    pins = pins if isinstance(pins, dict) else {}
    categories = pins.get("categories")
    if not categories or type(pins.get("per_category")) is not int:
        msg = "source receipt pins need categories and per_category"
        raise ValueError(msg)
    return [str(name) for name in categories]


def _check_row(index: int, row: object, categories: list[str]) -> None:
    if not isinstance(row, dict) or any(k not in row for k in SOURCE_ROW_KEYS):
        msg = f"source row {index} needs {list(SOURCE_ROW_KEYS)}"
        raise ValueError(msg)
    probabilities = row["probabilities"]
    if not isinstance(probabilities, dict) or set(probabilities) != set(categories):
        msg = f"source row {index} probabilities do not match the pinned categories"
        raise ValueError(msg)
    for field in ("chosen_label", "true_label"):
        if row[field] not in categories:
            msg = f"source row {index} {field} {row[field]} is not a category"
            raise ValueError(msg)


def load_source_receipt(path: Path) -> dict[str, Any]:
    """Read a doodle receipt and check that a duel can replay it.

    Args:
        path: A receipt that the doodle duel command wrote.

    Returns:
        The receipt body.

    Raises:
        ValueError: When the file is not JSON, ``issue`` is not 412, ``rows``
            is empty, a row lacks a field, the probabilities do not cover
            the pinned categories, or a label is not a category.
        OSError: When the file cannot be read.
    """
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(receipt, dict) or receipt.get("issue") != RECEIPT_ISSUE:
        msg = f"source receipt is not a doodle receipt for issue {RECEIPT_ISSUE}"
        raise ValueError(msg)
    categories = _check_pins(receipt)
    rows = receipt.get("rows")
    if not isinstance(rows, list) or not rows:
        msg = "source receipt rows are empty"
        raise ValueError(msg)
    for index, row in enumerate(rows):
        _check_row(index, row, categories)
    return receipt


def doodles_for_rows(
    receipt: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    cache_dir: Path | None = None,
    client: httpx.Client | None = None,
) -> dict[str, Doodle]:
    """Return the drawing of each round, by key id.

    For each true label the function reads the first ``per_category``
    doodles from the cache. A cache miss streams the same prefix from
    Google's public bucket and fills the cache; no model is called.

    Args:
        receipt: The source receipt; ``pins["per_category"]`` sets the prefix.
        rows: The rows to play.
        cache_dir: Explicit cache directory, as in ``resolve_cache_dir``.
        client: HTTP client for a miss; a new one when ``None``.

    Returns:
        The drawing of every row, keyed by key id.

    Raises:
        ValueError: When a download fails or a row's key id is not in the
            prefix of its category.
    """
    per_category = int(receipt["pins"]["per_category"])
    found: dict[str, Doodle] = {}
    for word in dict.fromkeys(str(row["true_label"]) for row in rows):
        try:
            doodles = fetch_doodles(
                word, per_category, cache_dir=cache_dir, client=client
            )
        except httpx.HTTPError as exc:
            msg = f"download of {word} failed: {exc}"
            raise ValueError(msg) from exc
        found.update({d.key_id: d for d in doodles})
    for row in rows:
        if row["key_id"] not in found:
            msg = (
                f"key_id {row['key_id']} ({row['true_label']}) "
                "not in the Quick, Draw! cache"
            )
            raise ValueError(msg)
    return {str(row["key_id"]): found[row["key_id"]] for row in rows}


def _mark(correct: object) -> str:
    return "right" if correct else "wrong"


def _reveal(record: Mapping[str, Any]) -> str:
    player = round(_PERCENT * record["player_confidence"])
    model = round(_PERCENT * record["model_confidence"])
    return (
        f"Answer: {record['true_label']}. "
        f"You: {record['player_label']} at {player}% "
        f"({_mark(record['player_correct'])}). "
        f"Model: {record['model_label']} at {model}% "
        f"({_mark(record['model_correct'])})."
    )


def play_duel(
    rows: Sequence[Mapping[str, Any]],
    doodles: Mapping[str, Doodle],
    categories: Sequence[str],
    answer: Answer,
    *,
    session_dir: Path,
    write: Callable[[str], object],
) -> list[dict[str, Any]]:
    """Play each row as one round and return the round records.

    Args:
        rows: Source receipt rows, in round order.
        doodles: Drawing of each row, by key id.
        categories: Every category of the question.
        answer: ``answer(round_number, key_id)`` gives the person's answer.
        session_dir: Directory for ``round-001.png`` and the next files.
        write: Prints one line of output.

    Returns:
        One record per round, as ``score_round`` builds it.
    """
    session_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for number, row in enumerate(rows, 1):
        key_id = str(row["key_id"])
        png = session_dir / f"round-{number:03d}.png"
        png.write_bytes(render_strokes(doodles[key_id].strokes))
        write(f"Round {number} of {len(rows)}: open {png}")
        record = score_round(number, row, answer(number, key_id), categories)
        write(_reveal(record))
        records.append(record)
    return records


def _ask[T](
    read_line: Callable[[str], str],
    write: Callable[[str], object],
    prompt: str,
    parse: Callable[[str], T],
) -> T:
    while True:
        try:
            return parse(read_line(prompt))
        except ValueError as exc:
            write(str(exc))


def interactive_answers(
    read_line: Callable[[str], str],
    write: Callable[[str], object],
    categories: Sequence[str],
) -> Answer:
    """Return an answer source that asks a person each round.

    A pick or confidence that does not parse prints the reason and asks
    again. End of input raises ``EOFError``, which ends the duel.

    Args:
        read_line: Shows a prompt and returns one typed line, as ``input``.
        write: Prints one line of output.
        categories: Categories, shown as a numbered list.

    Returns:
        ``answer(round_number, key_id)``.
    """

    def _typed(_number: int, key_id: str) -> DuelAnswer:
        for index, name in enumerate(categories, 1):
            write(f"  {index}. {name}")
        label = _ask(
            read_line, write, PICK_PROMPT, lambda t: parse_label(t, categories)
        )
        confidence = _ask(read_line, write, CONFIDENCE_PROMPT, parse_confidence)
        return DuelAnswer(key_id, label, confidence)

    return _typed


def file_answers(answers: Sequence[DuelAnswer]) -> Answer:
    """Return an answer source over answers checked by ``parse_answers``.

    Args:
        answers: One answer per round, in round order.

    Returns:
        ``answer(round_number, key_id)``; round 1 gives ``answers[0]``.
    """

    def _from_file(number: int, _key_id: str) -> DuelAnswer:
        return answers[number - 1]

    return _from_file

"""Quick, Draw! doodle ``Choice`` evaluation family (#412).

One recognized Quick, Draw! drawing becomes one typevet judgment with one
512 px PNG and one ``Choice`` over the curated categories. A run judges the
doodles in a seeded order, computes accuracy and builds a key-free receipt
whose rows carry the true label, the chosen label and every option
probability. A duel replays a receipt round by round against a person,
with no model call, and scores both players with the same pick Brier rule.
The package re-exports the names that callers outside the package use. The
probabilities are model confidence, not calibrated rates.

Attributes:
    BASELINE_BRIER (float): Brier of a player who always says 50%.
    CONFIDENCE_MAX (int): Highest confidence a person can give, in percent.
    CONFIDENCE_MIN (int): Lowest confidence a person can give, in percent.
    CONFIDENCE_PROMPT (str): Duel prompt for the confidence.
    DOODLE_CATEGORIES (tuple[str, ...]): Category names in option order.
    DOODLE_QUESTION (str): ``Choice`` question id.
    DOODLE_STATE (str): Fixed state text for every doodle.
    IMAGES_PER_JUDGMENT (int): Images sent with each judgment.
    LINE_WIDTH (int): Stroke width, in pixels.
    MARGIN (int): White margin on each side, in pixels.
    PICK_PROMPT (str): Duel prompt for the pick.
    RECEIPT_ISSUE (int): Issue number recorded in every receipt.
    RENDER_SIZE (int): Width and height of one render, in pixels.
    SOURCE_ROW_KEYS (tuple[str, ...]): Fields every source receipt row holds.
    DoodleOutcome (type): The answer and timing for one doodle.
    DoodleRequest (type): One one-image request for one doodle.
    DoodleRun (type): Outcomes of one run and its stopping failure.
    DuelAnswer (type): One person's pick and confidence for one round.
    build_doodle_receipt (callable): Run to a key-free receipt body.
    build_doodle_request (callable): Doodle and PNG bytes to a request.
    build_duel_receipt (callable): Round records to a duel receipt body.
    doodle_metrics (callable): Accuracy, per-category accuracy and mean
        chosen probability.
    doodle_questions (callable): New ``Choice`` over the categories.
    doodles_for_rows (callable): Drawings of the duel rounds, by key id.
    duel_scores (callable): Accuracy, mean Brier and mean confidence of
        both duel players.
    file_answers (callable): Duel answers read from an answers file.
    interactive_answers (callable): Duel answers typed by a person.
    judge_doodle (callable): Send one request to a judgment port.
    load_source_receipt (callable): Read and check a receipt for a duel.
    multiclass_brier (callable): Brier over every category for one row.
    parse_answers (callable): Answers file body to duel answers.
    parse_confidence (callable): Text to a confidence in percent.
    parse_label (callable): Number or name to a category.
    pick_brier (callable): Brier of one pick at one confidence.
    play_duel (callable): Play the duel rounds; return round records.
    reliability_rows (callable): Six reliability bins, 0-50 merged.
    render_strokes (callable): Strokes to 512 px PNG bytes.
    run_doodle_duel (callable): Judge each doodle once; stop at a failure.
    score_round (callable): One source row and one answer to a record.
    shuffle_rows (callable): Seeded run order.

Examples:
    ```python
    from typevet_evals.doodle_duel import build_doodle_request, render_strokes

    request = build_doodle_request(doodle, png=render_strokes(doodle.strokes))
    ```

See Also:
    - [typevet_evals.doodle_duel.categories][]: the curated categories
    - [typevet_evals.doodle_duel.render][]: the stroke renderer
    - [typevet_evals.doodle_duel.request][]: request builder
    - [typevet_evals.doodle_duel.runner][]: run, metrics and receipt
    - [typevet_evals.doodle_duel.duel][]: duel scores, bins and receipt
    - [typevet_evals.doodle_duel.duel_session][]: duel rounds and I/O
    - [typevet_evals.datasets.quickdraw][]: Quick, Draw! loader
"""

from __future__ import annotations

from typevet_evals.doodle_duel.categories import DOODLE_CATEGORIES
from typevet_evals.doodle_duel.duel import (
    BASELINE_BRIER,
    CONFIDENCE_MAX,
    CONFIDENCE_MIN,
    DuelAnswer,
    build_duel_receipt,
    duel_scores,
    multiclass_brier,
    parse_answers,
    parse_confidence,
    parse_label,
    pick_brier,
    reliability_rows,
    score_round,
)
from typevet_evals.doodle_duel.duel_session import (
    CONFIDENCE_PROMPT,
    PICK_PROMPT,
    SOURCE_ROW_KEYS,
    doodles_for_rows,
    file_answers,
    interactive_answers,
    load_source_receipt,
    play_duel,
)
from typevet_evals.doodle_duel.render import (
    LINE_WIDTH,
    MARGIN,
    RENDER_SIZE,
    render_strokes,
)
from typevet_evals.doodle_duel.request import (
    DOODLE_QUESTION,
    DOODLE_STATE,
    DoodleRequest,
    build_doodle_request,
    doodle_questions,
    judge_doodle,
)
from typevet_evals.doodle_duel.runner import (
    IMAGES_PER_JUDGMENT,
    RECEIPT_ISSUE,
    DoodleOutcome,
    DoodleRun,
    build_doodle_receipt,
    doodle_metrics,
    run_doodle_duel,
    shuffle_rows,
)

__all__ = [
    "BASELINE_BRIER",
    "CONFIDENCE_MAX",
    "CONFIDENCE_MIN",
    "CONFIDENCE_PROMPT",
    "DOODLE_CATEGORIES",
    "DOODLE_QUESTION",
    "DOODLE_STATE",
    "IMAGES_PER_JUDGMENT",
    "LINE_WIDTH",
    "MARGIN",
    "PICK_PROMPT",
    "RECEIPT_ISSUE",
    "RENDER_SIZE",
    "SOURCE_ROW_KEYS",
    "DoodleOutcome",
    "DoodleRequest",
    "DoodleRun",
    "DuelAnswer",
    "build_doodle_receipt",
    "build_doodle_request",
    "build_duel_receipt",
    "doodle_metrics",
    "doodle_questions",
    "doodles_for_rows",
    "duel_scores",
    "file_answers",
    "interactive_answers",
    "judge_doodle",
    "load_source_receipt",
    "multiclass_brier",
    "parse_answers",
    "parse_confidence",
    "parse_label",
    "pick_brier",
    "play_duel",
    "reliability_rows",
    "render_strokes",
    "run_doodle_duel",
    "score_round",
    "shuffle_rows",
]

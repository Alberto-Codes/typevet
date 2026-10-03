"""Quick, Draw! doodle ``Choice`` evaluation family (#412).

One recognized Quick, Draw! drawing becomes one typevet judgment with one
512 px PNG and one ``Choice`` over the curated categories. A run judges the
doodles in a seeded order, computes accuracy and builds a key-free receipt
whose rows carry the true label, the chosen label and every option
probability. The package re-exports the names that callers outside the
package use. The probabilities are model confidence, not calibrated rates.

Attributes:
    DOODLE_CATEGORIES (tuple[str, ...]): Category names in option order.
    DOODLE_QUESTION (str): ``Choice`` question id.
    DOODLE_STATE (str): Fixed state text for every doodle.
    IMAGES_PER_JUDGMENT (int): Images sent with each judgment.
    LINE_WIDTH (int): Stroke width, in pixels.
    MARGIN (int): White margin on each side, in pixels.
    RECEIPT_ISSUE (int): Issue number recorded in every receipt.
    RENDER_SIZE (int): Width and height of one render, in pixels.
    DoodleOutcome (type): The answer and timing for one doodle.
    DoodleRequest (type): One one-image request for one doodle.
    DoodleRun (type): Outcomes of one run and its stopping failure.
    build_doodle_receipt (callable): Run to a key-free receipt body.
    build_doodle_request (callable): Doodle and PNG bytes to a request.
    doodle_metrics (callable): Accuracy, per-category accuracy and mean
        chosen probability.
    doodle_questions (callable): New ``Choice`` over the categories.
    judge_doodle (callable): Send one request to a judgment port.
    render_strokes (callable): Strokes to 512 px PNG bytes.
    run_doodle_duel (callable): Judge each doodle once; stop at a failure.
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
    - [typevet_evals.datasets.quickdraw][]: Quick, Draw! loader
"""

from __future__ import annotations

from typevet_evals.doodle_duel.categories import DOODLE_CATEGORIES
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
    "DOODLE_CATEGORIES",
    "DOODLE_QUESTION",
    "DOODLE_STATE",
    "IMAGES_PER_JUDGMENT",
    "LINE_WIDTH",
    "MARGIN",
    "RECEIPT_ISSUE",
    "RENDER_SIZE",
    "DoodleOutcome",
    "DoodleRequest",
    "DoodleRun",
    "build_doodle_receipt",
    "build_doodle_request",
    "doodle_metrics",
    "doodle_questions",
    "judge_doodle",
    "render_strokes",
    "run_doodle_duel",
    "shuffle_rows",
]

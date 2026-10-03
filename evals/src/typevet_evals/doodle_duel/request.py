"""One-image doodle ``Choice`` request for one Quick, Draw! drawing (#412).

One doodle becomes one typevet judgment with one PNG and one ``Choice``
over the categories, in ``DOODLE_CATEGORIES`` order. The state text never
names the true label.

Attributes:
    DOODLE_QUESTION (str): ``Choice`` question id.
    DOODLE_STATE (str): Fixed state text for every doodle.

Examples:
    ```python
    from typevet_evals.doodle_duel.request import build_doodle_request, judge_doodle

    request = build_doodle_request(doodle, png=png)
    response = judge_doodle(port, request, "model-id")
    ```

See Also:
    - [typevet_evals.doodle_duel.render][]: the PNG bytes
    - [typevet.ports.judgment][]: the port that ``judge_doodle`` calls
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from typevet.domain import Choice, ImageInput, JudgmentResponse, Question
from typevet.ports import JudgmentPort
from typevet_evals.datasets.quickdraw import Doodle
from typevet_evals.doodle_duel.categories import DOODLE_CATEGORIES

DOODLE_QUESTION: Final[str] = "label"
DOODLE_STATE: Final[str] = (
    "The image is a quick hand drawing in black lines on a white background. "
    "A player drew it in 20 seconds or less to show one object."
)


def doodle_questions(
    categories: Sequence[str] = DOODLE_CATEGORIES,
) -> dict[str, Choice]:
    """Return a new ``Choice`` question over ``categories``.

    Args:
        categories: Option labels in prompt order.

    Returns:
        One ``Choice`` keyed by ``DOODLE_QUESTION``.
    """
    return {
        DOODLE_QUESTION: Choice(
            instructions="Which object does the drawing show? Choose one:",
            criteria={name: f"The drawing shows: {name}." for name in categories},
        )
    }


@dataclass(frozen=True, slots=True)
class DoodleRequest:
    """One judgment request for one doodle.

    Attributes:
        doodle (Doodle): Source drawing with its true label.
        state (str): State text for the judgment.
        questions (Mapping[str, Question]): The ``Choice`` by id.
        media (tuple[ImageInput]): The rendered PNG.

    Examples:
        ```python
        request = build_doodle_request(doodle, png=png)
        assert len(request.media) == 1
        ```
    """

    doodle: Doodle
    state: str
    questions: Mapping[str, Question]
    media: tuple[ImageInput]

    @property
    def key_id(self) -> str:
        """Return the drawing id for receipts.

        Returns:
            The dataset ``key_id``.
        """
        return self.doodle.key_id

    @property
    def true_label(self) -> str:
        """Return the category the player was asked to draw.

        Returns:
            The dataset ``word``.
        """
        return self.doodle.word


def build_doodle_request(
    doodle: Doodle,
    *,
    png: bytes,
    categories: Sequence[str] = DOODLE_CATEGORIES,
) -> DoodleRequest:
    """Build one one-image ``Choice`` request for ``doodle``.

    Args:
        doodle: Source drawing.
        png: PNG bytes of the rendered drawing.
        categories: Option labels in prompt order.

    Returns:
        The request with the PNG as its only image.

    Raises:
        typevet.domain.errors.ScoringValidationError: When ``png`` is empty.
    """
    return DoodleRequest(
        doodle=doodle,
        state=DOODLE_STATE,
        questions=doodle_questions(categories),
        media=(ImageInput(data=png, mime_type="image/png"),),
    )


def judge_doodle(
    port: JudgmentPort, request: DoodleRequest, model: str
) -> JudgmentResponse:
    """Send one doodle request to a judgment port.

    Args:
        port: Judgment port, for example a scoring judgment adapter.
        request: Request from :func:`build_doodle_request`.
        model: Backend model id or alias.

    Returns:
        Typed answer for the ``Choice``.
    """
    return port.judge(request.state, request.questions, model, media=request.media)

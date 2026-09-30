"""Two-image face-match judgment request for one LFW pair (#300).

One pair becomes one typevet judgment. Image 1 is the left face and image 2
is the right face. The judgment asks three questions: a ``Noul`` on the same
person, a ``Choice`` verdict and a 0 to 4 ``Score`` on face visibility in
image 2. The state and the questions never name the people in the pair.

Attributes:
    SAME_PERSON (str): ``Noul`` question id.
    VERDICT (str): ``Choice`` question id.
    FACE_VISIBILITY (str): ``Score`` question id.
    VERDICT_LABELS (tuple[str, ...]): ``Choice`` labels in prompt order.
    FACE_MATCH_STATE (str): Fixed state text for every pair.

Examples:
    ```python
    from typevet_evals.face_match.request import (
        build_face_match_request,
        judge_face_match,
    )

    request = build_face_match_request(pair, left_image=left, right_image=right)
    response = judge_face_match(port, request, "model-id")
    ```

See Also:
    - [typevet_evals.datasets.lfw][]: pairs, slice and image bytes
    - [typevet.ports.judgment][]: the port that ``judge_face_match`` calls
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

from typevet.domain import (
    Choice,
    ImageInput,
    JudgmentResponse,
    Noul,
    Question,
    Score,
)
from typevet.ports import JudgmentPort
from typevet_evals.datasets.lfw import LfwPair

SAME_PERSON: Final[str] = "same_person"
VERDICT: Final[str] = "verdict"
FACE_VISIBILITY: Final[str] = "face_visibility"
VERDICT_LABELS: Final[tuple[str, ...]] = (
    "same_person",
    "different_person",
    "cannot_tell",
)
FACE_MATCH_STATE: Final[str] = (
    "Image 1 and image 2 are two photographs. Each photograph can show more "
    "than one face. Compare the main face in image 1 with the main face in "
    "image 2."
)


def face_match_questions() -> dict[str, Question]:
    """Return new question objects for one face-match judgment.

    Returns:
        The ``Noul``, ``Choice`` and ``Score`` questions, in that order.
    """
    return {
        SAME_PERSON: Noul(
            instructions="Do image 1 and image 2 show the same person?",
            criteria={
                "true": "The main faces are the same person.",
                "false": "The main faces are two different people.",
            },
        ),
        VERDICT: Choice(
            instructions=(
                "Compare the main face in image 1 with the main face in "
                "image 2. Choose one:"
            ),
            criteria={
                "same_person": "The main faces are the same person.",
                "different_person": "The main faces are two different people.",
                "cannot_tell": "The images do not show enough of a face to decide.",
            },
        ),
        FACE_VISIBILITY: Score(
            instructions="Rate how clearly image 2 shows the main face:",
            criteria=[
                "No face is visible.",
                "A face is present but most of it is hidden or blurred.",
                "Part of the face is visible.",
                "Most of the face is visible.",
                "All of the face is clearly visible.",
            ],
        ),
    }


@dataclass(frozen=True, slots=True)
class FaceMatchRequest:
    """One judgment request for one LFW pair.

    Attributes:
        pair (LfwPair): Source pair with its gold label.
        state (str): State text for the judgment.
        questions (Mapping[str, Question]): The three questions by id.
        media (tuple[ImageInput, ImageInput]): Image 1, then image 2.

    Examples:
        ```python
        request = build_face_match_request(pair, left_image=a, right_image=b)
        assert len(request.media) == 2
        ```
    """

    pair: LfwPair
    state: str
    questions: Mapping[str, Question]
    media: tuple[ImageInput, ImageInput]

    @property
    def pair_id(self) -> str:
        """Return a stable id for receipts.

        Returns:
            ``<fold>:<left name>_<number>:<right name>_<number>``.
        """
        left, right = self.pair.left, self.pair.right
        return (
            f"{self.pair.fold}:{left.name}_{left.number:04d}:"
            f"{right.name}_{right.number:04d}"
        )

    @property
    def gold_same_person(self) -> bool:
        """Return the gold same-person label.

        Returns:
            ``True`` when the pair is a same-person pair.
        """
        return self.pair.same_person


def build_face_match_request(
    pair: LfwPair,
    *,
    left_image: bytes,
    right_image: bytes,
    mime_type: str = "image/jpeg",
) -> FaceMatchRequest:
    """Build one two-image judgment request for ``pair``.

    Args:
        pair: Source pair.
        left_image: Encoded bytes of the left face; becomes image 1.
        right_image: Encoded bytes of the right face; becomes image 2.
        mime_type: Mime type of both images. LFW archive images are JPEG.

    Returns:
        The request with both images in pair order.

    Raises:
        typevet.domain.errors.ScoringValidationError: When image bytes are
            empty or the mime type is not supported.
    """
    return FaceMatchRequest(
        pair=pair,
        state=FACE_MATCH_STATE,
        questions=face_match_questions(),
        media=(
            ImageInput(data=left_image, mime_type=mime_type),
            ImageInput(data=right_image, mime_type=mime_type),
        ),
    )


def judge_face_match(
    port: JudgmentPort, request: FaceMatchRequest, model: str
) -> JudgmentResponse:
    """Send one face-match request to a judgment port.

    Args:
        port: Judgment port, for example a scoring judgment adapter.
        request: Request from :func:`build_face_match_request`.
        model: Backend model id or alias.

    Returns:
        Typed answers for the three questions.
    """
    return port.judge(request.state, request.questions, model, media=request.media)

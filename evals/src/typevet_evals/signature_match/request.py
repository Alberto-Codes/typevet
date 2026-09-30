"""Two-image signature-match judgment request for one CEDAR pair (#318).

One pair becomes one typevet judgment. Image 1 is a genuine reference
signature and image 2 is the questioned signature. The judgment asks three
questions: a ``Noul`` on the same writer, a ``Choice`` verdict and a 0 to 4
``Score`` on the quality of image 2. The state and the questions never name
the writer, the file or the pair kind.

Attributes:
    SAME_WRITER (str): ``Noul`` question id.
    VERDICT (str): ``Choice`` question id.
    IMAGE_QUALITY (str): ``Score`` question id.
    VERDICT_LABELS (tuple[str, ...]): ``Choice`` labels in prompt order.
    SIGNATURE_MATCH_STATE (str): Fixed state text for every pair.

Examples:
    ```python
    from typevet_evals.signature_match.request import (
        build_signature_match_request,
        judge_signature_match,
    )

    request = build_signature_match_request(
        pair, reference_image=left, questioned_image=right
    )
    response = judge_signature_match(port, request, "model-id")
    ```

See Also:
    - [typevet_evals.datasets.cedar][]: pairs, slice and image bytes
    - [typevet_evals.face_match.request][]: the face-match request pattern
    - [typevet.ports.judgment][]: the port that ``judge_signature_match`` calls
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
from typevet_evals.datasets.cedar import CedarPair, PairKind

SAME_WRITER: Final[str] = "same_writer"
VERDICT: Final[str] = "verdict"
IMAGE_QUALITY: Final[str] = "image_quality"
VERDICT_LABELS: Final[tuple[str, ...]] = (
    "same_writer",
    "different_writer",
    "skilled_forgery_suspected",
    "cannot_tell",
)
SIGNATURE_MATCH_STATE: Final[str] = (
    "Image 1 and image 2 each show one handwritten signature. Image 1 is a "
    "reference signature. Compare the signature in image 2 with the signature "
    "in image 1."
)


def signature_match_questions() -> dict[str, Question]:
    """Return new question objects for one signature-match judgment.

    Returns:
        The ``Noul``, ``Choice`` and ``Score`` questions, in that order.
    """
    return {
        SAME_WRITER: Noul(
            instructions="Did the same person write image 1 and image 2?",
            criteria={
                "true": "The same person wrote both signatures.",
                "false": "Two different people wrote the signatures.",
            },
        ),
        VERDICT: Choice(
            instructions=(
                "Compare the signature in image 2 with the signature in "
                "image 1. Choose one:"
            ),
            criteria={
                "same_writer": "The same person wrote both signatures.",
                "different_writer": (
                    "Another person wrote image 2, with no attempt to copy image 1."
                ),
                "skilled_forgery_suspected": (
                    "Another person wrote image 2 as a copy of the signature "
                    "in image 1."
                ),
                "cannot_tell": "The images do not show enough to decide.",
            },
        ),
        IMAGE_QUALITY: Score(
            instructions="Rate how clearly image 2 shows the signature:",
            criteria=[
                "No signature is visible.",
                "A signature is present but most of it is unclear.",
                "Part of the signature is clear.",
                "Most of the signature is clear.",
                "All of the signature is clear.",
            ],
        ),
    }


@dataclass(frozen=True, slots=True)
class SignatureMatchRequest:
    """One judgment request for one CEDAR pair.

    Attributes:
        pair (CedarPair): Source pair with its gold kind.
        state (str): State text for the judgment.
        questions (Mapping[str, Question]): The three questions by id.
        media (tuple[ImageInput, ImageInput]): Image 1, then image 2.

    Examples:
        ```python
        request = build_signature_match_request(
            pair, reference_image=a, questioned_image=b
        )
        assert len(request.media) == 2
        ```
    """

    pair: CedarPair
    state: str
    questions: Mapping[str, Question]
    media: tuple[ImageInput, ImageInput]

    @property
    def pair_id(self) -> str:
        """Return a stable id for receipts.

        Returns:
            The pair id, for example
            ``genuine_skilled:original_12_3:forgeries_12_7``.
        """
        return self.pair.pair_id

    @property
    def gold_kind(self) -> PairKind:
        """Return the gold pair kind.

        Returns:
            How image 2 relates to image 1.
        """
        return self.pair.kind

    @property
    def gold_same_writer(self) -> bool:
        """Return the gold same-writer label.

        Returns:
            ``True`` when one writer wrote both signatures.
        """
        return self.pair.same_writer


def build_signature_match_request(
    pair: CedarPair,
    *,
    reference_image: bytes,
    questioned_image: bytes,
    mime_type: str = "image/png",
) -> SignatureMatchRequest:
    """Build one two-image judgment request for ``pair``.

    Args:
        pair: Source pair.
        reference_image: Encoded bytes of the genuine signature; becomes
            image 1.
        questioned_image: Encoded bytes of the questioned signature; becomes
            image 2.
        mime_type: Mime type of both images. CEDAR archive images are PNG.

    Returns:
        The request with both images in pair order.

    Raises:
        typevet.domain.errors.ScoringValidationError: When image bytes are
            empty or the mime type is not supported.
    """
    return SignatureMatchRequest(
        pair=pair,
        state=SIGNATURE_MATCH_STATE,
        questions=signature_match_questions(),
        media=(
            ImageInput(data=reference_image, mime_type=mime_type),
            ImageInput(data=questioned_image, mime_type=mime_type),
        ),
    )


def judge_signature_match(
    port: JudgmentPort, request: SignatureMatchRequest, model: str
) -> JudgmentResponse:
    """Send one signature-match request to a judgment port.

    Args:
        port: Judgment port, for example a scoring judgment adapter.
        request: Request from :func:`build_signature_match_request`.
        model: Backend model id or alias.

    Returns:
        Typed answers for the three questions.
    """
    return port.judge(request.state, request.questions, model, media=request.media)

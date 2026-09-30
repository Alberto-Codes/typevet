"""One-image check-versus-register judgment request (#315).

One case becomes one typevet judgment. The state holds the register row as
text; the media holds one check image. The judgment asks four questions: a
``Noul`` on the payee, a ``Noul`` on the amounts, a six-label ``Choice``
verdict and a 0 to 4 ``Score`` on legibility. The state never holds the
printed check face, so a changed payee shows only in the image.
``check_match_slice`` builds the seeded requests and slice pins of the live
run (#349).

Attributes:
    PAYEE_MATCHES (str): ``Noul`` question id for the payee.
    AMOUNTS_MATCH (str): ``Noul`` question id for both amounts.
    VERDICT (str): ``Choice`` question id for the verdict.
    LEGIBILITY (str): ``Score`` question id for legibility.
    VERDICT_LABELS (tuple[str, ...]): ``Choice`` labels in prompt order.
    CHECK_MATCH_STATE (str): State template; ``{row}`` is the register row.

Examples:
    ```python
    from typevet_evals.check_match import (
        build_check_match_request,
        check_cases,
        judge_check_match,
        render_check,
    )

    case = check_cases()[0]
    request = build_check_match_request(case, image=render_check(case))
    response = judge_check_match(port, request, "model-id")
    ```

See Also:
    - [typevet_evals.face_match.request][]: the two-image pattern this reuses
    - [typevet.ports.judgment][]: the port that ``judge_check_match`` calls
"""

from __future__ import annotations

import hashlib
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
from typevet_evals.check_match.cases import (
    CheckCase,
    CheckVariant,
    check_cases,
    check_match_seed,
    generator_pins,
)
from typevet_evals.check_match.render import render_check

PAYEE_MATCHES: Final[str] = "payee_matches"
AMOUNTS_MATCH: Final[str] = "amounts_match"
VERDICT: Final[str] = "verdict"
LEGIBILITY: Final[str] = "legibility"
VERDICT_LABELS: Final[tuple[str, ...]] = (
    "consistent",
    "payee_mismatch",
    "amount_mismatch",
    "date_mismatch",
    "unsigned",
    "cannot_tell",
)
CHECK_MATCH_STATE: Final[str] = (
    "The image shows one paper check. A check register row lists what the "
    "check must show: {row}. Compare the check in the image with the register "
    "row. The check shows the amount two times: as a number in the box and "
    "in words on the line below the payee."
)


def check_match_questions() -> dict[str, Question]:
    """Return new question objects for one check-versus-register judgment.

    Returns:
        The payee ``Noul``, the amounts ``Noul``, the ``Choice`` and the
        ``Score``, in that order.
    """
    return {
        PAYEE_MATCHES: Noul(
            instructions="Is the payee on the check the payee in the register row?",
            criteria={
                "true": "The check names the register payee.",
                "false": "The check names a different payee.",
            },
        ),
        AMOUNTS_MATCH: Noul(
            instructions=(
                "Do the number amount and the words amount on the check both "
                "equal the register amount?"
            ),
            criteria={
                "true": "Both amounts on the check equal the register amount.",
                "false": "One or both amounts on the check differ from the register.",
            },
        ),
        VERDICT: Choice(
            instructions="Compare the check with the register row. Choose one:",
            criteria={
                "consistent": "The check agrees with the register row and is signed.",
                "payee_mismatch": "The payee differs from the register row.",
                "amount_mismatch": "An amount differs from the register row.",
                "date_mismatch": "The date differs from the register row.",
                "unsigned": (
                    "There are no handwritten signature strokes on the signature "
                    "line. Printed marks such as VOID do not count."
                ),
                "cannot_tell": "The image is not clear enough to decide.",
            },
        ),
        LEGIBILITY: Score(
            instructions="Rate how clearly you can read the check text:",
            criteria=[
                "No text can be read.",
                "Only a few words can be read.",
                "Some fields can be read.",
                "Most fields can be read.",
                "All fields can be read clearly.",
            ],
        ),
    }


@dataclass(frozen=True, slots=True)
class CheckMatchRequest:
    """One judgment request for one check case.

    Attributes:
        case (CheckCase): Source case with its expected labels.
        state (str): State text with the register row.
        questions (Mapping[str, Question]): The four questions by id.
        media (tuple[ImageInput]): The one check image.

    Examples:
        ```python
        request = build_check_match_request(case, image=png)
        assert len(request.media) == 1
        ```
    """

    case: CheckCase
    state: str
    questions: Mapping[str, Question]
    media: tuple[ImageInput]

    @property
    def case_id(self) -> str:
        """Return the stable case id for receipts.

        Returns:
            ``r<row index>:<variant>``.
        """
        return self.case.case_id


def build_check_match_request(
    case: CheckCase, *, image: bytes, mime_type: str = "image/png"
) -> CheckMatchRequest:
    """Build one one-image judgment request for ``case``.

    Args:
        case: Source case.
        image: Encoded bytes of the rendered check.
        mime_type: Mime type of the image. Renders are PNG.

    Returns:
        The request with the register row in the state and one image.

    Raises:
        typevet.domain.errors.ScoringValidationError: When the image bytes
            are empty or the mime type is not supported.
    """
    return CheckMatchRequest(
        case=case,
        state=CHECK_MATCH_STATE.format(row=case.row.as_text()),
        questions=check_match_questions(),
        media=(ImageInput(data=image, mime_type=mime_type),),
    )


def judge_check_match(
    port: JudgmentPort, request: CheckMatchRequest, model: str
) -> JudgmentResponse:
    """Send one check-match request to a judgment port.

    Args:
        port: Judgment port, for example a scoring judgment adapter.
        request: Request from :func:`build_check_match_request`.
        model: Backend model id or alias.

    Returns:
        Typed answers for the four questions.
    """
    return port.judge(request.state, request.questions, model, media=request.media)


@dataclass(frozen=True, slots=True)
class CheckMatchSlice:
    """The seeded requests of one live run and the pins that name them.

    Attributes:
        seed (int): Generator seed from ``TYPEVET_CHECK_MATCH_SEED``.
        requests (tuple[CheckMatchRequest, ...]): One request per case.
        pins (dict[str, object]): ``generator_pins`` plus ``register_rows``,
            ``variants_per_row`` and ``slice_sha256``.

    Examples:
        ```python
        made = check_match_slice(os.environ, rows=20)
        made.pins["generator_seed"]
        ```
    """

    seed: int
    requests: tuple[CheckMatchRequest, ...]
    pins: dict[str, object]


def check_match_slice(environ: Mapping[str, str], rows: int) -> CheckMatchSlice:
    """Build the seeded requests and slice pins of the live run (#349).

    The live test calls this helper, so a unit test can prove offline that
    the seed reaches both the requests and the ``generator_seed`` pin.

    Args:
        environ: Process environment, or a mapping in its place.
        rows: Register rows in the slice.

    Returns:
        The seed, the requests in slice order and the slice pins.

    Raises:
        ValueError: When ``TYPEVET_CHECK_MATCH_SEED`` is not a seed.
    """
    seed = check_match_seed(environ)
    requests = tuple(
        build_check_match_request(case, image=render_check(case))
        for case in check_cases(seed, rows)
    )
    digest = hashlib.sha256()
    for request in requests:
        digest.update(request.case_id.encode())
        digest.update(hashlib.sha256(request.media[0].data).digest())
    pins: dict[str, object] = {
        **generator_pins(seed),
        "register_rows": rows,
        "variants_per_row": len(CheckVariant),
        "slice_sha256": digest.hexdigest(),
    }
    return CheckMatchSlice(seed, requests, pins)

"""Synthetic check-versus-register evaluation family (#303, #315).

A seeded generator makes 20 register rows and 7 check variants per row. One
case becomes one typevet judgment with the register row as text, one
rendered check image and four typed questions. The checks are generated
and not negotiable by construction. The repository stores no render.
Generated checks are not evidence about real checks.

Attributes:
    ACCOUNT_NUMBER (str): Account number that every check prints.
    AMOUNTS_MATCH (str): ``Noul`` question id for both amounts.
    BANK_NAME (str): Invented bank name printed on every check.
    BLUR_RANGE (tuple[float, float]): Blur radius range of low legibility.
    CHECK_MATCH_STATE (str): State template with the register row.
    CHECK_SIZE (tuple[int, int]): Width and height of one render.
    DEFAULT_SEED (int): Seed of the default slice.
    EXPECTED_LABELS (Mapping[CheckVariant, ExpectedLabels]): Amendment A1.
    LEGIBILITY (str): ``Score`` question id for legibility.
    MAX_CENTS (int): Largest amount that the words form supports.
    PAYEES (tuple[str, ...]): Invented payee names.
    PAYEE_MATCHES (str): ``Noul`` question id for the payee.
    ROW_COUNT (int): Register rows in the default slice.
    SPECIMEN_MARK (str): Mark across the check face.
    VERDICT (str): ``Choice`` question id for the verdict.
    VERDICT_LABELS (tuple[str, ...]): ``Choice`` labels in prompt order.
    VOID_MARK (str): Mark across the signature line.
    CheckCase (type): One row, one variant, its face and labels.
    CheckFace (type): What one rendered check shows.
    CheckMatchRequest (type): One one-image request for one case.
    CheckVariant (type): How the face differs from the register row.
    ExpectedLabels (type): Expected answers for one variant.
    RegisterRow (type): One row of the synthetic register.
    SeededDraws (type): Deterministic SHA-256 draw stream.
    aba_check_digit_ok (callable): ABA routing check-digit test.
    amount_in_words (callable): Cents to the check words form.
    build_check_match_request (callable): Case and image to a request.
    check_cases (callable): Every variant of every register row.
    check_match_questions (callable): New ``Noul``, ``Choice`` and ``Score``.
    dollars_in_words (callable): Whole dollars in lower-case words.
    judge_check_match (callable): Send one request to a judgment port.
    register_rows (callable): Seeded register rows.
    render_check (callable): One case to PNG bytes.
    render_check_image (callable): One case to a Pillow image.
    write_contact_sheet (callable): Grid of renders to a caller-given path.

Examples:
    ```python
    from typevet_evals.check_match import (
        build_check_match_request,
        check_cases,
        render_check,
    )

    case = check_cases()[0]
    request = build_check_match_request(case, image=render_check(case))
    ```

See Also:
    - [typevet_evals.check_match.cases][]: register, variants and labels
    - [typevet_evals.check_match.render][]: renders and contact sheet
    - [typevet_evals.check_match.request][]: request builder
    - [typevet_evals.check_match.words][]: written amount
    - [typevet_evals.face_match][]: the two-image family this reuses
"""

from __future__ import annotations

from typevet_evals.check_match.cases import (
    ACCOUNT_NUMBER,
    BLUR_RANGE,
    DEFAULT_SEED,
    EXPECTED_LABELS,
    PAYEES,
    ROW_COUNT,
    CheckCase,
    CheckFace,
    CheckVariant,
    ExpectedLabels,
    RegisterRow,
    SeededDraws,
    aba_check_digit_ok,
    check_cases,
    register_rows,
)
from typevet_evals.check_match.render import (
    BANK_NAME,
    CHECK_SIZE,
    SPECIMEN_MARK,
    VOID_MARK,
    render_check,
    render_check_image,
    write_contact_sheet,
)
from typevet_evals.check_match.request import (
    AMOUNTS_MATCH,
    CHECK_MATCH_STATE,
    LEGIBILITY,
    PAYEE_MATCHES,
    VERDICT,
    VERDICT_LABELS,
    CheckMatchRequest,
    build_check_match_request,
    check_match_questions,
    judge_check_match,
)
from typevet_evals.check_match.words import (
    MAX_CENTS,
    amount_in_words,
    dollars_in_words,
)

__all__ = [
    "ACCOUNT_NUMBER",
    "AMOUNTS_MATCH",
    "BANK_NAME",
    "BLUR_RANGE",
    "CHECK_MATCH_STATE",
    "CHECK_SIZE",
    "DEFAULT_SEED",
    "EXPECTED_LABELS",
    "LEGIBILITY",
    "MAX_CENTS",
    "PAYEES",
    "PAYEE_MATCHES",
    "ROW_COUNT",
    "SPECIMEN_MARK",
    "VERDICT",
    "VERDICT_LABELS",
    "VOID_MARK",
    "CheckCase",
    "CheckFace",
    "CheckMatchRequest",
    "CheckVariant",
    "ExpectedLabels",
    "RegisterRow",
    "SeededDraws",
    "aba_check_digit_ok",
    "amount_in_words",
    "build_check_match_request",
    "check_cases",
    "check_match_questions",
    "dollars_in_words",
    "judge_check_match",
    "register_rows",
    "render_check",
    "render_check_image",
    "write_contact_sheet",
]

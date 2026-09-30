"""Synthetic check-versus-register evaluation family (#303, #315, #316).

A seeded generator makes 20 register rows and 7 check variants per row. One
case becomes one typevet judgment with the register row as text, one
rendered check image and four typed questions. A run judges the cases,
computes the metrics and builds a key-free receipt. The checks are generated
and not negotiable by construction. The repository stores no render.
Generated checks are not evidence about real checks.

Attributes:
    ACCOUNT_NUMBER (str): Account number that every check prints.
    AMOUNTS_MATCH (str): ``Noul`` question id for both amounts.
    BANK_NAME (str): Invented bank name printed on every check.
    BLUR_RANGE (tuple[float, float]): Blur radius range of low legibility.
    CHECK_CONFIDENCE_NOTE (str): Honesty note stored with the metrics.
    CHECK_MATCH_STATE (str): State template with the register row.
    CHECK_SIZE (tuple[int, int]): Width and height of one render.
    DEFAULT_SEED (int): Seed of the default slice.
    EXPECTED_LABELS (Mapping[CheckVariant, ExpectedLabels]): Amendment A1.
    LEGIBILITY (str): ``Score`` question id for legibility.
    MAX_CENTS (int): Largest amount that the words form supports.
    NOUL_THRESHOLD (float): A ``Noul`` below this value says "mismatch".
    PAYEES (tuple[str, ...]): Invented payee names.
    PAYEE_MATCHES (str): ``Noul`` question id for the payee.
    RECEIPT_ISSUE (int): Issue number recorded in every receipt.
    ROW_COUNT (int): Register rows in the default slice.
    SEED_ENV (str): Environment variable that sets the live-run seed.
    SPECIMEN_MARK (str): Mark across the check face.
    VERDICT (str): ``Choice`` question id for the verdict.
    VERDICT_LABELS (tuple[str, ...]): ``Choice`` labels in prompt order.
    VOID_MARK (str): Mark across the signature line.
    CheckCase (type): One row, one variant, its face and labels.
    CheckFace (type): What one rendered check shows.
    CheckMatchOutcome (type): Typed answers and timing for one case.
    CheckMatchRequest (type): One one-image request for one case.
    CheckMatchRun (type): Outcomes of one run and its stopping failure.
    CheckVariant (type): How the face differs from the register row.
    ExpectedLabels (type): Expected answers for one variant.
    RegisterRow (type): One row of the synthetic register.
    SeededDraws (type): Deterministic SHA-256 draw stream.
    aba_check_digit_ok (callable): ABA routing check-digit test.
    accuracy_by_group (callable): Share of right answers per group.
    agreement_rate (callable): Share of agreeing cases, ``None`` skipped.
    amount_in_words (callable): Cents to the check words form.
    build_check_match_receipt (callable): Run to a key-free receipt body.
    build_check_match_request (callable): Case and image to a request.
    check_cases (callable): Every variant of every register row.
    check_match_metrics (callable): Every metric over the outcomes.
    check_match_questions (callable): New ``Noul``, ``Choice`` and ``Score``.
    check_match_seed (callable): Generator seed from the environment.
    check_outcome_from_response (callable): Typed answers to one outcome.
    class_key (callable): Accuracy class name of one label set.
    counts_for_false_clear (callable): Whether a case enters false clear.
    dollars_in_words (callable): Whole dollars in lower-case words.
    false_clear_rate (callable): Share of mismatch cases called consistent.
    generator_pins (callable): Receipt pins for the slice and its seed.
    judge_check_match (callable): Send one request to a judgment port.
    legibility_gap (callable): Clean minus low-legibility mean ``Score``.
    noul_choice_agreement (callable): Whether the ``Noul`` answers support
        the verdict.
    register_rows (callable): Seeded register rows.
    render_check (callable): One case to PNG bytes.
    render_check_image (callable): One case to a Pillow image.
    run_check_match (callable): Judge each case once; stop at a failure.
    score_summary (callable): ``Score`` mean and level counts.
    verdict_correct (callable): Whether a verdict is in the accepted set.
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
    - [typevet_evals.check_match.metrics][]: check-specific metric rules
    - [typevet_evals.check_match.runner][]: run, metrics and receipt
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
    SEED_ENV,
    CheckCase,
    CheckFace,
    CheckVariant,
    ExpectedLabels,
    RegisterRow,
    SeededDraws,
    aba_check_digit_ok,
    check_cases,
    check_match_seed,
    generator_pins,
    register_rows,
)
from typevet_evals.check_match.metrics import (
    NOUL_THRESHOLD,
    accuracy_by_group,
    agreement_rate,
    class_key,
    counts_for_false_clear,
    false_clear_rate,
    legibility_gap,
    noul_choice_agreement,
    score_summary,
    verdict_correct,
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
from typevet_evals.check_match.runner import (
    CHECK_CONFIDENCE_NOTE,
    RECEIPT_ISSUE,
    CheckMatchOutcome,
    CheckMatchRun,
    build_check_match_receipt,
    check_match_metrics,
    check_outcome_from_response,
    run_check_match,
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
    "CHECK_CONFIDENCE_NOTE",
    "CHECK_MATCH_STATE",
    "CHECK_SIZE",
    "DEFAULT_SEED",
    "EXPECTED_LABELS",
    "LEGIBILITY",
    "MAX_CENTS",
    "NOUL_THRESHOLD",
    "PAYEES",
    "PAYEE_MATCHES",
    "RECEIPT_ISSUE",
    "ROW_COUNT",
    "SEED_ENV",
    "SPECIMEN_MARK",
    "VERDICT",
    "VERDICT_LABELS",
    "VOID_MARK",
    "CheckCase",
    "CheckFace",
    "CheckMatchOutcome",
    "CheckMatchRequest",
    "CheckMatchRun",
    "CheckVariant",
    "ExpectedLabels",
    "RegisterRow",
    "SeededDraws",
    "aba_check_digit_ok",
    "accuracy_by_group",
    "agreement_rate",
    "amount_in_words",
    "build_check_match_receipt",
    "build_check_match_request",
    "check_cases",
    "check_match_metrics",
    "check_match_questions",
    "check_match_seed",
    "check_outcome_from_response",
    "class_key",
    "counts_for_false_clear",
    "dollars_in_words",
    "false_clear_rate",
    "generator_pins",
    "judge_check_match",
    "legibility_gap",
    "noul_choice_agreement",
    "register_rows",
    "render_check",
    "render_check_image",
    "run_check_match",
    "score_summary",
    "verdict_correct",
    "write_contact_sheet",
]

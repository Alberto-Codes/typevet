"""CEDAR signature-match evaluation family (#304, #318, #319).

One CEDAR pair becomes one typevet judgment with two images and three typed
questions. Image 1 is a genuine reference signature. Image 2 is another
genuine signature by the same writer, a skilled forgery or a genuine
signature by another writer. The runner judges a slice once, computes the
metrics and builds a key-free receipt. The package re-exports the names that
callers outside the package use. The ``same_writer`` probability is model
confidence, not a match percentage or a forensic score.

Attributes:
    SAME_WRITER (str): ``Noul`` question id for the same writer.
    VERDICT (str): ``Choice`` question id for the verdict.
    IMAGE_QUALITY (str): ``Score`` question id for the quality of image 2.
    VERDICT_LABELS (tuple[str, ...]): ``Choice`` labels in prompt order.
    SIGNATURE_MATCH_STATE (str): Fixed state text for every pair.
    ACCEPT_THRESHOLD (float): ``Noul`` value that accepts a pair.
    CONFIDENCE_NOTE (str): Honesty note stored with the metrics.
    RECEIPT_ISSUE (int): Issue number recorded in every receipt.
    Agreement (type): ``Noul`` and ``Choice`` agreement counts.
    FalseAccept (type): Skilled false-accept rates by two definitions.
    SignatureMatchOutcome (type): Typed answers for one pair.
    SignatureMatchRequest (type): One two-image request for one pair.
    SignatureMatchRun (type): Outcomes and the stopping failure of one run.
    build_signature_match_request (callable): Pair and image bytes to a
        request.
    build_signature_match_receipt (callable): Run to a key-free receipt.
    judge_signature_match (callable): Send one request to a judgment port.
    kind_accuracy (callable): Share of verdicts that name the pair kind.
    noul_accept_rate (callable): Share of ``Noul`` values that accept.
    noul_choice_agreement (callable): ``Noul`` side against the verdict.
    outcome_from_response (callable): Typed answers to one outcome.
    run_signature_match (callable): Judge each request and stop at a failure.
    signature_match_metrics (callable): Every metric over a run.
    signature_match_questions (callable): New ``Noul``, ``Choice`` and
        ``Score``.
    skilled_false_accept (callable): False-accept rates on skilled forgeries.
    verdict_accept_rate (callable): Share of ``same_writer`` verdicts.
    verdict_accuracy (callable): Same-writer accuracy of the verdicts.
    verdict_says_same_writer (callable): Verdict label to a same-writer side.

Examples:
    ```python
    from typevet_evals.signature_match import (
        build_signature_match_request,
        run_signature_match,
        signature_match_metrics,
    )

    request = build_signature_match_request(pair, reference_image=a, questioned_image=b)
    run = run_signature_match(port, [request], "model-id")
    metrics = signature_match_metrics(run.outcomes)
    ```

See Also:
    - [typevet_evals.signature_match.request][]: request builder
    - [typevet_evals.signature_match.metrics][]: signature metrics
    - [typevet_evals.signature_match.runner][]: run and receipt
    - [typevet_evals.datasets.cedar][]: CEDAR pairs loader
"""

from __future__ import annotations

from typevet_evals.signature_match.metrics import (
    ACCEPT_THRESHOLD,
    Agreement,
    FalseAccept,
    kind_accuracy,
    noul_accept_rate,
    noul_choice_agreement,
    skilled_false_accept,
    verdict_accept_rate,
    verdict_accuracy,
    verdict_says_same_writer,
)
from typevet_evals.signature_match.request import (
    IMAGE_QUALITY,
    SAME_WRITER,
    SIGNATURE_MATCH_STATE,
    VERDICT,
    VERDICT_LABELS,
    SignatureMatchRequest,
    build_signature_match_request,
    judge_signature_match,
    signature_match_questions,
)
from typevet_evals.signature_match.runner import (
    CONFIDENCE_NOTE,
    RECEIPT_ISSUE,
    SignatureMatchOutcome,
    SignatureMatchRun,
    build_signature_match_receipt,
    outcome_from_response,
    run_signature_match,
    signature_match_metrics,
)

__all__ = [
    "ACCEPT_THRESHOLD",
    "CONFIDENCE_NOTE",
    "IMAGE_QUALITY",
    "RECEIPT_ISSUE",
    "SAME_WRITER",
    "SIGNATURE_MATCH_STATE",
    "VERDICT",
    "VERDICT_LABELS",
    "Agreement",
    "FalseAccept",
    "SignatureMatchOutcome",
    "SignatureMatchRequest",
    "SignatureMatchRun",
    "build_signature_match_receipt",
    "build_signature_match_request",
    "judge_signature_match",
    "kind_accuracy",
    "noul_accept_rate",
    "noul_choice_agreement",
    "outcome_from_response",
    "run_signature_match",
    "signature_match_metrics",
    "signature_match_questions",
    "skilled_false_accept",
    "verdict_accept_rate",
    "verdict_accuracy",
    "verdict_says_same_writer",
]

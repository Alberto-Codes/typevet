"""LFW face-match evaluation family (#300, #301).

One LFW View 2 pair becomes one typevet judgment with two images and three
typed questions. A run judges a slice of pairs, computes the metrics and
builds a key-free receipt. The package re-exports the names that callers
outside the package use. The ``same_person`` probability is model
confidence, not a match percentage.

Attributes:
    FACE_MATCH_STATE (str): Fixed state text for every pair.
    FACE_VISIBILITY (str): ``Score`` question id for face visibility in image 2.
    SAME_PERSON (str): ``Noul`` question id for the same person.
    VERDICT (str): ``Choice`` question id for the verdict.
    VERDICT_LABELS (tuple[str, ...]): ``Choice`` labels in prompt order.
    CONFIDENCE_NOTE (str): Honesty note stored with the metrics.
    IMAGE_CONCURRENCY_ENV (str): Variable the live image runs read for the
        concurrency.
    RECEIPT_ISSUE (int): Issue number recorded in every receipt.
    VLLM_REVISION_ENV (str): Variable that names the served vLLM weights
        revision.
    FaceMatchOutcome (type): Typed answers and timing for one pair.
    FaceMatchRequest (type): One two-image request for one pair.
    FaceMatchRun (type): Outcomes of one run and its stopping failure.
    Judged (type): One judged request with its response and call time.
    JudgedBatch (type): Judged requests in slice order and the stop failure.
    JudgmentFailure (type): The backend failure that stopped a run.
    ReliabilityBin (type): One bin of the reliability table.
    build_face_match_request (callable): Pair and image bytes to a request.
    face_match_questions (callable): New ``Noul``, ``Choice`` and ``Score``.
    image_concurrency (callable): Read the run concurrency from the environment.
    judge_face_match (callable): Send one request to a judgment port.
    judge_in_order (callable): Judge requests in slice order, optionally
        several at one time; stop at the first failure.
    build_face_match_receipt (callable): Run to a key-free receipt body.
    cannot_tell_rate (callable): Share of ``cannot_tell`` verdicts.
    ensure_key_free (callable): Refuse receipt text with a key or auth header.
    expected_calibration_error (callable): ECE over equal-width bins.
    face_match_metrics (callable): Every metric over the outcomes.
    outcome_from_response (callable): Typed answers to one outcome.
    reliability_table (callable): Equal-width reliability bins.
    roc_auc (callable): Mann-Whitney ROC-AUC with average ranks.
    run_face_match (callable): Judge each pair once; stop at a failure.
    score_distribution (callable): ``Score`` level counts by gold label.
    served_weights_pins (callable): Weights revision pin for a vLLM run.
    verdict_accuracy (callable): Share of right verdicts.

Examples:
    ```python
    from typevet_evals.face_match import build_face_match_request

    request = build_face_match_request(pair, left_image=a, right_image=b)
    ```

See Also:
    - [typevet_evals.face_match.request][]: request builder
    - [typevet_evals.face_match.metrics][]: metric functions
    - [typevet_evals.face_match.runner][]: run and receipt
    - [typevet_evals.face_match.pool][]: shared ordered judgment loop
    - [typevet_evals.datasets.lfw][]: LFW View 2 loader
"""

from __future__ import annotations

from typevet_evals.face_match.metrics import (
    ReliabilityBin,
    cannot_tell_rate,
    expected_calibration_error,
    reliability_table,
    roc_auc,
    score_distribution,
    verdict_accuracy,
)
from typevet_evals.face_match.pool import (
    IMAGE_CONCURRENCY_ENV,
    Judged,
    JudgedBatch,
    JudgmentFailure,
    image_concurrency,
    judge_in_order,
)
from typevet_evals.face_match.request import (
    FACE_MATCH_STATE,
    FACE_VISIBILITY,
    SAME_PERSON,
    VERDICT,
    VERDICT_LABELS,
    FaceMatchRequest,
    build_face_match_request,
    face_match_questions,
    judge_face_match,
)
from typevet_evals.face_match.runner import (
    CONFIDENCE_NOTE,
    RECEIPT_ISSUE,
    VLLM_REVISION_ENV,
    FaceMatchOutcome,
    FaceMatchRun,
    build_face_match_receipt,
    ensure_key_free,
    face_match_metrics,
    outcome_from_response,
    run_face_match,
    served_weights_pins,
)

__all__ = [
    "CONFIDENCE_NOTE",
    "FACE_MATCH_STATE",
    "FACE_VISIBILITY",
    "IMAGE_CONCURRENCY_ENV",
    "RECEIPT_ISSUE",
    "SAME_PERSON",
    "VERDICT",
    "VERDICT_LABELS",
    "VLLM_REVISION_ENV",
    "FaceMatchOutcome",
    "FaceMatchRequest",
    "FaceMatchRun",
    "Judged",
    "JudgedBatch",
    "JudgmentFailure",
    "ReliabilityBin",
    "build_face_match_receipt",
    "build_face_match_request",
    "cannot_tell_rate",
    "ensure_key_free",
    "expected_calibration_error",
    "face_match_metrics",
    "face_match_questions",
    "image_concurrency",
    "judge_face_match",
    "judge_in_order",
    "outcome_from_response",
    "reliability_table",
    "roc_auc",
    "run_face_match",
    "score_distribution",
    "served_weights_pins",
    "verdict_accuracy",
]

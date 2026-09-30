"""CEDAR signature-match evaluation family (#304, #318).

One CEDAR pair becomes one typevet judgment with two images and three typed
questions. Image 1 is a genuine reference signature. Image 2 is another
genuine signature by the same writer, a skilled forgery or a genuine
signature by another writer. The package re-exports the names that callers
outside the package use. The ``same_writer`` probability is model
confidence, not a match percentage or a forensic score.

Attributes:
    SAME_WRITER (str): ``Noul`` question id for the same writer.
    VERDICT (str): ``Choice`` question id for the verdict.
    IMAGE_QUALITY (str): ``Score`` question id for the quality of image 2.
    VERDICT_LABELS (tuple[str, ...]): ``Choice`` labels in prompt order.
    SIGNATURE_MATCH_STATE (str): Fixed state text for every pair.
    SignatureMatchRequest (type): One two-image request for one pair.
    build_signature_match_request (callable): Pair and image bytes to a
        request.
    judge_signature_match (callable): Send one request to a judgment port.
    signature_match_questions (callable): New ``Noul``, ``Choice`` and
        ``Score``.

Examples:
    ```python
    from typevet_evals.signature_match import build_signature_match_request

    request = build_signature_match_request(pair, reference_image=a, questioned_image=b)
    ```

See Also:
    - [typevet_evals.signature_match.request][]: request builder
    - [typevet_evals.datasets.cedar][]: CEDAR pairs loader
"""

from __future__ import annotations

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

__all__ = [
    "IMAGE_QUALITY",
    "SAME_WRITER",
    "SIGNATURE_MATCH_STATE",
    "VERDICT",
    "VERDICT_LABELS",
    "SignatureMatchRequest",
    "build_signature_match_request",
    "judge_signature_match",
    "signature_match_questions",
]

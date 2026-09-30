"""LFW face-match evaluation family (#300).

One LFW View 2 pair becomes one typevet judgment with two images and three
typed questions. The package re-exports the names that callers outside the
package use.

Attributes:
    FACE_MATCH_STATE (str): Fixed state text for every pair.
    FACE_VISIBILITY (str): ``Score`` question id for face visibility in image 2.
    SAME_PERSON (str): ``Noul`` question id for the same person.
    VERDICT (str): ``Choice`` question id for the verdict.
    VERDICT_LABELS (tuple[str, ...]): ``Choice`` labels in prompt order.
    FaceMatchRequest (type): One two-image request for one pair.
    build_face_match_request (callable): Pair and image bytes to a request.
    face_match_questions (callable): New ``Noul``, ``Choice`` and ``Score``.
    judge_face_match (callable): Send one request to a judgment port.

Examples:
    ```python
    from typevet_evals.face_match import build_face_match_request

    request = build_face_match_request(pair, left_image=a, right_image=b)
    ```

See Also:
    - [typevet_evals.face_match.request][]: request builder
    - [typevet_evals.datasets.lfw][]: LFW View 2 loader
"""

from __future__ import annotations

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

__all__ = [
    "FACE_MATCH_STATE",
    "FACE_VISIBILITY",
    "SAME_PERSON",
    "VERDICT",
    "VERDICT_LABELS",
    "FaceMatchRequest",
    "build_face_match_request",
    "face_match_questions",
    "judge_face_match",
]

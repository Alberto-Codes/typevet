"""Typed questions and judgment payloads for the live demo web page.

The page asks one typed Choice about an expense claim and a receipt image, and
three typed questions (Noul, Choice and Score) about a customer message. This
module holds those questions and turns each typevet response into the page
result and the JSON receipt body. It makes no router call.

``demo.py`` imports this module from the same directory.

Examples:
    ```python
    results = text_results(resp, behind)
    ```

See Also:
    - [typevet.domain.JudgmentResponse][]: The typed response.
    - examples/live-demo/README.md: How to run the live demo.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from image_size import image_size

from typevet.domain import (
    Choice,
    ChoiceAnswer,
    ImageInput,
    JudgmentResponse,
    Noul,
    Question,
    Score,
)

HALF = 0.5

VERDICT_CRITERIA = {
    "supported": "The receipt image shows this total",
    "contradicted": "The receipt image shows a different total",
    "insufficient_evidence": "The image does not show enough to decide",
}
VERDICT_INSTRUCTIONS = "Look at the receipt image. Does it support the expense claim?"
VERDICT_QUESTIONS: dict[str, Question] = {
    "verdict": Choice(instructions=VERDICT_INSTRUCTIONS, criteria=VERDICT_CRITERIA)
}

FRAUD_CRITERIA = {
    "unauthorized_transaction": "A charge the customer did not make or approve",
    "duplicate_charge": "The same purchase was billed more than once",
    "phishing_or_scam": "Customer was tricked into paying or sharing data",
    "account_takeover": "Someone else gained control of the account",
    "not_fraud": "No fraud or billing error is described",
    "unclear": "Not enough information to decide",
}
URGENCY_LEVELS = [
    "none: no money at risk",
    "low: small issue, no money lost",
    "high: money already lost",
    "critical: ongoing loss, act now",
]
TEXT_QUESTIONS: dict[str, Question] = {
    "unauthorized": Noul(
        instructions="Does the customer report a transaction they did not authorize?",
        criteria={
            "true": "Yes, they report a charge they did not authorize",
            "false": "No unauthorized transaction is reported",
        },
    ),
    "fraud_type": Choice(
        instructions="Which type of issue does the customer report?",
        criteria=FRAUD_CRITERIA,
    ),
    "urgency": Score(
        instructions="How urgent is this customer's issue?",
        criteria=URGENCY_LEVELS,
    ),
}

Panel = Callable[[int, dict[str, float]], dict[str, Any]]


def _usage(resp: JudgmentResponse) -> dict[str, Any]:
    """Return the token usage of a response for the receipt.

    Args:
        resp: Typed judgment response.

    Returns:
        The input and output token counts.
    """
    return {
        "input_tokens": resp.usage.input_tokens,
        "output_tokens": resp.usage.output_tokens,
    }


def _choice_answer(c: ChoiceAnswer) -> dict[str, Any]:
    """Return a Choice answer for the receipt.

    Args:
        c: Typed Choice answer.

    Returns:
        The choice, the confidence and the probabilities.
    """
    return {
        "choice": c.choice,
        "confidence": c.confidence,
        "probabilities": c.probabilities,
    }


# ---------------------------------------------------------------- image
def image_desc(image: ImageInput, name: str, sha: str) -> str:
    """Return the text that replaces the media marker in the shown prompt.

    Args:
        image: Image that conditions the judgment.
        name: Display name of the image.
        sha: sha256 of the image bytes.

    Returns:
        The name, the size and the start of the sha256.
    """
    size = image_size(image.data, image.mime_type)
    dims = f"{size[0]}\u00d7{size[1]}" if size else "size unknown"
    return f"image: {name}, {dims}, sha256 {sha[:12]}\u2026"


def image_result(resp: JudgmentResponse, behind: dict[str, Any]) -> dict[str, Any]:
    """Return the page result of the verdict question.

    Args:
        resp: Typed judgment response.
        behind: Behind-the-scenes panel of the verdict question.

    Returns:
        The page result.
    """
    v = resp.choices["verdict"]
    return {
        "id": "verdict",
        "type": "choice",
        "title": VERDICT_INSTRUCTIONS,
        "options": [
            {"key": k, "label": k, "desc": d, "p": v.probabilities[k]}
            for k, d in VERDICT_CRITERIA.items()
        ],
        "winner": v.choice,
        "bts": behind,
    }


def image_request(
    claim: str, req: dict[str, Any], rec: dict[str, Any] | None, name: str
) -> dict[str, Any]:
    """Return the request part of an image receipt.

    Args:
        claim: Expense claim.
        req: Parsed request body.
        rec: Manifest receipt that the request names, or None.
        name: Display name of the image.

    Returns:
        The claim, the question, the image name and the manifest total.
    """
    return {
        "claim": claim,
        "question": VERDICT_INSTRUCTIONS,
        "criteria": VERDICT_CRITERIA,
        "image_name": name,
        "receipt_id": None if req.get("upload") else req.get("receipt_id"),
        "manifest_total": rec["annotated_total"]
        if rec and not req.get("upload")
        else None,
    }


def image_receipt(
    request: dict[str, Any],
    image: ImageInput,
    sha: str,
    resp: JudgmentResponse,
    elapsed: float,
    calls: int,
    behind: dict[str, Any],
) -> dict[str, Any]:
    """Return the body of an image receipt.

    Args:
        request: Request part from ``image_request``.
        image: Image that conditions the judgment.
        sha: sha256 of the image bytes.
        resp: Typed judgment response.
        elapsed: Seconds for the judgment.
        calls: Router HTTP calls for the judgment.
        behind: Behind-the-scenes panel of the verdict question.

    Returns:
        The receipt body.
    """
    return {
        "request": request,
        "image": {
            "sha256": sha,
            "mime_type": image.mime_type,
            "byte_count": len(image.data),
        },
        "answers": {"verdict": _choice_answer(resp.choices["verdict"])},
        "response_model": resp.model,
        "usage": _usage(resp),
        "timing": {"elapsed_s": elapsed, "router_http_calls": calls},
        "behind_the_scenes": {"verdict": behind},
    }


# ----------------------------------------------------------------- text
def text_behind(resp: JudgmentResponse, panel: Panel) -> dict[str, Any]:
    """Return the behind-the-scenes panel of each text question.

    Questions are scored in dict order, one ``/completion`` each.

    Args:
        resp: Typed judgment response.
        panel: Builds the panel for a scoring index and the answer
            probabilities.

    Returns:
        The panel of each question by name.
    """
    a = resp.nouls["unauthorized"]
    c = resp.choices["fraud_type"]
    s = resp.scores["urgency"]
    return {
        "unauthorized": panel(0, {"true": a.noul, "false": 1.0 - a.noul}),
        "fraud_type": panel(1, dict(c.probabilities)),
        "urgency": panel(2, {str(k): v for k, v in s.probabilities.items()}),
    }


def text_results(
    resp: JudgmentResponse, behind: dict[str, Any]
) -> list[dict[str, Any]]:
    """Return the page results of the three text questions.

    Args:
        resp: Typed judgment response.
        behind: Panels from ``text_behind``.

    Returns:
        One page result for each question.
    """
    a = resp.nouls["unauthorized"]
    c = resp.choices["fraud_type"]
    s = resp.scores["urgency"]
    top = max(s.probabilities, key=lambda k: s.probabilities[k])
    return [
        {
            "id": "unauthorized",
            "type": "yes/no",
            "title": TEXT_QUESTIONS["unauthorized"].instructions,
            "options": [
                {"key": "yes", "label": "yes", "p": a.noul},
                {"key": "no", "label": "no", "p": 1.0 - a.noul},
            ],
            "winner": "yes" if a.noul >= HALF else "no",
            "bts": behind["unauthorized"],
        },
        {
            "id": "fraud_type",
            "type": "pick one",
            "title": TEXT_QUESTIONS["fraud_type"].instructions,
            "options": [
                {"key": k, "label": k, "desc": d, "p": c.probabilities[k]}
                for k, d in FRAUD_CRITERIA.items()
            ],
            "winner": c.choice,
            "bts": behind["fraud_type"],
        },
        {
            "id": "urgency",
            "type": "score 0-3",
            "title": TEXT_QUESTIONS["urgency"].instructions,
            "options": [
                {
                    "key": str(lv),
                    "label": f"{lv}  {s.legend[lv]}",
                    "p": s.probabilities[lv],
                }
                for lv in sorted(s.probabilities)
            ],
            "winner": str(top),
            "expected": s.score,
            "max": len(URGENCY_LEVELS) - 1,
            "bts": behind["urgency"],
        },
    ]


def text_receipt(
    message: str,
    resp: JudgmentResponse,
    elapsed: float,
    calls: int,
    behind: dict[str, Any],
) -> dict[str, Any]:
    """Return the body of a text receipt.

    Args:
        message: Customer message.
        resp: Typed judgment response.
        elapsed: Seconds for the judgment.
        calls: Router HTTP calls for the judgment.
        behind: Panels from ``text_behind``.

    Returns:
        The receipt body.
    """
    s = resp.scores["urgency"]
    return {
        "request": {
            "message": message,
            "questions": {
                "unauthorized": {
                    "type": "noul",
                    "instructions": TEXT_QUESTIONS["unauthorized"].instructions,
                },
                "fraud_type": {"type": "choice", "criteria": FRAUD_CRITERIA},
                "urgency": {"type": "score", "levels": URGENCY_LEVELS},
            },
        },
        "answers": {
            "unauthorized": {"p_yes": resp.nouls["unauthorized"].noul},
            "fraud_type": _choice_answer(resp.choices["fraud_type"]),
            "urgency": {
                "expected_value": s.score,
                "confidence": s.confidence,
                "probabilities": {str(k): v for k, v in s.probabilities.items()},
            },
        },
        "response_model": resp.model,
        "usage": _usage(resp),
        "timing": {"elapsed_s": elapsed, "router_http_calls": calls},
        "behind_the_scenes": behind,
    }

"""Opt-in live CORD expense triage smoke with semantic metrics (#165).

Skips when the router or the multimodal model id is unavailable. **Fails** when
the router serves that id text-only or silently drops a receipt image, because
a text-only pass says nothing about reading a receipt.

Three modalities per receipt:

- ``text_only``: the claim statement, no image. Records the text prior.
- ``image_only``: the receipt with a claim that states no amount. Run once per
  receipt, because the text is identical for its three claims.
- ``combined``: the claim statement and the receipt. Semantic metrics come
  from this modality only.

Semantic metrics are recorded and soft-warned in the receipt, not gated. The
gates are request budget, typed-label validity and image attachment.

Examples:
    ```bash
    uv run pytest tests/live/test_cord_expense_smoke_live.py -m live -q
    ```

See Also:
    - [typevet.evaluation.datasets.cord_expense][]: cases, question and routing
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

import httpx
import pytest

from typevet.adapters.inbound.settings import load_llama_settings
from typevet.adapters.outbound.gemma import (
    ServedTemplateClass,
    classify_served_template,
)
from typevet.adapters.outbound.judgment_scoring import ScoringJudgmentAdapter
from typevet.adapters.outbound.llama_cpp_multimodal import fetch_media_capability
from typevet.adapters.outbound.llama_cpp_scoring import LlamaCppCandidateScoringAdapter
from typevet.domain.media import ImageInput
from typevet.evaluation.datasets.cord_expense import (
    INSUFFICIENT,
    INSUFFICIENT_EVIDENCE,
    LABEL_ORDER,
    SUPPORTED,
    VERDICTS,
    ExpenseCase,
    expense_question,
    load_expense_cases,
    route,
)
from typevet.evaluation.runner.live_gate import live_skip_reason

_LLAMA = load_llama_settings()
_N_VOCAB = 262144
FIXTURE_DIR = (
    Path(__file__).resolve().parents[1] / "fixtures" / "cord" / "expense_smoke"
)
_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "scratchpad" / "cord-expense"
_QUESTION = "expense"
_IMAGE_ONLY_STATE = (
    "Expense claim for this receipt: the claimed total is not stated. "
    "Decide whether the claim can be checked against the receipt."
)
_MAX_REQUESTS = 54
# One Gemma 3 image costs 256 prompt tokens. A silently dropped image grows the
# prompt by the marker text only, tens of tokens rather than hundreds (#155).
_MIN_IMAGE_TOKENS = 200
_CHANCE = 1 / len(LABEL_ORDER)
_ABSTAIN_FLOOR = 0.5


@pytest.fixture
def live_multimodal_model() -> str:
    """Skip unless the router catalog serves the multimodal model id.

    Returns:
        The multimodal model id from settings.
    """
    model = _LLAMA.multimodal_model
    reason = live_skip_reason(replace(_LLAMA, default_model=model))
    if reason is not None:
        pytest.skip(reason)
    return model


@pytest.fixture
def cases() -> tuple[ExpenseCase, ...]:
    """Load the 18 vendored CORD expense cases.

    Returns:
        Cases in manifest order.
    """
    text = (FIXTURE_DIR / "manifest.json").read_text(encoding="utf-8")
    return load_expense_cases(text)


def _receipt_image(case: ExpenseCase) -> ImageInput:
    data = (FIXTURE_DIR / case.image_file_name).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    assert digest == case.image_sha256, f"{case.image_file_name} digest drifted"
    return ImageInput(data=data, mime_type=case.image_mime_type)


def _tokenizer(client: httpx.Client, model: str):
    def tokenize_content(text: str) -> tuple[int, ...]:
        """Tokenize one control string without special tokens.

        Args:
            text: Control string to tokenize.

        Returns:
            Token ids the router reports.
        """
        payload = (
            client.post(
                "/tokenize",
                json={"model": model, "content": text, "add_special": False},
            )
            .raise_for_status()
            .json()
        )
        return tuple(payload["tokens"])

    return tokenize_content


def _served_template(client: httpx.Client, model: str) -> ServedTemplateClass:
    rendered = (
        client.post(
            "/apply-template",
            json={
                "model": model,
                "messages": [{"role": "user", "content": "hello"}],
                "add_generation_prompt": True,
            },
        )
        .raise_for_status()
        .json()["prompt"]
    )
    return classify_served_template(rendered)


def _judge(port, model: str, state: str, media: tuple[ImageInput, ...]):
    started = time.perf_counter()
    response = port.judge(state, {_QUESTION: expense_question()}, model, media=media)
    answer = response.choices[_QUESTION]
    return {
        "label": answer.choice,
        "probabilities": dict(answer.probabilities),
        "tokens_evaluated": response.usage.input_tokens,
        "seconds": round(time.perf_counter() - started, 3),
    }


def _run(port, model: str, cases: tuple[ExpenseCase, ...]):
    text_only: dict[str, dict[str, object]] = {}
    image_only: dict[str, dict[str, object]] = {}
    combined: dict[str, dict[str, object]] = {}
    for case in cases:
        image = _receipt_image(case)
        statement = case.model_inputs()["statement"]
        text_only[case.claim_id] = _judge(port, model, statement, ())
        if case.receipt_id not in image_only:
            image_only[case.receipt_id] = _judge(
                port, model, _IMAGE_ONLY_STATE, (image,)
            )
        combined[case.claim_id] = _judge(port, model, statement, (image,))
    return text_only, image_only, combined


def semantic_metrics(
    gold: dict[str, str], predicted: dict[str, str]
) -> dict[str, object]:
    """Score routed verdicts against gold verdicts for one modality.

    Args:
        gold: Gold verdict per claim id.
        predicted: Routed verdict per claim id.

    Returns:
        Accuracy, per-verdict recall, macro recall, confusion counts, the
        insufficient abstention rate and the false-supported count.
    """
    confusion = {g: dict.fromkeys(VERDICTS, 0) for g in VERDICTS}
    for claim_id, verdict in gold.items():
        confusion[verdict][predicted[claim_id]] += 1
    totals = Counter(gold.values())
    recall = {v: confusion[v][v] / totals[v] for v in VERDICTS if totals[v]}
    correct = sum(confusion[v][v] for v in VERDICTS)
    false_supported = sum(confusion[v][SUPPORTED] for v in VERDICTS if v != SUPPORTED)
    return {
        "n": len(gold),
        "accuracy": correct / len(gold),
        "recall": recall,
        "macro_recall": sum(recall.values()) / len(recall),
        "insufficient_abstention_rate": recall.get(INSUFFICIENT, 0.0),
        "false_supported": false_supported,
        "confusion_gold_by_predicted": confusion,
        "predicted_counts": dict(Counter(predicted.values())),
    }


def _soft_warnings(metrics: dict[str, object]) -> list[str]:
    notes: list[str] = []
    accuracy = metrics["accuracy"]
    abstention = metrics["insufficient_abstention_rate"]
    assert isinstance(accuracy, float)
    assert isinstance(abstention, float)
    if accuracy <= _CHANCE:
        notes.append(f"combined accuracy {accuracy:.3f} is at or below chance")
    if abstention < _ABSTAIN_FLOOR:
        notes.append(
            f"combined insufficient abstention {abstention:.3f} is below "
            f"{_ABSTAIN_FLOOR}; masked claims were not mostly abstained"
        )
    if metrics["false_supported"]:
        notes.append(
            f"{metrics['false_supported']} non-supported claims routed to supported"
        )
    predicted = metrics["predicted_counts"]
    assert isinstance(predicted, dict)
    if len(predicted) == 1:
        notes.append(f"combined answered one verdict for every claim: {predicted}")
    return notes


@pytest.mark.live
def test_cord_expense_triage_reads_the_receipt(
    live_multimodal_model: str, cases: tuple[ExpenseCase, ...]
) -> None:
    """Run three modalities on 18 claims and record combined semantic metrics."""
    assert len(cases) == 18
    settings = replace(_LLAMA, timeout=max(_LLAMA.timeout, 900.0))
    base = settings.base_url.rstrip("/")

    with httpx.Client(base_url=base, timeout=settings.timeout) as client:
        capability = fetch_media_capability(client, f"{base}/", live_multimodal_model)
        assert capability.vision, (
            f"{live_multimodal_model} reports text-only input modalities; "
            "a text-only router cannot prove receipt reading"
        )
        served = _served_template(client, live_multimodal_model)
        assert served is ServedTemplateClass.NATIVE_GEMMA3_TURN, served
        with LlamaCppCandidateScoringAdapter(
            base_url=base,
            timeout=settings.timeout,
            n_vocab=_N_VOCAB,
        ) as scoring:
            port = ScoringJudgmentAdapter(
                scoring,
                tokenize_content=_tokenizer(client, live_multimodal_model),
                served_template=served,
            )
            text_only, image_only, combined = _run(port, live_multimodal_model, cases)

    requests = len(text_only) + len(image_only) + len(combined)
    gold = {case.claim_id: case.expected_verdict for case in cases}
    metrics = {
        "combined": semantic_metrics(
            gold, {cid: route(str(row["label"])) for cid, row in combined.items()}
        ),
        "text_only": semantic_metrics(
            gold, {cid: route(str(row["label"])) for cid, row in text_only.items()}
        ),
    }
    image_only_abstained = sum(
        row["label"] == INSUFFICIENT_EVIDENCE for row in image_only.values()
    )
    image_moved = sum(
        combined[cid]["label"] != text_only[cid]["label"] for cid in combined
    )
    _write_receipt(
        {
            "issue": 165,
            "model": live_multimodal_model,
            "media_marker": capability.marker,
            "vision": capability.vision,
            "served_template": served.value,
            "requests": requests,
            "max_requests": _MAX_REQUESTS,
            "label_order": list(LABEL_ORDER),
            "cases": [
                {
                    "claim_id": case.claim_id,
                    "receipt_id": case.receipt_id,
                    "expected_verdict": case.expected_verdict,
                    "provenance": case.provenance(),
                }
                for case in cases
            ],
            "text_only": text_only,
            "image_only": {"state": _IMAGE_ONLY_STATE, "rows": image_only},
            "combined": combined,
            "metrics": metrics,
            "image_only_insufficient_rate": image_only_abstained / len(image_only),
            "combined_label_differs_from_text_only": image_moved,
            "soft_warnings": _soft_warnings(metrics["combined"]),
            "limits": [
                "Smoke only; not a production readiness or calibration claim.",
                "One model, one router build, six receipts, eighteen claims.",
                "Semantic metrics are recorded and soft-warned, not gated.",
                (
                    "text_only uses the ChatML prefix and imaged rows use the "
                    "native Gemma 3 turn, so the token delta includes template text."
                ),
            ],
        }
    )

    assert requests <= _MAX_REQUESTS
    for modality in (text_only, image_only, combined):
        for row in modality.values():
            assert row["label"] in LABEL_ORDER
            probabilities = row["probabilities"]
            assert isinstance(probabilities, dict)
            assert set(probabilities) == set(LABEL_ORDER)
    _assert_attachment(text_only, image_only, combined, cases)


def _assert_attachment(text_only, image_only, combined, cases) -> None:
    """Fail loud on a silently dropped receipt image (#155)."""
    for case in cases:
        text_tokens = text_only[case.claim_id]["tokens_evaluated"]
        image_tokens = combined[case.claim_id]["tokens_evaluated"]
        assert isinstance(text_tokens, int), "router did not report tokens"
        assert isinstance(image_tokens, int), "router did not report tokens"
        assert image_tokens - text_tokens >= _MIN_IMAGE_TOKENS, (
            f"{case.claim_id} combined evaluated {image_tokens} prompt tokens "
            f"against a {text_tokens}-token text baseline; the image was not attached"
        )
    for receipt_id, row in image_only.items():
        tokens = row["tokens_evaluated"]
        assert isinstance(tokens, int), "router did not report tokens"
        assert tokens >= _MIN_IMAGE_TOKENS, (
            f"{receipt_id} image_only evaluated {tokens} prompt tokens; "
            "the image was not attached"
        )


def _write_receipt(receipt: dict[str, object]) -> None:
    """Write the live receipt JSON under the gitignored scratchpad."""
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (_OUTPUT_DIR / "receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )

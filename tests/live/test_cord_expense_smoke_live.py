"""Opt-in live CORD expense smoke with semantic metrics (#165, #185, #186).

Skips when the router or the multimodal model id is unavailable. **Fails** when
the router serves that id text-only or silently drops a receipt image, because
a text-only pass says nothing about reading a receipt. Native Gemma 3 or
Gemma 4 turns are accepted (#185). Receipts carry experiment identity (#186).

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
import os
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
from typevet.evaluation.cord_expense_smoke import assert_cord_expense_attachment
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
from typevet.evaluation.experiment_identity import (
    ExperimentIdentityRequest,
    PromptSpec,
    RunIdentityStart,
    RuntimeBuild,
    WorkingTreeState,
    begin_run_identity,
    capture_experiment_identity,
    capture_working_tree_at_run_start,
    cord_expense_receipt_path,
    read_baseline_commit,
)
from typevet.evaluation.runner.live_gate import live_skip_reason

_LLAMA = load_llama_settings()
_N_VOCAB = 262144
FIXTURE_DIR = (
    Path(__file__).resolve().parents[1] / "fixtures" / "cord" / "expense_smoke"
)
_REPO_ROOT = Path(__file__).resolve().parents[2]
_OUTPUT_DIR = _REPO_ROOT / "scratchpad" / "cord-expense"
_QUESTION = "expense"
_IMAGE_ONLY_STATE = (
    "Expense claim for this receipt: the claimed total is not stated. "
    "Decide whether the claim can be checked against the receipt."
)
_MAX_REQUESTS = 54
_CHANCE = 1 / len(LABEL_ORDER)
_ABSTAIN_FLOOR = 0.5


def _working_tree_at_run_start(repo_root: Path) -> WorkingTreeState:
    """Fingerprint the tree at run start without spawning git in library code.

    When ``TYPEVET_GIT_STATUS_PORCELAIN`` is set, fingerprint that text.
    Otherwise mark dirty with an explicit unknown path so receipts never claim
    a false-clean checkout.

    Args:
        repo_root: Repository root for this checkout.

    Returns:
        Working-tree fingerprint for experiment identity.
    """
    porcelain = os.environ.get("TYPEVET_GIT_STATUS_PORCELAIN")
    if porcelain is None:
        baseline = read_baseline_commit(repo_root)
        return WorkingTreeState(
            baseline,
            True,
            ("dirty-unknown-without-porcelain",),
            "unknown",
        )
    return capture_working_tree_at_run_start(repo_root, porcelain=porcelain)


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
    """Run three modalities on 18 claims and record combined semantic metrics.

    Accepts native Gemma 3 or Gemma 4 served templates (#185) and attaches
    experiment identity on the receipt (#186).
    """
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
        # #185: CORD may exercise Gemma 3 or Gemma 4 native turns.
        assert served in (
            ServedTemplateClass.NATIVE_GEMMA3_TURN,
            ServedTemplateClass.NATIVE_GEMMA4_TURN,
        ), served
        run_start = begin_run_identity(
            repo_root=_REPO_ROOT,
            runtime=_runtime_build(client, live_multimodal_model, served),
            working_tree=_working_tree_at_run_start(_REPO_ROOT),
        )
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
        run_start,
        {
            "issue": 165,
            **_experiment_identity_receipt(
                run_start=run_start,
                prompts=(_prompt_spec_from_expense_question(),),
                arm_call_counts={
                    "text_only": len(text_only),
                    "image_only": len(image_only),
                    "combined": len(combined),
                },
            ),
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
                    "Every row uses one native Gemma 3 or Gemma 4 turn, so the "
                    "token delta is the media marker and image payload only."
                ),
            ],
        },
    )

    assert requests <= _MAX_REQUESTS
    for modality in (text_only, image_only, combined):
        for row in modality.values():
            assert row["label"] in LABEL_ORDER
            probabilities = row["probabilities"]
            assert isinstance(probabilities, dict)
            assert set(probabilities) == set(LABEL_ORDER)
    _assert_attachment(live_multimodal_model, text_only, image_only, combined, cases)


def _assert_attachment(model_id, text_only, image_only, combined, cases) -> None:
    """Fail loud on a silently dropped receipt image (#155, #185).

    Raises:
        AssertionError: When token growth falls below the model-specific floor.
    """
    claim_ids = tuple(case.claim_id for case in cases)
    receipt_ids = tuple(dict.fromkeys(case.receipt_id for case in cases))
    try:
        assert_cord_expense_attachment(
            model_id=model_id,
            text_only=text_only,
            image_only=image_only,
            combined=combined,
            claim_ids=claim_ids,
            receipt_ids=receipt_ids,
        )
    except ValueError as exc:
        raise AssertionError(str(exc)) from exc


def _write_receipt(run_start: RunIdentityStart, receipt: dict[str, object]) -> None:
    """Write one immutable receipt JSON under the gitignored scratchpad."""
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = cord_expense_receipt_path(_OUTPUT_DIR, run_start.run_id)
    path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")


def _prompt_spec_from_expense_question() -> PromptSpec:
    question = expense_question()
    criteria = question.criteria or {}
    return PromptSpec(
        name="expense",
        label_order=LABEL_ORDER,
        instructions=str(question.instructions or ""),
        criteria={label: str(criteria[label]) for label in LABEL_ORDER},
    )


def _runtime_build(
    client: httpx.Client, model: str, served: ServedTemplateClass
) -> RuntimeBuild:
    build_info = "unknown"
    try:
        payload = client.get("/props").raise_for_status().json()
        if isinstance(payload.get("build_info"), str):
            build_info = payload["build_info"]
    except httpx.HTTPError:
        pass
    return RuntimeBuild(model, served.value, build_info)


def _experiment_identity_receipt(
    *,
    run_start: RunIdentityStart,
    prompts: tuple[PromptSpec, ...],
    arm_call_counts: dict[str, int],
) -> dict[str, object]:
    """Attach #186 experiment identity for the direct judgment path (no triage).

    Args:
        run_start: Identity captured before any scoring calls.
        prompts: Prompt specs included in the identity digest.
        arm_call_counts: Per-arm request counts for the receipt.

    Returns:
        Mapping with a single ``experiment_identity`` receipt field.
    """
    identity = capture_experiment_identity(
        ExperimentIdentityRequest(
            repo_root=_REPO_ROOT,
            prompts=prompts,
            code_paths={
                "cord_expense": _REPO_ROOT
                / "src/typevet/evaluation/datasets/cord_expense.py",
                "cord_expense_smoke_live": Path(__file__).resolve(),
            },
            fixture_paths={"expense_smoke_manifest": FIXTURE_DIR / "manifest.json"},
            runtime=run_start.runtime,
            arm_call_counts=arm_call_counts,
            working_tree=run_start.working_tree,
        ),
        run_id=run_start.run_id,
    )
    return {"experiment_identity": identity.to_receipt_mapping()}

"""Opt-in live PSAI image+question multimodal smoke (#154).

Skips when the router or the multimodal model id is unavailable. **Fails** when
the router serves that id text-only, because a text-only pass cannot prove
image conditioning.

The visual leg is the proof. Present, omitted and swapped carry byte-identical
text, so only the pixels differ. A polarity flip between present and swapped is
image conditioning; an omitted match is the model's prior and never scores.
"""

from __future__ import annotations

import json
import time
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
from typevet.evaluation.datasets.psai_vision import (
    FOX_FAMILY,
    NON_FOX_FAMILY,
    VISUAL_QUESTION_NAME,
    example_image_input,
    load_vision_smoke,
)
from typevet.evaluation.datasets.psai_vision_controls import (
    CONDITION_OMITTED,
    NOUL_THRESHOLD,
    annotation_questions,
    annotation_state,
    control_matrix,
    control_summary,
    noul_polarity,
    paired_image_ordering,
    question_provenance,
    semantic_hit,
    visual_question,
)
from typevet.evaluation.runner.live_gate import live_skip_reason

_LLAMA = load_llama_settings()
_N_VOCAB = 262144
FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "psai" / "vision_smoke"
_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "scratchpad" / "psai-vision"
# One Gemma 3 image costs 256 prompt tokens. A silently dropped image grows the
# prompt by the marker text only, tens of tokens rather than hundreds (#155).
_MIN_IMAGE_TOKENS = 200


@pytest.fixture
def live_multimodal_model() -> str:
    """Skip unless the router catalog serves the multimodal model id."""
    model = _LLAMA.multimodal_model
    reason = live_skip_reason(replace(_LLAMA, default_model=model))
    if reason is not None:
        pytest.skip(reason)
    return model


@pytest.fixture
def fixture_set():
    """Load the vendored vision smoke fixtures."""
    text = (FIXTURE_DIR / "manifest.json").read_text(encoding="utf-8")
    return load_vision_smoke(text)


def _read_image(file_name: str) -> bytes:
    return (FIXTURE_DIR / file_name).read_bytes()


def _tokenizer(client: httpx.Client, model: str):
    def tokenize_content(text: str) -> tuple[int, ...]:
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


def _image_for(control, fixture_set):
    if control.image_unique_data_id is None:
        return ()
    donor = next(
        example
        for example in fixture_set.examples
        if example.unique_data_id == control.image_unique_data_id
    )
    return (example_image_input(donor, _read_image),)


def _run_visual_controls(port, model: str, fixture_set, controls):
    """Score every control row and collect receipt fields."""
    rows: list[dict[str, object]] = []
    probabilities: dict[tuple[str, str], float] = {}
    for control in controls:
        started = time.perf_counter()
        response = port.judge(
            control.state,
            {VISUAL_QUESTION_NAME: visual_question()},
            model,
            media=_image_for(control, fixture_set),
        )
        elapsed = time.perf_counter() - started
        probability = response.nouls[VISUAL_QUESTION_NAME].noul
        probabilities[(control.unique_data_id, control.condition)] = probability
        rows.append(
            {
                "unique_data_id": control.unique_data_id,
                "visual_family": control.visual_family,
                "condition": control.condition,
                "image_unique_data_id": control.image_unique_data_id,
                "expected": control.expected,
                "probability": probability,
                "polarity": noul_polarity(probability),
                "counts_as_semantic_hit": control.counts_as_semantic_hit,
                "semantic_hit": semantic_hit(control, probability),
                "tokens_evaluated": response.usage.input_tokens,
                "seconds": round(elapsed, 3),
            }
        )
    return rows, probabilities


def _run_annotation_leg(port, model: str, fixture_set):
    """Score the Hub-backed questions present-only for typed validity."""
    rows: list[dict[str, object]] = []
    for example in fixture_set.examples:
        response = port.judge(
            annotation_state(example),
            annotation_questions(),
            model,
            media=(example_image_input(example, _read_image),),
        )
        expected = example.expected()
        category = response.choices["category"]
        login = response.nouls["requires_login"]
        rows.append(
            {
                "unique_data_id": example.unique_data_id,
                "category": {
                    "predicted": category.choice,
                    "expected": expected["category"],
                    "confidence": category.confidence,
                    "gold_match": category.choice == expected["category"],
                },
                "requires_login": {
                    "probability": login.noul,
                    "polarity": noul_polarity(login.noul),
                    "expected": expected["requires_login"],
                    "gold_match": noul_polarity(login.noul)
                    is expected["requires_login"],
                },
            }
        )
    return rows


@pytest.mark.live
def test_psai_screenshots_condition_the_visual_judgment(
    live_multimodal_model: str, fixture_set
) -> None:
    """Run the accepted control matrix and the annotation leg on real pixels."""
    assert len(fixture_set.by_family(FOX_FAMILY)) >= 2
    assert len(fixture_set.by_family(NON_FOX_FAMILY)) >= 2

    settings = replace(_LLAMA, timeout=max(_LLAMA.timeout, 900.0))
    base = settings.base_url.rstrip("/")
    controls = control_matrix(fixture_set.examples)

    with httpx.Client(base_url=base, timeout=settings.timeout) as client:
        capability = fetch_media_capability(client, f"{base}/", live_multimodal_model)
        assert capability.vision, (
            f"{live_multimodal_model} reports text-only input modalities; "
            "a text-only router cannot prove image conditioning"
        )
        with LlamaCppCandidateScoringAdapter(
            base_url=base,
            timeout=settings.timeout,
            n_vocab=_N_VOCAB,
        ) as scoring:
            port = ScoringJudgmentAdapter(
                scoring,
                tokenize_content=_tokenizer(client, live_multimodal_model),
                served_template=_served_template(client, live_multimodal_model),
            )
            visual_rows, probabilities = _run_visual_controls(
                port, live_multimodal_model, fixture_set, controls
            )
            annotation_rows = _run_annotation_leg(
                port, live_multimodal_model, fixture_set
            )

    summary = control_summary(controls, probabilities)
    pairs = paired_image_ordering(controls, probabilities)
    _write_receipt(
        live_multimodal_model,
        capability,
        fixture_set,
        visual_rows,
        annotation_rows,
        summary,
        pairs,
    )
    _assert_attachment(visual_rows)
    _assert_semantics(visual_rows, pairs)


def _assert_attachment(visual_rows: list[dict[str, object]]) -> None:
    """Fail loud on a silently dropped image (#155)."""
    omitted = {
        row["unique_data_id"]: row["tokens_evaluated"]
        for row in visual_rows
        if row["condition"] == CONDITION_OMITTED
    }
    for row in visual_rows:
        if row["condition"] == CONDITION_OMITTED:
            continue
        text_tokens = omitted[row["unique_data_id"]]
        image_tokens = row["tokens_evaluated"]
        assert isinstance(text_tokens, int)
        assert isinstance(image_tokens, int)
        assert image_tokens - text_tokens >= _MIN_IMAGE_TOKENS, (
            f"{row['unique_data_id']} {row['condition']} evaluated "
            f"{image_tokens} prompt tokens against a {text_tokens}-token "
            "text baseline; the image was not attached"
        )


def _assert_semantics(visual_rows: list[dict[str, object]], pairs) -> None:
    """Gate on the paired swap rule, not on an uncalibrated threshold.

    Every omitted row is recorded and never credited. The image-conditioning
    claim rests on each row's two imaged controls: identical text, opposite
    families, so the Fox screenshot must outscore the non-Fox one. Reading a
    single probability against ``NOUL_THRESHOLD`` is kept in the receipt as a
    diagnostic, because this model is not calibrated on this Noul.
    """
    for row in visual_rows:
        if row["condition"] != CONDITION_OMITTED:
            continue
        assert row["counts_as_semantic_hit"] is False
        assert row["semantic_hit"] is False
    for pair in pairs:
        assert pair.ordered, (
            f"{pair.unique_data_id}: the Fox screenshot "
            f"{pair.fox_image_id} scored {pair.fox_probability:.4f} against "
            f"{pair.non_fox_probability:.4f} for the non-Fox screenshot "
            f"{pair.non_fox_image_id} (margin {pair.margin:+.4f}); "
            "swapping the image did not move the judgment"
        )


def _write_receipt(
    model: str,
    capability,
    fixture_set,
    visual_rows: list[dict[str, object]],
    annotation_rows: list[dict[str, object]],
    summary: dict[str, int],
    pairs,
) -> None:
    """Write the live receipt JSON under the gitignored scratchpad."""
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    receipt = {
        "issue": 154,
        "model": model,
        "media_marker": capability.marker,
        "vision": capability.vision,
        "dataset_id": fixture_set.dataset_id,
        "license": fixture_set.license,
        "split": fixture_set.split,
        "question_provenance": question_provenance(),
        "gold_provenance": dict(fixture_set.gold_provenance),
        "rows": [example.provenance(fixture_set) for example in fixture_set.examples],
        "visual_controls": visual_rows,
        "annotation_present_only": annotation_rows,
        "paired_ordering": [
            {
                "unique_data_id": pair.unique_data_id,
                "fox_image_id": pair.fox_image_id,
                "fox_probability": pair.fox_probability,
                "non_fox_image_id": pair.non_fox_image_id,
                "non_fox_probability": pair.non_fox_probability,
                "margin": pair.margin,
                "ordered": pair.ordered,
            }
            for pair in pairs
        ],
        "threshold_diagnostic": {
            "threshold": NOUL_THRESHOLD,
            "note": (
                "Recorded, not gated. This model is not calibrated on this "
                "Noul, so an absolute cut is not a semantic accuracy claim."
            ),
            "summary": summary,
        },
        "limits": [
            "Valid structure and semantic controls only; no calibration claim.",
            "One model, one router build, five screenshots.",
            "An omitted image is recorded but never credited as a hit.",
            "The gate is the paired swap margin, not the 0.5 threshold.",
        ],
    }
    (_OUTPUT_DIR / "live_receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )

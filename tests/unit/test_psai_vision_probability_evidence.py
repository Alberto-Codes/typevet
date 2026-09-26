"""Offline PSAI vision Choice raw mass repair ([#180][i180]).

Examples:
    ```bash
    uv run pytest -q tests/unit/test_psai_vision_probability_evidence.py
    ```

See Also:
    - [typevet.evaluation.psai_vision_probability_evidence][]: mass helpers
    - [docs.reference.psai-vision-choice-probability-evidence][]: reporting

[i180]: https://github.com/Alberto-Codes/typevet/issues/180
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

from typevet.evaluation.psai_vision_probability_evidence import (
    ARTIFACT_VERSION,
    build_corrected_artifact,
    legacy_denominator_mass,
    normalized_confidence,
    raw_candidate_mass,
)

pytestmark = pytest.mark.unit

_FIXTURE_DIR = (
    Path(__file__).resolve().parents[1] / "fixtures" / "psai" / "vision_choice_evidence"
)


def test_raw_mass_from_independent_probabilities() -> None:
    """Raw mass equals the sum of known candidate probabilities."""
    p_true, p_false = 0.7, 0.05
    logprobs = (math.log(p_true), math.log(p_false))
    assert raw_candidate_mass(logprobs) == pytest.approx(p_true + p_false)


def test_normalized_confidence_is_quotient_not_mass() -> None:
    """Legacy stored denominator differs from true raw mass on low-mass rows."""
    logprobs = (-5.15266227722168, -6.921392440795898)
    labels = ("true", "false")
    probs = normalized_confidence(logprobs, labels)
    assert sum(probs.values()) == pytest.approx(1.0)
    raw_mass = raw_candidate_mass(logprobs)
    legacy = legacy_denominator_mass(logprobs)
    assert raw_mass < 0.02
    assert legacy == pytest.approx(1.170549, rel=1e-3)
    assert legacy != pytest.approx(raw_mass, rel=1e-3)


def test_low_mass_omitted_row_matches_acceptance_audit() -> None:
    """C01 omitted logprobs repair matches independent acceptance recomputation."""
    logprobs = (-5.15266227722168, -6.921392440795898)
    assert raw_candidate_mass(logprobs) == pytest.approx(0.006770, rel=1e-3)


def test_log_sum_exp_extreme_gap() -> None:
    """Stable log-sum-exp handles large negative gaps without overflow."""
    logprobs = (-1000.0, -0.001)
    mass = raw_candidate_mass(logprobs)
    assert math.isfinite(mass)
    assert mass == pytest.approx(math.exp(-0.001), rel=1e-6)


def test_corrected_fixture_links_sources_and_omitted_arms() -> None:
    """Tracked artifact pins sources and records four omitted non-credits."""
    corrected = json.loads((_FIXTURE_DIR / "corrected_v1.json").read_text())
    matrix = json.loads((_FIXTURE_DIR / "source_matrix_v1.json").read_text())
    repair = json.loads((_FIXTURE_DIR / "source_c10_repair_v1.json").read_text())
    matrix_sha = hashlib.sha256(
        (_FIXTURE_DIR / "source_matrix_v1.json").read_bytes()
    ).hexdigest()
    repair_sha = hashlib.sha256(
        (_FIXTURE_DIR / "source_c10_repair_v1.json").read_bytes()
    ).hexdigest()
    rebuilt = build_corrected_artifact(
        matrix_receipt=matrix,
        matrix_source_sha256=matrix_sha,
        c10_repair_receipt=repair,
        c10_repair_source_sha256=repair_sha,
    )
    assert corrected["artifact_version"] == ARTIFACT_VERSION
    assert corrected["sources"]["matrix_receipt_sha256"] == matrix_sha
    assert corrected["omitted_arms"]["count"] == 4
    assert corrected["omitted_arms"]["semantic_credit"] is False
    assert "not a pristine" in corrected["completeness_verdict"]
    assert rebuilt["matrix_outcomes"][0]["raw_mass"] == pytest.approx(
        corrected["matrix_outcomes"][0]["raw_mass"]
    )


def test_imputed_mass_does_not_change_normalized_probs_in_fixture() -> None:
    """Repair changes raw mass only; normalized confidence stays aligned."""
    corrected = json.loads((_FIXTURE_DIR / "corrected_v1.json").read_text())
    for row in corrected["matrix_outcomes"]:
        recorded = row["normalized_probs"]
        recomputed = row["normalized_confidence_recomputed"]
        assert recorded == pytest.approx(recomputed, rel=1e-6, abs=1e-9)

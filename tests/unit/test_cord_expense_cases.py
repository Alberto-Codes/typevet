"""Claim cases built from the CORD expense smoke manifest (#164)."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from typevet.evaluation.datasets.cord_expense import (
    LABEL_ORDER,
    ExpenseCase,
    load_expense_cases,
    route,
)

pytestmark = pytest.mark.unit

MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "cord"
    / "expense_smoke"
    / "manifest.json"
)
PROVENANCE_KEYS = {
    "dataset_id",
    "dataset_revision",
    "license",
    "split",
    "source_file",
    "source_file_sha256",
    "receipt_id",
    "row_idx",
    "image_id",
    "image_sha256",
    "claim_provenance",
}


def _raw() -> dict[str, Any]:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _gold_by_claim() -> dict[str, tuple[str, str]]:
    return {
        c["claim_id"]: (c["expected_verdict"], r["normalized_total"])
        for r in _raw()["receipts"]
        for c in r["claims"]
    }


def _cases() -> tuple[ExpenseCase, ...]:
    return load_expense_cases(MANIFEST.read_text(encoding="utf-8"))


def test_manifest_yields_eighteen_cases_in_manifest_order() -> None:
    cases = _cases()
    assert len(cases) == 18
    assert [c.claim_id for c in cases] == list(_gold_by_claim())


def test_each_case_expected_verdict_matches_manifest_gold() -> None:
    gold = _gold_by_claim()
    for case in _cases():
        assert case.expected_verdict == gold[case.claim_id][0]


def test_each_receipt_has_one_case_per_label() -> None:
    by_receipt: dict[str, list[str]] = {}
    for case in _cases():
        by_receipt.setdefault(case.receipt_id, []).append(case.expected_verdict)
    assert len(by_receipt) == 6
    expected = sorted(route(label) for label in LABEL_ORDER)
    assert all(sorted(v) == expected for v in by_receipt.values())


def test_every_case_carries_full_provenance() -> None:
    raw = _raw()
    for case in _cases():
        provenance = case.provenance()
        assert set(provenance) == PROVENANCE_KEYS
        assert all(value not in (None, "") for value in provenance.values())
        assert provenance["dataset_id"] == raw["dataset_id"]
        assert provenance["dataset_revision"] == raw["dataset_revision"]
        assert provenance["license"] == "CC-BY-4.0"
        assert provenance["claim_provenance"] in {"synthetic_claim", "synthetic_damage"}


def test_gold_never_reaches_model_inputs() -> None:
    gold = _gold_by_claim()
    for case in _cases():
        inputs = case.model_inputs()
        assert set(inputs) == {"statement"}
        verdict, total = gold[case.claim_id]
        rendered = json.dumps(inputs)
        assert verdict not in rendered
        for label in LABEL_ORDER:
            assert label not in rendered
        if case.expected_verdict != "supported":
            assert total not in rendered


def test_case_image_names_the_receipt_png() -> None:
    for case in _cases():
        assert case.image_file_name == f"{case.receipt_id}.png"
        assert case.image_mime_type == "image/png"
        assert len(case.image_sha256) == 64


def _mutated(edit: Any) -> str:
    raw = copy.deepcopy(_raw())
    edit(raw)
    return json.dumps(raw)


def _set_claim(field: str, value: Any, claim: int = 0) -> Any:
    def edit(raw: dict[str, Any]) -> None:
        raw["receipts"][0]["claims"][claim][field] = value

    return edit


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (_set_claim("expected_verdict", "contradicted"), "verdict"),
        (_set_claim("expected_verdict", "supported", claim=1), "verdict"),
        (_set_claim("claimed_amount", "80500", claim=2), "verdict"),
        (_set_claim("expected_verdict", "maybe"), "verdict"),
        (_set_claim("provenance", "manual"), "provenance"),
        (_set_claim("claim_id", "R01-C2"), "duplicate"),
        (
            lambda raw: raw["receipts"][0].__setitem__("normalized_total", "80501"),
            "normalized_total",
        ),
        (lambda raw: raw.__setitem__("manifest_version", 2), "manifest_version"),
        (lambda raw: raw.__setitem__("license", "MIT"), "license"),
        (lambda raw: raw.pop("dataset_revision"), "dataset_revision"),
        (lambda raw: raw.__setitem__("receipts", {}), "receipts"),
        (lambda raw: raw["receipts"].__setitem__(0, []), "object"),
    ],
)
def test_load_rejects_inconsistent_manifests(edit: Any, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        load_expense_cases(_mutated(edit))


def test_load_rejects_invalid_json() -> None:
    with pytest.raises(ValueError, match="JSON"):
        load_expense_cases("{not json")

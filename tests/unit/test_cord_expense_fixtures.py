"""Fixture integrity for the vendored CORD expense smoke (#163)."""

from __future__ import annotations

import copy
import hashlib
import json
import struct
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.unit

FIXTURE_DIR = (
    Path(__file__).resolve().parents[1] / "fixtures" / "cord" / "expense_smoke"
)
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

TOP_KEYS = frozenset(
    {
        "manifest_version",
        "issue",
        "dataset_id",
        "dataset_revision",
        "license",
        "split",
        "notes",
        "receipts",
    }
)
RECEIPT_KEYS = frozenset(
    {
        "receipt_id",
        "row_idx",
        "image_id",
        "image",
        "annotated_total",
        "normalized_total",
        "claims",
    }
)
IMAGE_KEYS = frozenset(
    {"file_name", "mime_type", "sha256", "byte_count", "width", "height"}
)
CLAIM_KEYS = frozenset(
    {"claim_id", "expected_verdict", "claimed_amount", "statement", "provenance"}
)
VERDICTS = ("supported", "contradicted", "insufficient")
PROVENANCES = frozenset({"synthetic_claim", "synthetic_damage"})

PINNED_RECEIPTS: dict[str, tuple[int, str, Decimal]] = {
    "R01": (24, "80,500", Decimal(80500)),
    "R02": (25, "207,900", Decimal(207900)),
    "R03": (41, "2,352,460", Decimal(2352460)),
    "R04": (59, "664,329", Decimal(664329)),
    "R05": (79, "72.000", Decimal(72000)),
    "R06": (81, "276,000", Decimal(276000)),
}


def _raw_manifest() -> dict[str, Any]:
    return json.loads((FIXTURE_DIR / "manifest.json").read_text(encoding="utf-8"))


def _require_keys(obj: dict[str, Any], keys: frozenset[str], where: str) -> None:
    missing = keys - set(obj)
    if missing:
        msg = f"{where} missing keys: {sorted(missing)}"
        raise ValueError(msg)


def _decimal(text: object, where: str) -> Decimal:
    if isinstance(text, str):
        try:
            return Decimal(text)
        except InvalidOperation:
            pass
    msg = f"{where} is not a decimal string: {text!r}"
    raise ValueError(msg)


def _normalize_annotated(text: str) -> Decimal:
    """Drop CORD thousands separators (comma or dot) from a whole-unit total."""
    return Decimal(text.replace(",", "").replace(".", ""))


def _check_image(image: dict[str, Any], fixture_dir: Path, where: str) -> None:
    _require_keys(image, IMAGE_KEYS, where)
    data = (fixture_dir / image["file_name"]).read_bytes()
    if data[:8] != PNG_MAGIC or image["mime_type"] != "image/png":
        msg = f"{where} is not a PNG"
        raise ValueError(msg)
    if len(data) != image["byte_count"]:
        msg = f"{where} byte_count mismatch"
        raise ValueError(msg)
    if hashlib.sha256(data).hexdigest() != image["sha256"]:
        msg = f"{where} sha256 mismatch"
        raise ValueError(msg)
    if struct.unpack(">II", data[16:24]) != (image["width"], image["height"]):
        msg = f"{where} dimensions mismatch"
        raise ValueError(msg)


def _check_claims(receipt: dict[str, Any], total: Decimal, where: str) -> None:
    claims = receipt["claims"]
    verdicts = tuple(claim.get("expected_verdict") for claim in claims)
    if sorted(verdicts) != sorted(VERDICTS):
        msg = f"{where} claims must be one each of {VERDICTS}, got {verdicts}"
        raise ValueError(msg)
    for claim in claims:
        cwhere = f"{where} claim {claim.get('claim_id')}"
        _require_keys(claim, CLAIM_KEYS, cwhere)
        if claim["provenance"] not in PROVENANCES:
            msg = f"{cwhere} provenance {claim['provenance']!r} unknown"
            raise ValueError(msg)
        verdict = claim["expected_verdict"]
        amount = claim["claimed_amount"]
        if verdict == "supported":
            if _decimal(amount, cwhere) != total:
                msg = f"{cwhere} supported amount must equal the annotated total"
                raise ValueError(msg)
        elif verdict == "contradicted":
            if _decimal(amount, cwhere) == total:
                msg = f"{cwhere} contradicted amount must differ from the total"
                raise ValueError(msg)
        elif amount is not None and not claim.get("damaged"):
            msg = f"{cwhere} insufficient needs a null amount or a damage flag"
            raise ValueError(msg)
        if claim["provenance"] == "synthetic_damage" and not claim.get("damaged"):
            msg = f"{cwhere} synthetic_damage needs damaged=true"
            raise ValueError(msg)


def validate_manifest(raw: dict[str, Any], fixture_dir: Path = FIXTURE_DIR) -> None:
    """Raise ValueError when the manifest drifts from the #161 pins."""
    _require_keys(raw, TOP_KEYS, "manifest")
    if raw["license"] != "CC-BY-4.0" or raw["dataset_id"] != "naver-clova-ix/cord-v2":
        msg = "manifest license or dataset_id drifted"
        raise ValueError(msg)
    if raw["split"] != "validation":
        msg = "manifest split must be validation"
        raise ValueError(msg)
    ids = tuple(receipt.get("receipt_id") for receipt in raw["receipts"])
    if ids != tuple(PINNED_RECEIPTS):
        msg = f"receipt ids {ids} do not match the pins"
        raise ValueError(msg)
    for receipt in raw["receipts"]:
        where = f"receipt {receipt['receipt_id']}"
        _require_keys(receipt, RECEIPT_KEYS, where)
        row_idx, annotated, total = PINNED_RECEIPTS[receipt["receipt_id"]]
        if receipt["row_idx"] != row_idx or receipt["image_id"] != row_idx:
            msg = f"{where} row_idx or image_id drifted"
            raise ValueError(msg)
        if receipt["annotated_total"] != annotated:
            msg = f"{where} annotated_total drifted"
            raise ValueError(msg)
        normalized = _decimal(receipt["normalized_total"], where)
        if normalized != total or _normalize_annotated(annotated) != total:
            msg = f"{where} normalized_total drifted"
            raise ValueError(msg)
        _check_image(receipt["image"], fixture_dir, f"{where} image")
        _check_claims(receipt, total, where)


def test_vendored_manifest_matches_the_pins() -> None:
    validate_manifest(_raw_manifest())


def test_license_file_names_cc_by_and_the_source() -> None:
    text = (FIXTURE_DIR / "LICENSE.md").read_text(encoding="utf-8")
    assert "CC BY 4.0" in text
    assert "naver-clova-ix/cord-v2" in text


def test_every_receipt_has_three_distinct_claims() -> None:
    for receipt in _raw_manifest()["receipts"]:
        claims = receipt["claims"]
        by_verdict = {claim["expected_verdict"]: claim for claim in claims}
        assert set(by_verdict) == set(VERDICTS)
        total = Decimal(receipt["normalized_total"])
        supported = Decimal(by_verdict["supported"]["claimed_amount"])
        contradicted = Decimal(by_verdict["contradicted"]["claimed_amount"])
        assert supported == total
        assert contradicted != total
        insufficient = by_verdict["insufficient"]
        assert insufficient["claimed_amount"] is None or insufficient.get("damaged")
        assert len({claim["claim_id"] for claim in claims}) == 3


def _mutated(change: Any) -> dict[str, Any]:
    raw = copy.deepcopy(_raw_manifest())
    change(raw)
    return raw


def _set_claim(raw: dict[str, Any], verdict: str, key: str, value: object) -> None:
    for claim in raw["receipts"][0]["claims"]:
        if claim["expected_verdict"] == verdict:
            claim[key] = value


@pytest.mark.parametrize(
    ("change", "match"),
    [
        (lambda raw: raw.pop("license"), "missing"),
        (lambda raw: raw["receipts"][0].pop("normalized_total"), "missing"),
        (lambda raw: raw["receipts"][0]["image"].pop("sha256"), "missing"),
        (
            lambda raw: raw["receipts"][0]["image"].__setitem__("sha256", "0" * 64),
            "sha256",
        ),
        (lambda raw: raw["receipts"][0]["image"].__setitem__("byte_count", 1), "byte"),
        (lambda raw: raw["receipts"][0].__setitem__("row_idx", 23), "row_idx"),
        (
            lambda raw: raw["receipts"][0].__setitem__("annotated_total", "80.500"),
            "annotated_total",
        ),
        (
            lambda raw: raw["receipts"][0].__setitem__("normalized_total", "80501"),
            "normalized_total",
        ),
        (lambda raw: raw["receipts"].reverse(), "pins"),
        (lambda raw: raw.__setitem__("license", "MIT"), "license"),
        (
            lambda raw: _set_claim(raw, "supported", "claimed_amount", "1"),
            "supported",
        ),
        (
            lambda raw: _set_claim(
                raw,
                "contradicted",
                "claimed_amount",
                raw["receipts"][0]["normalized_total"],
            ),
            "contradicted",
        ),
        (
            lambda raw: (
                _set_claim(raw, "insufficient", "claimed_amount", "5"),
                _set_claim(raw, "insufficient", "damaged", False),
            ),
            "insufficient|synthetic_damage",
        ),
        (
            lambda raw: _set_claim(
                raw, "insufficient", "expected_verdict", "supported"
            ),
            "one each",
        ),
        (
            lambda raw: _set_claim(raw, "supported", "provenance", "annotation"),
            "provenance",
        ),
    ],
)
def test_validate_manifest_rejects_drift(change: Any, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        validate_manifest(_mutated(change))

"""The #343 post-hoc calibration analysis on the committed receipts.

The analysis reads only committed receipt files. It calls no model. The test
writes ``evals/fixtures/calibration/post_hoc_receipt.json`` when the file is
missing. After that it asserts that a fresh run gives the same bytes, so a
change to the fitters or the receipts shows up as a failure.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from typevet_evals.calibration import (
    DECISION_SERIES,
    Series,
    post_hoc_receipt,
    render_receipt,
)

pytestmark = pytest.mark.unit

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"
RECEIPT = FIXTURES / "calibration" / "post_hoc_receipt.json"
DIFRAUD = (
    "wording252_held_out_gemma_llama_cpp",
    "wording252_held_out_gemma_vllm",
    "wording252_held_out_jev",
    "wording_held_out_llama_cpp",
    "wording_held_out_vllm",
)
BACKENDS = ("llama_cpp", "vllm")
PAIR_FAMILIES = (
    (
        "signatures",
        "cedar/receipts/signature_match_{}.json",
        "same_writer_confidence",
        "gold_same_writer",
    ),
    (
        "faces",
        "lfw/receipts/face_match_{}_receipt.json",
        "same_person_confidence",
        "gold_same_person",
    ),
)


def _rows(source: str, key: str) -> list[dict[str, Any]]:
    return json.loads((FIXTURES / source).read_text(encoding="utf-8"))[key]


def _series(
    name: str, source: str, rows: list[dict[str, Any]], keys: tuple[str, str, str]
) -> Series:
    id_key, p_key, y_key = keys
    return Series(
        name,
        source,
        tuple(str(r[id_key]) for r in rows),
        tuple(float(r[p_key]) for r in rows),
        tuple(bool(r[y_key]) for r in rows),
    )


def load_series() -> tuple[Series, ...]:
    """Build the 16 series of the #343 contract from the committed receipts.

    DIFrauD columns hold the probability of label 1 (``is_scam``). A check
    row uses its largest verdict probability and its ``correct`` flag.
    """
    out: list[Series] = []
    for stem in DIFRAUD:
        source = f"difraud/receipts/{stem}.json"
        rows = _rows(source, "pairs")
        for column in ("seed", "evolved"):
            keys = ("record_id", column, "label")
            out.append(_series(f"difraud/{stem}/{column}", source, rows, keys))
    for backend in BACKENDS:
        source = f"checks/receipts/check_match_{backend}_receipt.json"
        rows = [
            {**c, "p": max(c["verdict_probabilities"].values())}
            for c in _rows(source, "cases")
        ]
        keys = ("case_id", "p", "correct")
        out.append(_series(f"checks/check_match_{backend}", source, rows, keys))
    for family, pattern, prob, gold in PAIR_FAMILIES:
        for backend in BACKENDS:
            source = pattern.format(backend)
            name = f"{family}/{Path(source).stem.removesuffix('_receipt')}"
            keys = ("pair_id", prob, gold)
            out.append(_series(name, source, _rows(source, "pairs"), keys))
    return tuple(out)


def test_sixteen_series_load_with_the_contract_sizes() -> None:
    series = load_series()
    sizes = {s.name: len(s.ids) for s in series}
    assert len(series) == 16
    assert sorted(set(sizes.values())) == [140, 158, 180, 200]
    assert set(DECISION_SERIES) <= set(sizes)
    for s in series:
        assert len(s.ids) == len(s.probabilities) == len(s.labels)
        assert all(0.0 <= p <= 1.0 for p in s.probabilities)


def test_post_hoc_receipt_is_reproducible() -> None:
    text = render_receipt(post_hoc_receipt(load_series(), FIXTURES))
    if not RECEIPT.exists():
        RECEIPT.parent.mkdir(parents=True, exist_ok=True)
        RECEIPT.write_text(text, encoding="utf-8")
    assert RECEIPT.read_text(encoding="utf-8") == text
    assert render_receipt(post_hoc_receipt(load_series(), FIXTURES)) == text
    receipt = json.loads(text)
    assert len(receipt["series"]) == 16
    for row in receipt["series"]:
        assert row["n_calibration"] + row["n_evaluation"] == row["n"]
        assert set(row["methods"]) == {"temperature", "platt", "isotonic"}
    assert receipt["decision"]["verdict"] in {"sufficient", "consider_fine_tuning"}
    assert len(receipt["transfer"]) == 2

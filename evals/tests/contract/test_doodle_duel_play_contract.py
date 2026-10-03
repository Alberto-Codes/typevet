"""Offline contract: the committed slice 1 receipt plays end to end (#412).

The test fills a temporary Quick, Draw! cache with synthetic strokes for
every key id the receipt pins, blocks the network and plays all rounds from
an answers file. It proves that the duel replays the receipt without a model
call and scores the model from the receipt alone. It says nothing about the
real drawings or about model quality.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from typevet_evals.cli import doodle_duel_play
from typevet_evals.datasets import quickdraw
from typevet_evals.doodle_duel import pick_brier

pytestmark = pytest.mark.contract

SOURCE = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "quickdraw"
    / "receipts"
    / "doodle_duel_llama_cpp_receipt.json"
)
MODEL_BINS = {
    "0-50": 0,
    "50-60": 2,
    "60-70": 1,
    "70-80": 4,
    "80-90": 1,
    "90-100": 112,
}


def _no_network(*args: object, **kwargs: object) -> None:
    msg = "the duel opened an HTTP client"
    raise AssertionError(msg)


def _fill_cache(cache: Path, pins: dict) -> None:
    cache.mkdir(parents=True)
    for word, ids in pins["key_ids"].items():
        lines = [
            json.dumps(
                {
                    "word": word,
                    "countrycode": "US",
                    "recognized": True,
                    "key_id": key_id,
                    "drawing": [[[0, 20 * i, 255], [10, 200, 25 * i]]],
                }
            )
            for i, key_id in enumerate(ids)
        ]
        path = quickdraw.cache_path(word, pins["per_category"], cache)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_committed_receipt_plays_end_to_end_in_file_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(quickdraw.httpx, "Client", _no_network)
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    cache = tmp_path / "cache"
    _fill_cache(cache, source["pins"])
    answers = [
        {"key_id": row["key_id"], "label": "moon", "confidence": 60}
        for row in source["rows"]
    ]
    answers_path = tmp_path / "answers.json"
    answers_path.write_text(json.dumps(answers), encoding="utf-8")
    duel = tmp_path / "out.json"

    code = doodle_duel_play.main(
        [
            "--source-receipt",
            str(SOURCE),
            "--duel-receipt",
            str(duel),
            "--answers",
            str(answers_path),
            "--cache-dir",
            str(cache),
        ]
    )

    assert code == 0
    text = duel.read_text(encoding="utf-8")
    receipt = json.loads(text)
    rows = receipt["rows"]
    assert receipt["rounds"] == len(rows) == 120
    assert [r["key_id"] for r in rows] == [r["key_id"] for r in source["rows"]]
    model = receipt["scores"]["model"]
    assert model["accuracy"] == source["metrics"]["accuracy"] == 0.925
    bins = {row["bin"]: row["count"] for row in receipt["reliability"]["model"]}
    assert bins == MODEL_BINS
    expected = [
        pick_brier(
            r["probabilities"][r["chosen_label"]],
            correct=r["chosen_label"] == r["true_label"],
        )
        for r in source["rows"]
    ]
    assert model["mean_brier"] == pytest.approx(sum(expected) / len(expected))
    assert receipt["source_receipt"]["model"] == source["model"]
    assert receipt["input_mode"] == "answers_file"
    assert "base64" not in text
    assert "PNG" not in text
    assert len(list((tmp_path / "out-rounds").glob("round-*.png"))) == 120
    out = capsys.readouterr().out
    assert "always-50% baseline brier 0.250" in out
    assert "model multi-class brier" in out
    assert "lower Brier is better" in out
    assert "90-100" in out

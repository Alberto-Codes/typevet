"""Unit tests for the doodle duel command on a prefilled cache (#412)."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from typevet_evals.cli import doodle_duel
from typevet_evals.datasets import quickdraw
from typevet_evals.doodle_duel import DOODLE_CATEGORIES

pytestmark = pytest.mark.unit

PER_CATEGORY = 5
REAL_CLIENT = httpx.Client
ROW_KEYS = {
    "key_id",
    "true_label",
    "chosen_label",
    "probabilities",
    "correct",
    "latency_seconds",
}


def _no_network(*args: object, **kwargs: object) -> None:
    msg = "the command opened an HTTP client on a cache hit"
    raise AssertionError(msg)


def _fill_cache(cache: Path) -> None:
    cache.mkdir(parents=True)
    for word in DOODLE_CATEGORIES:
        lines = [
            json.dumps(
                {
                    "word": word,
                    "countrycode": "US",
                    "recognized": True,
                    "key_id": f"{word}-{i}",
                    "drawing": [[[0, 40 * i, 255], [10, 200, 30 * i]]],
                }
            )
            for i in range(PER_CATEGORY)
        ]
        path = cache / f"{word}.first{PER_CATEGORY}.ndjson"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TYPEVET_BACKEND", "fake")
    monkeypatch.delenv("TYPEVET_FAKE__DISTRIBUTIONS", raising=False)
    monkeypatch.setattr(quickdraw.httpx, "Client", _no_network)


@pytest.mark.usefixtures("offline")
def test_cli_writes_receipt_from_cache_with_fake_backend(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cache = tmp_path / "cache"
    _fill_cache(cache)
    receipt_path = tmp_path / "out" / "receipt.json"

    code = doodle_duel.main(
        [
            "--receipt",
            str(receipt_path),
            "--per-category",
            str(PER_CATEGORY),
            "--cache-dir",
            str(cache),
        ]
    )

    assert code == 0
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    rows = receipt["rows"]
    assert len(rows) == PER_CATEGORY * len(DOODLE_CATEGORIES) == 120
    assert {row["true_label"] for row in rows} == set(DOODLE_CATEGORIES)
    for row in rows:
        assert set(row) == ROW_KEYS
        assert list(row["probabilities"]) == list(DOODLE_CATEGORIES)
        assert sum(row["probabilities"].values()) == pytest.approx(1.0)
    assert [row["key_id"] for row in rows] != sorted(row["key_id"] for row in rows)
    metrics = receipt["metrics"]
    assert metrics["count"] == 120
    assert set(metrics) == {
        "count",
        "accuracy",
        "per_category_accuracy",
        "mean_chosen_probability",
    }
    assert receipt["backend"] == "fake"
    assert receipt["stopped"] is None
    assert receipt["pins"]["credit"] == "Google Quick, Draw!"
    assert receipt["pins"]["per_category"] == PER_CATEGORY
    assert "PNG" not in receipt_path.read_text(encoding="utf-8")
    assert f"accuracy {metrics['accuracy']}" in capsys.readouterr().out


@pytest.mark.usefixtures("offline")
def test_cli_refuses_existing_receipt(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text("keep", encoding="utf-8")

    code = doodle_duel.main(
        ["--receipt", str(receipt_path), "--cache-dir", str(tmp_path / "empty")]
    )

    assert code == 1
    assert receipt_path.read_text(encoding="utf-8") == "keep"
    assert not (tmp_path / "empty").exists()
    assert "exists" in capsys.readouterr().err


@pytest.mark.usefixtures("offline")
def test_cli_refuses_duplicate_categories(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cache = tmp_path / "cache"
    _fill_cache(cache)
    receipt_path = tmp_path / "receipt.json"

    code = doodle_duel.main(
        [
            "--receipt",
            str(receipt_path),
            "--categories",
            "cat,cat,moon",
            "--cache-dir",
            str(cache),
        ]
    )

    assert code == 2
    assert not receipt_path.exists()
    assert "repeats: ['cat']" in capsys.readouterr().err


def _failing_client() -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        msg = "no route"
        raise httpx.ConnectError(msg, request=request)

    return REAL_CLIENT(transport=httpx.MockTransport(handler))


def test_cli_refuses_http_error_on_cache_miss(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("TYPEVET_BACKEND", "fake")
    monkeypatch.setattr(quickdraw.httpx, "Client", _failing_client)
    receipt_path = tmp_path / "receipt.json"

    code = doodle_duel.main(
        [
            "--receipt",
            str(receipt_path),
            "--categories",
            "cat,moon",
            "--per-category",
            "1",
            "--cache-dir",
            str(tmp_path / "empty"),
        ]
    )

    assert code == 1
    assert not receipt_path.exists()
    assert "refused:" in capsys.readouterr().err

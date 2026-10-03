"""Unit tests for the duel round loop, its inputs and its refusals (#412)."""

from __future__ import annotations

import io
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from typevet_evals.cli import doodle_duel_play
from typevet_evals.datasets import quickdraw
from typevet_evals.datasets.quickdraw import Doodle
from typevet_evals.doodle_duel import (
    DuelAnswer,
    doodles_for_rows,
    file_answers,
    interactive_answers,
    load_source_receipt,
    play_duel,
)

pytestmark = pytest.mark.unit

CATEGORIES = ("cat", "moon", "sun")
PER_CATEGORY = 2
KEY_IDS = {word: [f"{word}-{i}" for i in range(PER_CATEGORY)] for word in CATEGORIES}
ORDER = ["moon-1", "cat-0", "sun-1", "cat-1", "sun-0", "moon-0"]
CHOSEN = {"moon-1": ("sun", 0.7), "moon-0": ("moon", 0.95)}
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
REAL_CLIENT = httpx.Client


def _row(key_id: str) -> dict[str, Any]:
    word = key_id.partition("-")[0]
    chosen, p = CHOSEN.get(key_id, (word, 0.9))
    rest = (1.0 - p) / (len(CATEGORIES) - 1)
    return {
        "key_id": key_id,
        "true_label": word,
        "chosen_label": chosen,
        "probabilities": {c: (p if c == chosen else rest) for c in CATEGORIES},
        "correct": chosen == word,
        "latency_seconds": 0.5,
    }


def _receipt() -> dict[str, Any]:
    return {
        "issue": 412,
        "backend": "fake",
        "model": "fake-doodle",
        "pins": {
            "categories": list(CATEGORIES),
            "per_category": PER_CATEGORY,
            "key_ids": KEY_IDS,
        },
        "rows": [_row(key_id) for key_id in ORDER],
    }


def _write_receipt(path: Path, receipt: dict[str, Any] | None = None) -> Path:
    path.write_text(json.dumps(receipt or _receipt()), encoding="utf-8")
    return path


def _fill_cache(cache: Path, *, skip: str | None = None) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    for word, ids in KEY_IDS.items():
        lines = [
            json.dumps(
                {
                    "word": word,
                    "countrycode": "US",
                    "recognized": True,
                    "key_id": key_id if key_id != skip else "other",
                    "drawing": [[[0, 100, 255], [10, 200, 30]]],
                }
            )
            for key_id in ids
        ]
        path = quickdraw.cache_path(word, PER_CATEGORY, cache)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _answers(order: list[str] = ORDER) -> list[dict[str, Any]]:
    return [
        {"key_id": key_id, "label": key_id.partition("-")[0], "confidence": 80}
        for key_id in order
    ]


def _no_network(*args: object, **kwargs: object) -> None:
    msg = "an HTTP client was opened"
    raise AssertionError(msg)


def _failing_client() -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        msg = "no route"
        raise httpx.ConnectError(msg, request=request)

    return REAL_CLIENT(transport=httpx.MockTransport(handler))


@pytest.fixture
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(quickdraw.httpx, "Client", _no_network)


def _doodles() -> dict[str, Doodle]:
    return {
        key_id: Doodle(key_id, key_id.partition("-")[0], "US", (((0, 0), (99, 99)),))
        for key_id in ORDER
    }


def test_rounds_follow_receipt_order_and_rounds_limit(tmp_path: Path) -> None:
    rows = _receipt()["rows"][:4]
    seen: list[tuple[int, str]] = []

    def answer(round_index: int, key_id: str) -> DuelAnswer:
        seen.append((round_index, key_id))
        return DuelAnswer(key_id, "cat", 60)

    records = play_duel(
        rows,
        _doodles(),
        CATEGORIES,
        answer,
        session_dir=tmp_path / "rounds",
        write=lambda _line: None,
    )

    assert seen == list(enumerate(ORDER[:4], 1))
    assert [r["key_id"] for r in records] == ORDER[:4]
    assert [r["round"] for r in records] == [1, 2, 3, 4]
    names = sorted(p.name for p in (tmp_path / "rounds").iterdir())
    assert names == [f"round-00{i}.png" for i in range(1, 5)]


def test_round_writes_png_and_prints_reveal(tmp_path: Path) -> None:
    lines: list[str] = []
    session = tmp_path / "rounds"

    records = play_duel(
        _receipt()["rows"][:1],
        _doodles(),
        CATEGORIES,
        lambda _i, key_id: DuelAnswer(key_id, "cat", 70),
        session_dir=session,
        write=lines.append,
    )

    png = session / "round-001.png"
    assert png.read_bytes().startswith(PNG_SIGNATURE)
    assert lines[0] == f"Round 1 of 1: open {png}"
    assert lines[-1] == (
        "Answer: moon. You: cat at 70% (wrong). Model: sun at 70% (wrong)."
    )
    assert records[0]["player_correct"] is False
    assert records[0]["model_label"] == "sun"


def test_interactive_reprompts_on_invalid_pick_and_confidence() -> None:
    replies: Iterator[str] = iter(["dog", "9", "Moon", "49", "101", "77%"])
    prompts: list[str] = []
    lines: list[str] = []

    def read_line(prompt: str) -> str:
        prompts.append(prompt)
        return next(replies)

    answer = interactive_answers(read_line, lines.append, CATEGORIES)
    result = answer(1, "moon-1")

    assert result == DuelAnswer("moon-1", "moon", 77)
    assert prompts.count("Your pick (number or name): ") == 3
    assert prompts.count("How sure are you, 50 to 100? ") == 3
    assert "  1. cat" in lines
    assert "  3. sun" in lines
    assert sum("pick" in line for line in lines) >= 2
    assert sum("50 to 100" in line for line in lines) >= 2


def test_interactive_end_of_input_is_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cache = tmp_path / "cache"
    _fill_cache(cache)
    monkeypatch.setattr(quickdraw.httpx, "Client", _no_network)
    monkeypatch.setattr("sys.stdin", io.StringIO("cat\n"))
    duel = tmp_path / "duel.json"

    code = doodle_duel_play.main(
        [
            "--source-receipt",
            str(_write_receipt(tmp_path / "source.json")),
            "--duel-receipt",
            str(duel),
            "--cache-dir",
            str(cache),
        ]
    )

    assert code == 1
    assert not duel.exists()
    assert "refused:" in capsys.readouterr().err


@pytest.mark.usefixtures("offline")
@pytest.mark.parametrize(
    "answers",
    [
        _answers()[:-1],
        _answers(ORDER[1:] + ORDER[:1]),
        [{**_answers()[0], "label": "dog"}, *_answers()[1:]],
        [{**_answers()[0], "confidence": 101}, *_answers()[1:]],
        [{**_answers()[0], "confidence": -1}, *_answers()[1:]],
        [{**_answers()[0], "confidence": 49}, *_answers()[1:]],
        [{"label": "moon", "confidence": 80}, *_answers()[1:]],
    ],
)
def test_file_mode_refuses_invalid_answers_and_writes_nothing(
    tmp_path: Path, answers: list[dict[str, Any]], capsys: pytest.CaptureFixture[str]
) -> None:
    cache = tmp_path / "cache"
    _fill_cache(cache)
    answers_path = tmp_path / "answers.json"
    answers_path.write_text(json.dumps(answers), encoding="utf-8")
    duel = tmp_path / "duel.json"

    code = doodle_duel_play.main(
        [
            "--source-receipt",
            str(_write_receipt(tmp_path / "source.json")),
            "--duel-receipt",
            str(duel),
            "--answers",
            str(answers_path),
            "--cache-dir",
            str(cache),
        ]
    )

    assert code == 1
    assert not duel.exists()
    assert not (tmp_path / "duel-rounds").exists()
    assert "refused:" in capsys.readouterr().err


@pytest.mark.usefixtures("offline")
def test_cache_hit_sends_no_request(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    _fill_cache(cache)
    receipt = _receipt()

    doodles = doodles_for_rows(receipt, receipt["rows"], cache_dir=cache)

    assert set(doodles) == set(ORDER)
    assert doodles["moon-1"].word == "moon"


@pytest.mark.usefixtures("offline")
def test_missing_key_id_is_refused(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    _fill_cache(cache, skip="sun-1")
    receipt = _receipt()

    with pytest.raises(
        ValueError, match=r"key_id sun-1 \(sun\) not in the Quick, Draw! cache"
    ):
        doodles_for_rows(receipt, receipt["rows"], cache_dir=cache)


def test_http_error_on_cache_miss_is_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    receipt = _receipt()
    with _failing_client() as client, pytest.raises(ValueError, match="moon"):
        doodles_for_rows(
            receipt, receipt["rows"], cache_dir=tmp_path / "empty", client=client
        )

    monkeypatch.setattr(quickdraw.httpx, "Client", _failing_client)
    answers_path = tmp_path / "answers.json"
    answers_path.write_text(json.dumps(_answers()), encoding="utf-8")
    duel = tmp_path / "duel.json"
    code = doodle_duel_play.main(
        [
            "--source-receipt",
            str(_write_receipt(tmp_path / "source.json")),
            "--duel-receipt",
            str(duel),
            "--answers",
            str(answers_path),
            "--cache-dir",
            str(tmp_path / "empty"),
        ]
    )

    assert code == 1
    assert not duel.exists()
    assert "refused:" in capsys.readouterr().err


@pytest.mark.usefixtures("offline")
def test_cli_refuses_existing_duel_receipt(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    duel = tmp_path / "duel.json"
    duel.write_text("keep", encoding="utf-8")

    code = doodle_duel_play.main(
        [
            "--source-receipt",
            str(_write_receipt(tmp_path / "source.json")),
            "--duel-receipt",
            str(duel),
            "--cache-dir",
            str(tmp_path / "empty"),
        ]
    )

    assert code == 1
    assert duel.read_text(encoding="utf-8") == "keep"
    assert not (tmp_path / "duel-rounds").exists()
    assert "exists" in capsys.readouterr().err


@pytest.mark.usefixtures("offline")
def test_cli_refuses_nonempty_session_dir(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    session = tmp_path / "duel-rounds"
    session.mkdir()
    (session / "round-001.png").write_bytes(b"old")
    duel = tmp_path / "duel.json"

    code = doodle_duel_play.main(
        [
            "--source-receipt",
            str(_write_receipt(tmp_path / "source.json")),
            "--duel-receipt",
            str(duel),
            "--cache-dir",
            str(tmp_path / "empty"),
        ]
    )

    assert code == 1
    assert not duel.exists()
    assert (session / "round-001.png").read_bytes() == b"old"
    assert "not empty" in capsys.readouterr().err


def _wrong_shapes() -> list[tuple[dict[str, Any], str]]:
    base = _receipt()
    no_field = _receipt()
    del no_field["rows"][2]["probabilities"]
    off_keys = _receipt()
    off_keys["rows"][0]["probabilities"] = {"cat": 1.0}
    off_chosen = _receipt()
    off_chosen["rows"][0]["chosen_label"] = "dog"
    return [
        ({**base, "issue": 1}, "412"),
        ({**base, "rows": []}, "rows"),
        (no_field, "probabilities"),
        (off_keys, "categories"),
        (off_chosen, "dog"),
        ({**base, "pins": {}}, "pins"),
    ]


@pytest.mark.parametrize(("receipt", "message"), _wrong_shapes())
def test_source_receipt_with_wrong_shape_is_refused(
    tmp_path: Path, receipt: dict[str, Any], message: str
) -> None:
    path = _write_receipt(tmp_path / "source.json", receipt)

    with pytest.raises(ValueError, match=message):
        load_source_receipt(path)


def test_file_answers_return_answers_by_round() -> None:
    answers = (DuelAnswer("a", "cat", 10), DuelAnswer("b", "sun", 90))

    answer = file_answers(answers)

    assert answer(2, "b") == answers[1]
    assert answer(1, "a") == answers[0]


DEFINITIONS = (
    (
        "Brier: squared distance between your confidence and what happened; "
        "0 is perfect, 1 is worst"
    ),
    "baseline: what always saying 50 percent would score",
    "gap: mean confidence minus accuracy; positive means overconfident",
)
INTRO = (
    "Each round: name the drawing, then say how sure you are, 50 to 100. "
    "50 means a coin flip, 100 means certain. "
    "Your number is the percent chance your pick is right."
)


def _main(tmp_path: Path, *extra: str) -> int:
    cache = tmp_path / "cache"
    _fill_cache(cache)
    return doodle_duel_play.main(
        [
            "--source-receipt",
            str(_write_receipt(tmp_path / "source.json")),
            "--duel-receipt",
            str(tmp_path / "duel.json"),
            "--cache-dir",
            str(cache),
            *extra,
        ]
    )


@pytest.mark.usefixtures("offline")
def test_file_mode_summary_defines_brier_baseline_and_gap(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    answers_path = tmp_path / "answers.json"
    answers_path.write_text(json.dumps(_answers()), encoding="utf-8")

    code = _main(tmp_path, "--answers", str(answers_path))

    assert code == 0
    lines = capsys.readouterr().out.splitlines()
    for definition in DEFINITIONS:
        assert lines.count(definition) == 1
    assert INTRO not in lines


@pytest.mark.usefixtures("offline")
def test_interactive_mode_prints_intro_before_round_one_and_definitions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    typed = "".join(f"{key_id.partition('-')[0]}\n80\n" for key_id in ORDER)
    monkeypatch.setattr("sys.stdin", io.StringIO(typed))

    code = _main(tmp_path)

    assert code == 0
    lines = capsys.readouterr().out.splitlines()
    first_round = next(i for i, line in enumerate(lines) if line.startswith("Round 1"))
    assert lines.index(INTRO) < first_round
    assert lines.count(INTRO) == 1
    for definition in DEFINITIONS:
        assert lines.count(definition) == 1


@pytest.mark.parametrize("gap", [-0.0004, -0.0, 0.0, 0.0004])
def test_gap_that_rounds_to_zero_prints_without_minus_sign(gap: float) -> None:
    row = {
        "bin": "60-70",
        "count": 3,
        "mean_confidence": 0.6663,
        "accuracy": 0.6667,
        "gap": gap,
    }
    receipt = {
        "scores": {
            "player": {"accuracy": 0.5, "mean_brier": 0.2},
            "model": {"accuracy": 0.5, "mean_brier": 0.2, "multiclass_brier": 0.4},
            "baseline_brier": 0.25,
        },
        "reliability": {"player": [row], "model": [row]},
    }
    lines: list[str] = []

    doodle_duel_play._print_summary(receipt, lines.append)

    table = [line for line in lines if line.startswith("60-70")]
    assert len(table) == 2
    assert all(line.endswith(" 0.000") for line in table)
    assert not any("-0.000" in line for line in lines)


def test_gap_below_zero_keeps_its_sign() -> None:
    row = {
        "bin": "60-70",
        "count": 3,
        "mean_confidence": 0.6,
        "accuracy": 0.6667,
        "gap": -0.0667,
    }
    receipt = {
        "scores": {
            "player": {"accuracy": 0.5, "mean_brier": 0.2},
            "model": {"accuracy": 0.5, "mean_brier": 0.2, "multiclass_brier": 0.4},
            "baseline_brier": 0.25,
        },
        "reliability": {"player": [row], "model": [row]},
    }
    lines: list[str] = []

    doodle_duel_play._print_summary(receipt, lines.append)

    assert sum(line.endswith("-0.067") for line in lines) == 2

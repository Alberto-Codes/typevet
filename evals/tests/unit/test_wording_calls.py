"""Unit checks for per-call latency and input-token records (#327).

A fake clock and a fake port make the latency and the token counts exact.

Examples:
    ```bash
    uv run pytest -q evals/tests/unit/test_wording_calls.py
    ```

See Also:
    - [typevet_evals.wording.calls][]: the timed port
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections.abc import Iterator, Mapping
from typing import Any

import pytest
from judgevet import NoulAnswer, SystemOneResponse, Usage
from judgevet.domain.questions import Noul

from typevet_evals.wording.calls import CallRecord, TimedJudgePort, call_summary

pytestmark = pytest.mark.unit

KEY = "is_scam"
QUESTIONS = {KEY: Noul(instructions="Is this message a scam?")}


class TokenPort:
    """Answer 0.7 with ``len(state)`` input tokens; raise on the state ``down``."""

    def system_one(
        self, state: str, questions: Mapping[str, Any], model: str
    ) -> SystemOneResponse:
        """Answer or raise.

        Returns:
            One Noul answer for each question.

        Raises:
            RuntimeError: When ``state`` is ``down``.
        """
        if state == "down":
            raise RuntimeError("backend down")
        return SystemOneResponse(
            model=model,
            usage=Usage(input_tokens=len(state), output_tokens=1),
            answers={name: NoulAnswer(noul=0.7) for name in questions},
        )


def _clock(ticks: list[float]) -> Iterator[float]:
    """Return an iterator over ``ticks``."""
    return iter(ticks)


def test_each_call_records_latency_input_tokens_and_the_state_hash() -> None:
    ticks = _clock([10.0, 10.25, 20.0, 20.5])
    port = TimedJudgePort(TokenPort(), clock=lambda: next(ticks))

    first = port.system_one("Win cash now", QUESTIONS, "gemma")
    port.system_one("hi", QUESTIONS, "gemma")

    assert first.nouls[KEY].noul == pytest.approx(0.7)
    digest = hashlib.sha256(b"Win cash now").hexdigest()
    assert port.records == (
        CallRecord(0, digest, (KEY,), 0.25, 12, None),
        CallRecord(1, hashlib.sha256(b"hi").hexdigest(), (KEY,), 0.5, 2, None),
    )


def test_a_failed_call_is_recorded_and_raised() -> None:
    ticks = _clock([1.0, 3.0])
    port = TimedJudgePort(TokenPort(), clock=lambda: next(ticks))

    with pytest.raises(RuntimeError, match="backend down"):
        port.system_one("down", QUESTIONS, "gemma")

    (record,) = port.records
    assert record.latency_seconds == pytest.approx(2.0)
    assert record.input_tokens is None
    assert record.error == "RuntimeError: backend down"


THREADS = 8
CALLS_PER_THREAD = 10


def _slow_clock() -> float:
    """Sleep, so another thread runs inside any unlocked index step."""
    time.sleep(0.001)
    return 0.0


def test_concurrent_calls_get_unique_contiguous_indices_and_every_record() -> None:
    """Threads start together; no index repeats and no record is lost."""
    port = TimedJudgePort(TokenPort(), clock=_slow_clock)
    barrier = threading.Barrier(THREADS)

    def work() -> None:
        barrier.wait()
        for _ in range(CALLS_PER_THREAD):
            port.system_one("hi", QUESTIONS, "gemma")

    threads = [threading.Thread(target=work) for _ in range(THREADS)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    total = THREADS * CALLS_PER_THREAD
    assert len(port.records) == total
    assert sorted(r.index for r in port.records) == list(range(total))


def test_the_summary_counts_calls_tokens_and_latency() -> None:
    records = (
        CallRecord(0, "a", (KEY,), 0.2, 10, None),
        CallRecord(1, "b", (KEY,), 0.4, None, None),
        CallRecord(2, "c", (KEY,), 0.9, 30, "RuntimeError: x"),
    )

    summary = call_summary(records)

    assert summary == {
        "calls": 3,
        "failed": 1,
        "input_tokens_total": 40,
        "input_tokens_known": 2,
        "latency_seconds_total": pytest.approx(1.5),
        "latency_seconds_median": pytest.approx(0.4),
        "latency_seconds_max": pytest.approx(0.9),
    }


def test_an_empty_summary_has_no_latency() -> None:
    assert call_summary(())["latency_seconds_median"] is None


def test_a_record_maps_to_json_fields() -> None:
    record = CallRecord(3, "abc", (KEY,), 0.123456789, 5, None)

    assert record.to_mapping() == {
        "index": 3,
        "state_sha256": "abc",
        "questions": [KEY],
        "latency_seconds": 0.123457,
        "input_tokens": 5,
        "error": None,
    }

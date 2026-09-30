"""Record the latency and input tokens of each judge call (#327).

``TimedJudgePort`` wraps a judgevet ``SystemOnePort``. Each ``system_one``
call adds one ``CallRecord``: the call index, the SHA-256 of the state, the
question names, the latency, the input tokens the backend reports and the
error of a failed call. ``call_summary`` totals the records for a receipt.

Calls may run in several threads, as gepa-adk evaluates concurrently. One
lock covers the index step, the append of each record and the copy of the
records, so no index repeats and no record is lost, also on a free-threaded
interpreter.

The llama.cpp input count is ``tokens_evaluated``. The llama.cpp scoring
adapter sends ``cache_prompt: false``, so the count is the full prompt. vLLM
reports ``prompt_tokens``.

Examples:
    ```python
    timed = TimedJudgePort(TypevetSystemOnePort(port))
    timed.system_one("Win cash now", {"is_scam": noul}, "gemma")
    summary = call_summary(timed.records)
    ```

See Also:
    - [typevet_evals.wording.held_out][]: the receipt that holds the records
"""

from __future__ import annotations

import hashlib
import statistics
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from typevet_evals.wording.transport import JudgePort

if TYPE_CHECKING:
    from judgevet import SystemOneResponse


@dataclass(frozen=True, slots=True)
class CallRecord:
    """One judge call.

    Attributes:
        index (int): The call number, from 0; unique per port.
        state_sha256 (str): SHA-256 of the state text.
        questions (tuple[str, ...]): The question names asked.
        latency_seconds (float): Wall time of the call.
        input_tokens (int | None): Input tokens the backend reports, or None.
        error (str | None): ``Type: message`` of a failed call, or None.

    Examples:
        ```python
        CallRecord(0, "ab12", ("is_scam",), 0.4, 180, None).to_mapping()
        ```
    """

    index: int
    state_sha256: str
    questions: tuple[str, ...]
    latency_seconds: float
    input_tokens: int | None
    error: str | None

    def to_mapping(self) -> dict[str, Any]:
        """Return the JSON fields, latency rounded to microseconds.

        Returns:
            A JSON-serializable mapping.
        """
        return {
            "index": self.index,
            "state_sha256": self.state_sha256,
            "questions": list(self.questions),
            "latency_seconds": round(self.latency_seconds, 6),
            "input_tokens": self.input_tokens,
            "error": self.error,
        }


class TimedJudgePort:
    """A ``SystemOnePort`` wrapper that records each call.

    Attributes:
        records (tuple[CallRecord, ...]): The calls, in finish order.

    Examples:
        ```python
        timed = TimedJudgePort(port)
        ```
    """

    def __init__(
        self, port: JudgePort, *, clock: Callable[[], float] = time.perf_counter
    ) -> None:
        """Wrap ``port`` and read time from ``clock``.

        Args:
            port: The judgevet ``SystemOnePort``.
            clock: A monotonic clock in seconds.
        """
        self._port = port
        self._clock = clock
        self._lock = threading.Lock()
        self._next = 0
        self._records: list[CallRecord] = []

    @property
    def records(self) -> tuple[CallRecord, ...]:
        """Return the calls recorded so far.

        Returns:
            The records, in finish order.
        """
        with self._lock:
            return tuple(self._records)

    def system_one(
        self, state: str, questions: Mapping[str, Any], model: str
    ) -> SystemOneResponse:
        """Call the wrapped port and record the call.

        Args:
            state: The state text.
            questions: The named questions.
            model: The model name.

        Returns:
            The wrapped port's response.

        Raises:
            Exception: Whatever the wrapped port raises, after the record.
        """
        digest = hashlib.sha256(state.encode()).hexdigest()
        names = tuple(questions)
        with self._lock:
            index = self._next
            start = self._clock()
            self._next += 1
        try:
            response = self._port.system_one(state, questions, model)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            latency = self._clock() - start
            self._add(CallRecord(index, digest, names, latency, None, error))
            raise
        latency = self._clock() - start
        usage = response.usage
        tokens = None if usage is None else usage.input_tokens
        self._add(CallRecord(index, digest, names, latency, tokens, None))
        return response

    def _add(self, record: CallRecord) -> None:
        with self._lock:
            self._records.append(record)


def call_summary(records: Sequence[CallRecord]) -> dict[str, Any]:
    """Total the calls, failures, input tokens and latency.

    Args:
        records: The call records.

    Returns:
        Counts, the known input-token total and the latency total, median
        and maximum; the median and maximum are None without records.
    """
    latencies = [r.latency_seconds for r in records]
    tokens = [r.input_tokens for r in records if r.input_tokens is not None]
    return {
        "calls": len(records),
        "failed": sum(r.error is not None for r in records),
        "input_tokens_total": sum(tokens),
        "input_tokens_known": len(tokens),
        "latency_seconds_total": sum(latencies),
        "latency_seconds_median": statistics.median(latencies) if latencies else None,
        "latency_seconds_max": max(latencies) if latencies else None,
    }

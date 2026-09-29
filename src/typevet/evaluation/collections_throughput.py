"""Concurrency sweep and receipt for the finvet collections workload ([#236][i236]).

``run_throughput`` opens the sync judgment port with ``open_judgment`` and the
``TYPEVET_VLLM__*`` settings, so the port masks the key and the client sends
``TYPEVET_VLLM__USER_AGENT``. Each level runs every record once through a
``ThreadPoolExecutor`` of that width. Every request passes
``CountingTransport``, which enforces the call caps; the harness makes no
retry. A level stops when its errors exceed 1% of its records, when a call
cap is reached or when its time cap passes. Records not yet sent are then
skipped, and no higher level runs. The ``/metrics`` text is read before and
after each level, and ``kv_cache_usage`` is read while calls are in flight.
A1 ``parity`` scores each level against ``baseline``; a skipped or failed
record counts as a failure. Any record with ``state`` and ``positive`` fits,
and ``noul`` names the question whose probability is scored.
``cold_start_seconds``, ``hourly_usd``, ``gpu_memory`` and ``pod_cost_usd``
are ``unknown`` unless the caller supplies them.

Attributes:
    LEVELS (tuple[int, ...]): Pre-registered in-flight record counts.
    CAPS (CallCaps): Call caps for one run.
    MAX_ERROR_RATE (float): Highest error rate that does not stop a level.
    KV_WAIT_SECONDS (float): Longest wait for calls in flight before the read.
    RUN_SECONDS (float): Time budget shared by every run of one measurement.
    METHOD (dict[str, str]): Measurement rules copied into each receipt.
    SUPPLIED (tuple[str, ...]): Receipt fields only the caller can supply.
    ScoredRecord (Protocol): A record with a ``state`` and a ``positive`` label.
    Scoring (TypedDict): Optional ``noul`` and ``baseline`` run keywords.

Examples:
    ```python
    import httpx

    from typevet.evaluation.collections_throughput import run_throughput

    with httpx.HTTPTransport() as transport:
        receipt = run_throughput(env, records, questions, transport=transport)
    ```

See Also:
    - [typevet.evaluation.collections_workload][]: records, questions, parity
    - [typevet.evaluation.public_workload][]: Banking77 and DIFrauD workloads
    - [typevet.evaluation.collections_metrics][]: /metrics deltas
    - [typevet.evaluation.vllm_acceptance][]: caps, counter and receipt writer

[i236]: https://github.com/Alberto-Codes/typevet/issues/236
"""

from __future__ import annotations

import threading
from collections.abc import Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from time import perf_counter
from typing import Any, Final, Protocol, TypedDict, Unpack

import httpx

from typevet.adapters.inbound.backend_settings import (
    load_backend,
    load_vllm_settings,
    open_judgment,
)
from typevet.domain.errors import GenerationError
from typevet.domain.judgment_questions import Choice, Noul
from typevet.evaluation.collections_metrics import (
    UNKNOWN,
    latency,
    read_metrics,
    server_delta,
)
from typevet.evaluation.collections_workload import (
    COLLECTIONS_BASELINE,
    Baseline,
    parity,
)
from typevet.evaluation.vllm_acceptance import (
    AcceptanceStoppedError,
    CallCaps,
    CountingTransport,
    kv_cache_usage,
)

LEVELS: Final[tuple[int, ...]] = (1, 8, 32, 64)
CAPS: Final[CallCaps] = CallCaps(model=11000, tokenizer=256, metadata=40)
MAX_ERROR_RATE: Final[float] = 0.01
KV_WAIT_SECONDS: Final[float] = 10.0
RUN_SECONDS: Final[float] = 2700.0
METHOD: Final[dict[str, str]] = {
    "time_cap": (
        "Checked before each send; a record not yet sent is skipped and a "
        "request in flight finishes."
    ),
    "run_budget": (
        "One budget for the sweep and the later run; the later run gets "
        "RUN_SECONDS minus the sweep elapsed_seconds."
    ),
    "wall_clock": "Last record end minus first record start, client clock.",
    "records_per_second": "Answered records divided by wall_seconds.",
    "client_latency": "Seconds per sent record, answered and failed records.",
    "server_percentiles": (
        "Upper edge of the first /metrics bucket that holds the nearest rank; "
        "+Inf means the rank is above the last bucket edge."
    ),
}
SUPPLIED: Final[tuple[str, ...]] = (
    "cold_start_seconds",
    "hourly_usd",
    "gpu_memory",
    "pod_cost_usd",
)
_NOUL: Final[str] = "will_engage"
_POLL_SECONDS: Final[float] = 0.05
_ERROR_STOP: Final[str] = f"error rate above {MAX_ERROR_RATE}"


class ScoredRecord(Protocol):
    """A record the runner can send and score.

    Examples:
        ```python
        from typevet.evaluation.public_workload import PublicRecord

        record: ScoredRecord = PublicRecord(state="text", positive=True)
        ```
    """

    @property
    def state(self) -> str | dict[str, Any]:
        """Return the state sent to the judgment port."""
        ...

    @property
    def positive(self) -> bool:
        """Return True when the logged label is positive."""
        ...


class Scoring(TypedDict, total=False):
    """Optional ``run_throughput`` keywords that choose what parity scores.

    Attributes:
        noul (str): Question whose probability is scored; default
            ``will_engage``.
        baseline (Baseline | None): Parity reference; default the collections
            baseline; ``None`` records the measures only.
    """

    noul: str
    baseline: Baseline | None


@dataclass(frozen=True, slots=True)
class RunOptions:
    """Time caps and caller-supplied receipt values.

    Attributes:
        level_seconds (float): Time cap for one level.
        run_seconds (float): Time cap for the whole run.
        supplied (Mapping[str, Any]): Values for ``SUPPLIED`` keys.

    Examples:
        ```python
        RunOptions(supplied={"hourly_usd": 3.49})
        ```
    """

    level_seconds: float = 900.0
    run_seconds: float = RUN_SECONDS
    supplied: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class _Run:
    port: Any
    client: httpx.Client
    counter: CountingTransport
    model: str
    records: Sequence[ScoredRecord]
    questions: Mapping[str, Noul | Choice]
    noul: str
    baseline: Baseline | None


class _Level:
    """Shared, lock-guarded state of one level.

    Attributes:
        deadline (float): ``perf_counter`` value after which no record is sent.
        rows (list[dict[str, Any] | None]): One row per record; ``None`` when
            the record was not sent.
        stop (threading.Event): Set when the level stops.
        reason (str | None): First stop reason.
        errors (int): Records that raised a ``GenerationError``.

    Examples:
        ```python
        level = _Level(size=592, deadline=perf_counter() + 900.0)
        ```
    """

    def __init__(self, size: int, deadline: float) -> None:
        self.deadline = deadline
        self.rows: list[dict[str, Any] | None] = [None] * size
        self.stop = threading.Event()
        self.reason: str | None = None
        self.errors = 0
        self._size = size
        self._lock = threading.Lock()

    def halt(self, reason: str) -> None:
        """Stop the level; the first reason is kept.

        Args:
            reason: Why the level stops.
        """
        with self._lock:
            self.reason = self.reason or reason
            self.stop.set()

    def record(self, index: int, row: dict[str, Any]) -> None:
        """Store one row and stop the level when errors pass the rate.

        Args:
            index: Record position.
            row: Row with ``p`` and ``tokens``, or ``error``.
        """
        with self._lock:
            self.rows[index] = row
            if "error" in row:
                self.errors += 1
        if self.errors > MAX_ERROR_RATE * self._size:
            self.halt(_ERROR_STOP)


def _one(run: _Run, level: _Level, index: int) -> None:
    if level.stop.is_set():
        return
    if perf_counter() >= level.deadline:
        level.halt("time cap")
        return
    record = run.records[index]
    started = perf_counter()
    try:
        response = run.port.judge(record.state, run.questions, run.model)
    except AcceptanceStoppedError as exc:
        level.halt(str(exc))
        return
    except GenerationError as exc:
        row: dict[str, Any] = {
            "error": {"type": type(exc).__name__, "message": str(exc)}
        }
    else:
        row = {
            "p": response.nouls[run.noul].noul,
            "tokens": response.usage.input_tokens,
        }
    level.record(index, {**row, "started": started, "ended": perf_counter()})


def _kv_in_flight(run: _Run, target: int, futures: Sequence[Future[None]]) -> Any:
    end = perf_counter() + KV_WAIT_SECONDS
    while not run.counter.wait_for("model", target, _POLL_SECONDS):
        if perf_counter() >= end or all(f.done() for f in futures):
            break
    return kv_cache_usage(run.client)


def _rate(amount: float, seconds: float) -> float | None:
    return amount / seconds if seconds > 0 else None


def _tokens(rows: Sequence[dict[str, Any]]) -> Any:
    known = sorted(r["tokens"] for r in rows if isinstance(r.get("tokens"), int))
    if not known:
        return UNKNOWN
    return {"n": len(known), "mean": sum(known) / len(known), "max": known[-1]}


def level_timing(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return the wall clock and answered-record rate of sent rows.

    Args:
        rows: Sent rows with ``started`` and ``ended``; a failed row has
            ``error``.

    Returns:
        ``wall_seconds`` (last end minus first start, 0.0 when empty) and
        ``records_per_second`` (answered rows over wall, ``None`` at 0 s).
    """
    ended = max((r["ended"] for r in rows), default=0.0)
    wall = ended - min((r["started"] for r in rows), default=0.0)
    answered = sum("error" not in r for r in rows)
    return {"wall_seconds": wall, "records_per_second": _rate(answered, wall)}


def remaining_run_seconds(
    earlier: Mapping[str, Any], total: float = RUN_SECONDS
) -> float:
    """Return the run budget that an earlier receipt left.

    Args:
        earlier: Receipt from ``run_throughput`` with ``elapsed_seconds``.
        total: Budget shared by both runs.

    Returns:
        ``total`` minus the earlier elapsed seconds, never below 0.0.
    """
    return max(total - earlier["elapsed_seconds"], 0.0)


def _run_level(run: _Run, width: int, deadline: float) -> dict[str, Any]:
    size = len(run.records)
    level = _Level(size, deadline)
    before, calls = read_metrics(run.client), run.counter.calls["model"]
    with ThreadPoolExecutor(max_workers=width) as pool:
        futures = [pool.submit(_one, run, level, i) for i in range(size)]
        kv = _kv_in_flight(run, calls + min(width, size), futures)
        for future in futures:
            future.result()
    after = read_metrics(run.client)
    done = [row for row in level.rows if row is not None]
    answered = [row for row in done if "error" not in row]
    timing = level_timing(done)
    wall = timing["wall_seconds"]
    scoring = run.counter.calls["model"] - calls
    probs = [None if r is None or "error" in r else r["p"] for r in level.rows]
    return {
        "level": width,
        "records": size,
        "sent": len(done),
        "answered": len(answered),
        "errors": level.errors,
        "error_rate": level.errors / size if size else 0.0,
        "error_rows": [r["error"] for r in done if "error" in r],
        "stopped": level.reason,
        "wall_seconds": round(wall, 6),
        "records_per_second": timing["records_per_second"],
        "scoring_calls": scoring,
        "scoring_calls_per_second": _rate(scoring, wall),
        "latency": latency([r["ended"] - r["started"] for r in done]),
        "prompt_tokens": _tokens(answered),
        "server": server_delta(before, after),
        "kv_cache_usage": kv,
        "parity": parity(
            [(p, rec.positive) for p, rec in zip(probs, run.records, strict=True)],
            run.baseline,
        ),
    }


def best_level(levels: Sequence[Mapping[str, Any]]) -> int | str:
    """Return the level with the highest records/s and at most 1% errors.

    Args:
        levels: Level results from ``run_throughput``.

    Returns:
        The level width, or ``unknown`` when no level finished unstopped with
        a measured rate.
    """
    usable = [
        lv
        for lv in levels
        if lv["stopped"] is None
        and lv["error_rate"] <= MAX_ERROR_RATE
        and lv["records_per_second"] is not None
    ]
    if not usable:
        return UNKNOWN
    return max(usable, key=lambda lv: lv["records_per_second"])["level"]


def _json(client: httpx.Client, path: str) -> Any:
    try:
        return client.get(path).json()
    except (httpx.HTTPError, ValueError):
        return UNKNOWN


def _pins(client: httpx.Client, environ: Mapping[str, str]) -> dict[str, Any]:
    settings = load_vllm_settings(environ)
    models = _json(client, "/v1/models")
    data = models.get("data") if isinstance(models, dict) else None
    return {
        "version": _json(client, "/version"),
        "served_models": [m.get("id") for m in data or [] if isinstance(m, dict)],
        "configured_model": settings.model,
        "base_url": settings.base_url,
        "user_agent": settings.user_agent,
    }


def _costs(receipt: dict[str, Any], supplied: Mapping[str, Any]) -> None:
    receipt.update({key: supplied.get(key, UNKNOWN) for key in SUPPLIED})
    best = best_level(receipt["levels"])
    receipt["best_level"] = best
    hourly = receipt["hourly_usd"]
    rates = [
        lv["records_per_second"] for lv in receipt["levels"] if lv["level"] == best
    ]
    if isinstance(hourly, int | float) and rates:
        receipt["cost_per_1000"] = hourly / 3600 * 1000 / rates[0]
    else:
        receipt["cost_per_1000"] = UNKNOWN


def run_throughput(
    environ: Mapping[str, str],
    records: Sequence[ScoredRecord],
    questions: Mapping[str, Noul | Choice],
    *,
    transport: httpx.BaseTransport,
    levels: Sequence[int] = LEVELS,
    caps: CallCaps = CAPS,
    options: RunOptions | None = None,
    **scoring: Unpack[Scoring],
) -> dict[str, Any]:
    """Run the concurrency sweep once and return the receipt mapping.

    Args:
        environ: Mapping with ``TYPEVET_BACKEND=vllm`` and ``TYPEVET_VLLM__*``.
        records: Records with ``state`` and ``positive``, for example from
            ``load_records`` or a public workload.
        questions: Questions from ``load_questions`` or a public workload.
        transport: Transport that sends requests; the caller closes it.
        levels: In-flight record counts, run in order.
        caps: Call caps for the whole run.
        options: Time caps and supplied values. Defaults to ``RunOptions()``.

    Other Parameters:
        noul (str): Question whose probability each level scores; default
            ``will_engage``.
        baseline (Baseline | None): Parity reference values; default the
            collections baseline; ``None`` records the measures only.

    Returns:
        Receipt with ``pins``, ``caps``, ``time_caps``, ``method``,
        ``elapsed_seconds``, ``levels``,
        ``best_level``, ``calls``, ``stopped``, the ``SUPPLIED`` fields and
        ``cost_per_1000``. A stop leaves ``stopped`` set and still returns.

    Raises:
        TypeError: When a keyword is not a ``Scoring`` key.
        ValueError: When the backend is not ``vllm`` or a setting is invalid.
    """
    unknown = sorted(set(scoring) - set(Scoring.__annotations__))
    if unknown:
        raise TypeError(f"run_throughput got unknown keywords: {unknown}")
    if load_backend(environ) != "vllm":
        raise ValueError("TYPEVET_BACKEND must be vllm for the collections run")
    opts = options or RunOptions()
    counter = CountingTransport(transport, caps)
    receipt: dict[str, Any] = {"issue": 236, "artifact": "collections_throughput_v1"}
    receipt.update(records=len(records), levels_planned=list(levels), pins={})
    receipt.update(caps=asdict(caps), max_error_rate=MAX_ERROR_RATE)
    receipt.update(time_caps={"level_seconds": opts.level_seconds})
    receipt["time_caps"]["run_seconds"] = opts.run_seconds
    receipt.update(transport_retries=0, stopped=None, levels=[], method=dict(METHOD))
    started = perf_counter()
    run_deadline = started + opts.run_seconds
    try:
        with open_judgment(environ, transport=counter) as session:
            receipt["pins"] = _pins(session.client, environ)
            run = _Run(
                session.port,
                session.client,
                counter,
                receipt["pins"]["configured_model"],
                records,
                questions,
                scoring.get("noul", _NOUL),
                scoring.get("baseline", COLLECTIONS_BASELINE),
            )
            for width in levels:
                deadline = min(perf_counter() + opts.level_seconds, run_deadline)
                out = _run_level(run, width, deadline)
                receipt["levels"].append(out)
                if out["stopped"] is not None:
                    receipt["stopped"] = f"level {width}: {out['stopped']}"
                    break
    except AcceptanceStoppedError as exc:
        receipt["stopped"] = str(exc)
    receipt.update(
        calls=dict(counter.calls), tokenizer_memo_hits=counter.tokenizer_memo_hits
    )
    receipt["elapsed_seconds"] = perf_counter() - started
    _costs(receipt, opts.supplied)
    return receipt

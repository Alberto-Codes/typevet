"""Client latency, run rates and vLLM ``/metrics`` deltas ([#236][i236], [#335][i335]).

The collections throughput run (#236) and the face, check and signature image
runs (#335) share these pure helpers. The module sits below both in the
import layers: it imports no evaluation family.

The series names come from vLLM ``v0.30.0`` ``vllm/v1/metrics/loggers.py``:
``vllm:e2e_request_latency_seconds`` (line 842),
``vllm:request_queue_time_seconds`` (line 852),
``vllm:request_prefill_time_seconds`` (line 872), ``vllm:prefix_cache_queries``
(line 590), ``vllm:prefix_cache_hits`` (line 601), ``vllm:mm_cache_queries``
(line 642), ``vllm:mm_cache_hits`` (line 653), ``vllm:num_preemptions``
(line 667), ``vllm:prompt_tokens`` (line 676), ``vllm:prompt_tokens_cached``
(line 701), ``vllm:generation_tokens`` (line 710), ``vllm:request_success``
(line 720), and the gauges ``vllm:num_requests_running`` (line 499),
``vllm:num_requests_waiting`` (line 509) and ``vllm:kv_cache_usage_perc``
(line 567). The counters are ``prometheus_client`` counters, so the text
exposition adds ``_total``; the parser also accepts the bare name. Samples
with the same name and ``le`` label are summed across the other labels
(``model_name``, ``engine``, ``finished_reason``). A gauge summed across
several engines is not a usage fraction, so ``run_server_delta`` refuses a
reading in which one ``GAUGES`` series carries more than one ``engine`` label
(#339); the pinned runs serve one engine. A series that is absent from
either reading is recorded as ``unknown``. Server percentiles are the upper
edge of the first bucket that holds the nearest rank, so they are bounds, not
exact values; ``+Inf`` means the rank is above the last edge. A gauge is a
point reading before and after the run, not a peak.

Attributes:
    UNKNOWN (str): Value recorded for an absent series.
    HISTOGRAMS (dict[str, str]): Receipt key to vLLM histogram name.
    PREFIX_HITS (str): vLLM prefix-cache hit counter name.
    PREFIX_QUERIES (str): vLLM prefix-cache query counter name.
    COUNTERS (dict[str, str]): Receipt key to vLLM counter name for the
        image runs.
    GAUGES (dict[str, str]): Receipt key to vLLM gauge name for the image
        runs.
    QUANTILES (tuple[tuple[str, float], ...]): Reported percentiles.

Examples:
    ```python
    from typevet_evals.serving_metrics import latency, run_server_delta

    assert latency([0.2, 0.1])["p50"] == 0.1
    assert run_server_delta(None, None)["e2e"] == "unknown"
    ```

See Also:
    - [typevet_evals.throughput.collections_metrics][]: the #236 names
    - [typevet_evals.face_match.runner][]: the face-match receipt

[i236]: https://github.com/Alberto-Codes/typevet/issues/236
[i335]: https://github.com/Alberto-Codes/typevet/issues/335
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any, Final

import httpx

UNKNOWN: Final[str] = "unknown"
HISTOGRAMS: Final[dict[str, str]] = {
    "e2e": "vllm:e2e_request_latency_seconds",
    "queue": "vllm:request_queue_time_seconds",
    "prefill": "vllm:request_prefill_time_seconds",
}
PREFIX_HITS: Final[str] = "vllm:prefix_cache_hits"
PREFIX_QUERIES: Final[str] = "vllm:prefix_cache_queries"
COUNTERS: Final[dict[str, str]] = {
    "prefix_cache_hits": PREFIX_HITS,
    "prefix_cache_queries": PREFIX_QUERIES,
    "mm_cache_hits": "vllm:mm_cache_hits",
    "mm_cache_queries": "vllm:mm_cache_queries",
    "prompt_tokens": "vllm:prompt_tokens",
    "prompt_tokens_cached": "vllm:prompt_tokens_cached",
    "generation_tokens": "vllm:generation_tokens",
    "request_success": "vllm:request_success",
    "preemptions": "vllm:num_preemptions",
}
GAUGES: Final[dict[str, str]] = {
    "kv_cache_usage_perc": "vllm:kv_cache_usage_perc",
    "num_requests_running": "vllm:num_requests_running",
    "num_requests_waiting": "vllm:num_requests_waiting",
}
QUANTILES: Final[tuple[tuple[str, float], ...]] = (
    ("p50", 0.5),
    ("p95", 0.95),
    ("p99", 0.99),
)
_SAMPLE: Final = re.compile(
    r"^(?P<name>[A-Za-z_:][A-Za-z0-9_:]*)(?:\{(?P<labels>[^}]*)\})?\s+(?P<value>\S+)"
)
_LE: Final = re.compile(r'(?:^|,)\s*le="(?P<le>[^"]*)"')
_ENGINE: Final = re.compile(r'(?:^|,)\s*engine="(?P<engine>[^"]*)"')

Snapshot = dict[tuple[str, str | None], float]


def nearest_rank(ordered: Sequence[float], quantile: float) -> float | None:
    """Return the nearest-rank quantile of sorted values.

    Args:
        ordered: Values in ascending order.
        quantile: Quantile in (0, 1].

    Returns:
        The value at rank ``ceil(quantile * n)``, or ``None`` when empty.
    """
    if not ordered:
        return None
    return ordered[max(math.ceil(quantile * len(ordered)) - 1, 0)]


def latency(samples: Sequence[float]) -> dict[str, Any]:
    """Summarize client seconds with nearest-rank p50, p95 and p99.

    Args:
        samples: Seconds per record.

    Returns:
        ``n`` and ``p50``, ``p95``, ``p99``; each is ``None`` when empty.
    """
    ordered = sorted(samples)
    out: dict[str, Any] = {"n": len(ordered)}
    out.update((key, nearest_rank(ordered, q)) for key, q in QUANTILES)
    return out


def read_metrics(client: httpx.Client) -> str | None:
    """Read the ``/metrics`` text once (one metadata call).

    Args:
        client: Session client.

    Returns:
        The response text, or ``None`` when the request fails or is not 200.
    """
    try:
        response = client.get("/metrics")
    except httpx.HTTPError:
        return None
    return response.text if response.status_code == httpx.codes.OK else None


def snapshot(text: str) -> Snapshot:
    """Parse Prometheus text into sums keyed by sample name and ``le``.

    Args:
        text: ``/metrics`` exposition text.

    Returns:
        ``{(name, le or None): value}`` summed over the other labels. Comment
        lines and unreadable values are skipped.
    """
    out: Snapshot = {}
    for line in text.splitlines():
        match = _SAMPLE.match(line)
        if match is None:
            continue
        try:
            value = float(match.group("value"))
        except ValueError:
            continue
        le = _LE.search(match.group("labels") or "")
        key = (match.group("name"), le.group("le") if le else None)
        out[key] = out.get(key, 0.0) + value
    return out


def _bucket_edges(snap: Snapshot, name: str) -> dict[str, float]:
    bucket = f"{name}_bucket"
    return {le: v for (n, le), v in snap.items() if n == bucket and le is not None}


def _edge(le: str) -> float | str:
    return "+Inf" if le == "+Inf" else float(le)


def histogram_delta(before: Snapshot, after: Snapshot, name: str) -> Any:
    """Return the change of one histogram between two readings.

    Args:
        before: Reading before the level.
        after: Reading after the level.
        name: vLLM histogram name without suffix.

    Returns:
        ``count``, ``sum``, bucket-bound ``p50``, ``p95``, ``p99`` (``None``
        when the count did not change) and ``percentile_basis``
        (``bucket_upper_bound``), or ``unknown`` when the series is absent
        from either reading.
    """
    keys = [(f"{name}_count", None), (f"{name}_sum", None)]
    old, new = _bucket_edges(before, name), _bucket_edges(after, name)
    if not all(k in before and k in after for k in keys) or not new:
        return UNKNOWN
    if set(old) != set(new):
        return UNKNOWN
    count = after[keys[0]] - before[keys[0]]
    edges = sorted(new, key=lambda le: math.inf if le == "+Inf" else float(le))
    cumulative = [(le, new[le] - old[le]) for le in edges]
    out: dict[str, Any] = {"count": count, "sum": after[keys[1]] - before[keys[1]]}
    out["percentile_basis"] = "bucket_upper_bound"
    for key, q in QUANTILES:
        rank = math.ceil(q * count)
        hit = next((le for le, c in cumulative if c >= rank), None)
        out[key] = None if count <= 0 or hit is None else _edge(hit)
    return out


def _counter(snap: Snapshot, name: str) -> float | None:
    for key in ((f"{name}_total", None), (name, None)):
        if key in snap:
            return snap[key]
    return None


def counter_delta(before: Snapshot, after: Snapshot, name: str) -> float | str:
    """Return the change of one counter between two readings.

    Args:
        before: Reading before the run.
        after: Reading after the run.
        name: vLLM counter name without the ``_total`` suffix.

    Returns:
        The change, or ``unknown`` when the counter is absent from either
        reading.
    """
    old, new = _counter(before, name), _counter(after, name)
    if old is None or new is None:
        return UNKNOWN
    return new - old


def prefix_hit_rate(before: Snapshot, after: Snapshot) -> float | str:
    """Return Δ prefix-cache hits over Δ prefix-cache queries.

    Args:
        before: Reading before the level.
        after: Reading after the level.

    Returns:
        The hit rate, or ``unknown`` when a counter is absent or no query ran.
    """
    values = [
        _counter(snap, name)
        for name in (PREFIX_HITS, PREFIX_QUERIES)
        for snap in (before, after)
    ]
    match values:
        case [float() as h0, float() as h1, float() as q0, float() as q1] if q1 > q0:
            return (h1 - h0) / (q1 - q0)
        case _:
            return UNKNOWN


def server_delta(before: str | None, after: str | None) -> dict[str, Any]:
    """Return the server-side histogram and prefix-cache deltas.

    Args:
        before: ``/metrics`` text before the level, or ``None``.
        after: ``/metrics`` text after the level, or ``None``.

    Returns:
        ``e2e``, ``queue``, ``prefill`` and ``prefix_cache_hit_rate``; each is
        ``unknown`` when a reading or its series is absent.
    """
    keys = [*HISTOGRAMS, "prefix_cache_hit_rate"]
    if before is None or after is None:
        return dict.fromkeys(keys, UNKNOWN)
    old, new = snapshot(before), snapshot(after)
    out: dict[str, Any] = {
        key: histogram_delta(old, new, name) for key, name in HISTOGRAMS.items()
    }
    out["prefix_cache_hit_rate"] = prefix_hit_rate(old, new)
    return out


def _gauge(snap: Snapshot, name: str) -> float | str:
    return snap.get((name, None), UNKNOWN)


def _refuse_several_engines(text: str) -> None:
    """Refuse a reading in which one ``GAUGES`` series has several engines.

    Args:
        text: ``/metrics`` exposition text.

    Raises:
        ValueError: When a ``GAUGES`` series carries more than one distinct
            ``engine`` label, because ``snapshot`` would sum them.
    """
    names = set(GAUGES.values())
    engines: dict[str, set[str]] = {}
    for line in text.splitlines():
        match = _SAMPLE.match(line)
        if match is None or match.group("name") not in names:
            continue
        engine = _ENGINE.search(match.group("labels") or "")
        if engine is not None:
            engines.setdefault(match.group("name"), set()).add(engine.group("engine"))
    for name, seen in sorted(engines.items()):
        if len(seen) > 1:
            msg = (
                f"gauge {name} carries engine labels {sorted(seen)}; a sum "
                "across engines is not one reading"
            )
            raise ValueError(msg)


def run_server_delta(before: str | None, after: str | None) -> dict[str, Any]:
    """Return the ``server_delta`` fields plus counter and gauge readings.

    Args:
        before: ``/metrics`` text before the run, or ``None``.
        after: ``/metrics`` text after the run, or ``None``.

    Returns:
        The ``server_delta`` keys, ``counters`` (``COUNTERS`` key to change)
        and ``gauges`` (``GAUGES`` key to ``before`` and ``after`` values).
        A value is ``unknown`` when a reading or its series is absent; a
        missing reading makes each gauge ``unknown``. A gauge of one engine
        is the sum over its other labels.

    Raises:
        ValueError: When a ``GAUGES`` series in either reading carries more
            than one distinct ``engine`` label (#339).
    """
    for text in (before, after):
        if text is not None:
            _refuse_several_engines(text)
    out = server_delta(before, after)
    if before is None or after is None:
        out["counters"] = dict.fromkeys(COUNTERS, UNKNOWN)
        out["gauges"] = dict.fromkeys(GAUGES, UNKNOWN)
        return out
    old, new = snapshot(before), snapshot(after)
    out["counters"] = {
        key: counter_delta(old, new, name) for key, name in COUNTERS.items()
    }
    out["gauges"] = {
        key: {"before": _gauge(old, name), "after": _gauge(new, name)}
        for key, name in GAUGES.items()
    }
    return out


def _rate(count: int, seconds: float) -> float | None:
    return count / seconds if seconds > 0 else None


def run_throughput(
    latencies: Sequence[float],
    *,
    images_per_judgment: int,
    wall_seconds: float,
    concurrency: int,
    discarded: int,
    server: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return the ``throughput`` receipt block of one image run.

    Rates count the judgments the run kept. Discarded completions reached the
    server but are not in the rates.

    Args:
        latencies: Client seconds per kept judgment.
        images_per_judgment: Images sent with each judgment.
        wall_seconds: Wall time of the whole run.
        concurrency: Most judgments in flight at one time.
        discarded: Judgments after the first failure that were dropped.
        server: ``run_server_delta`` result, or ``None`` when the backend
            has no ``/metrics`` reading.

    Returns:
        Concurrency, wall seconds, judgment and image counts, judgments and
        images per second (``None`` when the wall time is not positive),
        nearest-rank latency percentiles, the discarded count and the server
        block.
    """
    judgments = len(latencies)
    images = judgments * images_per_judgment
    return {
        "concurrency": concurrency,
        "wall_seconds": round(wall_seconds, 3),
        "judgments": judgments,
        "images": images,
        "judgments_per_second": _rate(judgments, wall_seconds),
        "images_per_second": _rate(images, wall_seconds),
        "latency_seconds": latency(latencies),
        "discarded": discarded,
        "server": None if server is None else dict(server),
    }

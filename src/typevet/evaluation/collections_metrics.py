"""vLLM ``/metrics`` deltas and client latency for the collections run ([#236][i236]).

The series names come from vLLM ``v0.30.0`` ``vllm/v1/metrics/loggers.py``:
``vllm:e2e_request_latency_seconds`` (line 842),
``vllm:request_queue_time_seconds`` (line 852),
``vllm:request_prefill_time_seconds`` (line 872), ``vllm:prefix_cache_queries``
(line 590) and ``vllm:prefix_cache_hits`` (line 601). The counters are
``prometheus_client`` counters, so the text exposition adds ``_total``; the
parser also accepts the bare name. Samples with the same name and ``le`` label
are summed across the ``model_name`` and ``engine`` labels. A series that is
absent from either reading is recorded as ``unknown``. Server percentiles are
the upper edge of the first bucket that holds the nearest rank, so they are
bounds, not exact values; ``+Inf`` means the rank is above the last edge.

Attributes:
    UNKNOWN (str): Value recorded for an absent series.
    HISTOGRAMS (dict[str, str]): Receipt key to vLLM histogram name.
    PREFIX_HITS (str): vLLM prefix-cache hit counter name.
    PREFIX_QUERIES (str): vLLM prefix-cache query counter name.
    QUANTILES (tuple[tuple[str, float], ...]): Reported percentiles.

Examples:
    ```python
    from typevet.evaluation.collections_metrics import latency, server_delta

    assert latency([0.2, 0.1])["p50"] == 0.1
    assert server_delta(None, None)["e2e"] == "unknown"
    ```

See Also:
    - [typevet.evaluation.collections_throughput][]: the runner that reads them
    - [typevet.evaluation.vllm_acceptance][]: ``kv_cache_usage`` gauge read

[i236]: https://github.com/Alberto-Codes/typevet/issues/236
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
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
QUANTILES: Final[tuple[tuple[str, float], ...]] = (
    ("p50", 0.5),
    ("p95", 0.95),
    ("p99", 0.99),
)
_SAMPLE: Final = re.compile(
    r"^(?P<name>[A-Za-z_:][A-Za-z0-9_:]*)(?:\{(?P<labels>[^}]*)\})?\s+(?P<value>\S+)"
)
_LE: Final = re.compile(r'(?:^|,)\s*le="(?P<le>[^"]*)"')

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

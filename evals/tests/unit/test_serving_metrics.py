"""Offline tests for the shared run rates and vLLM ``/metrics`` deltas (#335).

The ``/metrics`` text is a fixed fixture in the vLLM ``v0.30.0`` exposition
shape. No test reads the network.
"""

from __future__ import annotations

import pytest

from typevet_evals.serving_metrics import (
    COUNTERS,
    GAUGES,
    counter_delta,
    latency,
    nearest_rank,
    run_server_delta,
    run_throughput,
    snapshot,
)
from typevet_evals.throughput import collections_metrics

pytestmark = pytest.mark.unit

_BEFORE = """\
# HELP vllm:prefix_cache_hits Prefix cache hits, in terms of number of cached tokens.
# TYPE vllm:prefix_cache_hits counter
vllm:prefix_cache_hits_total{engine="0",model_name="m"} 10.0
vllm:prefix_cache_queries_total{engine="0",model_name="m"} 40.0
vllm:mm_cache_hits_total{engine="0",model_name="m"} 1.0
vllm:mm_cache_queries_total{engine="0",model_name="m"} 4.0
vllm:prompt_tokens_total{engine="0",model_name="m"} 500.0
vllm:prompt_tokens_cached_total{engine="0",model_name="m"} 8.0
vllm:generation_tokens_total{engine="0",model_name="m"} 7.0
vllm:request_success_total{engine="0",finished_reason="stop",model_name="m"} 2.0
vllm:request_success_total{engine="0",finished_reason="abort",model_name="m"} 1.0
vllm:num_preemptions_total{engine="0",model_name="m"} 0.0
vllm:kv_cache_usage_perc{engine="0",model_name="m"} 0.1
vllm:num_requests_running{engine="0",model_name="m"} 0.0
vllm:num_requests_waiting{engine="0",model_name="m"} 0.0
vllm:e2e_request_latency_seconds_bucket{le="1.0",model_name="m"} 1.0
vllm:e2e_request_latency_seconds_bucket{le="+Inf",model_name="m"} 3.0
vllm:e2e_request_latency_seconds_count{model_name="m"} 3.0
vllm:e2e_request_latency_seconds_sum{model_name="m"} 6.0
"""
_AFTER = """\
vllm:prefix_cache_hits_total{engine="0",model_name="m"} 70.0
vllm:prefix_cache_queries_total{engine="0",model_name="m"} 120.0
vllm:mm_cache_hits_total{engine="0",model_name="m"} 7.0
vllm:mm_cache_queries_total{engine="0",model_name="m"} 12.0
vllm:prompt_tokens_total{engine="0",model_name="m"} 2500.0
vllm:prompt_tokens_cached_total{engine="0",model_name="m"} 1208.0
vllm:generation_tokens_total{engine="0",model_name="m"} 15.0
vllm:request_success_total{engine="0",finished_reason="stop",model_name="m"} 9.0
vllm:request_success_total{engine="0",finished_reason="abort",model_name="m"} 2.0
vllm:num_preemptions_total{engine="0",model_name="m"} 1.0
vllm:kv_cache_usage_perc{engine="0",model_name="m"} 0.4
vllm:num_requests_running{engine="0",model_name="m"} 3.0
vllm:num_requests_waiting{engine="0",model_name="m"} 2.0
vllm:e2e_request_latency_seconds_bucket{le="1.0",model_name="m"} 5.0
vllm:e2e_request_latency_seconds_bucket{le="+Inf",model_name="m"} 11.0
vllm:e2e_request_latency_seconds_count{model_name="m"} 11.0
vllm:e2e_request_latency_seconds_sum{model_name="m"} 22.0
"""


def test_counter_and_gauge_deltas_from_fixed_metrics_text() -> None:
    delta = run_server_delta(_BEFORE, _AFTER)

    assert delta["counters"] == {
        "prefix_cache_hits": 60.0,
        "prefix_cache_queries": 80.0,
        "mm_cache_hits": 6.0,
        "mm_cache_queries": 8.0,
        "prompt_tokens": 2000.0,
        "prompt_tokens_cached": 1200.0,
        "generation_tokens": 8.0,
        "request_success": 8.0,
        "preemptions": 1.0,
    }
    assert delta["gauges"] == {
        "kv_cache_usage_perc": {"before": 0.1, "after": 0.4},
        "num_requests_running": {"before": 0.0, "after": 3.0},
        "num_requests_waiting": {"before": 0.0, "after": 2.0},
    }
    assert delta["prefix_cache_hit_rate"] == pytest.approx(0.75)
    # 8 requests: 4 at or below 1 s, 4 above; p50 is the 1 s edge.
    assert delta["e2e"]["count"] == 8.0
    assert delta["e2e"]["p50"] == 1.0
    assert delta["e2e"]["p95"] == "+Inf"
    assert delta["queue"] == "unknown"


def test_a_gauge_of_two_engines_is_refused() -> None:
    two = _AFTER + 'vllm:kv_cache_usage_perc{engine="1",model_name="m"} 0.3\n'

    with pytest.raises(ValueError, match="engine") as caught:
        run_server_delta(_BEFORE, two)
    assert "vllm:kv_cache_usage_perc" in str(caught.value)
    with pytest.raises(ValueError, match="engine"):
        run_server_delta(two, _AFTER)


def test_a_gauge_mixing_labelled_and_unlabelled_samples_is_refused() -> None:
    # A sample without an engine label counts as its own engine (#346).
    mixed = _AFTER + 'vllm:kv_cache_usage_perc{model_name="m"} 0.3\n'

    with pytest.raises(ValueError, match="engine") as caught:
        run_server_delta(_BEFORE, mixed)
    assert "vllm:kv_cache_usage_perc" in str(caught.value)
    with pytest.raises(ValueError, match="engine"):
        run_server_delta(mixed, _AFTER)


def test_a_gauge_of_unlabelled_samples_sums_as_one_engine() -> None:
    before = "vllm:num_requests_running 1.0\n"
    after = 'vllm:num_requests_running{model_name="m"} 2.0\n'
    after += 'vllm:num_requests_running{model_name="n"} 3.0\n'

    delta = run_server_delta(before, after)

    assert delta["gauges"]["num_requests_running"] == {"before": 1.0, "after": 5.0}


def test_a_gauge_of_one_engine_sums_over_its_other_labels() -> None:
    one = _AFTER + 'vllm:num_requests_running{engine="0",model_name="n"} 4.0\n'

    delta = run_server_delta(_BEFORE, one)

    assert delta["gauges"]["num_requests_running"] == {"before": 0.0, "after": 7.0}


def test_metric_names_follow_the_vllm_pin() -> None:
    assert COUNTERS["prefix_cache_hits"] == "vllm:prefix_cache_hits"
    assert COUNTERS["prefix_cache_queries"] == "vllm:prefix_cache_queries"
    assert COUNTERS["request_success"] == "vllm:request_success"
    assert GAUGES["kv_cache_usage_perc"] == "vllm:kv_cache_usage_perc"


def test_absent_series_and_bare_counter_names() -> None:
    before = snapshot("vllm:prompt_tokens 5\n")
    after = snapshot("vllm:prompt_tokens 9\n")

    assert counter_delta(before, after, "vllm:prompt_tokens") == 4.0
    assert counter_delta(before, snapshot(""), "vllm:prompt_tokens") == "unknown"
    delta = run_server_delta("vllm:prompt_tokens_total 1\n", "vllm:prompt_tokens 3\n")
    assert delta["counters"]["prompt_tokens"] == 2.0
    assert delta["counters"]["generation_tokens"] == "unknown"
    assert delta["gauges"]["kv_cache_usage_perc"] == {
        "before": "unknown",
        "after": "unknown",
    }


def test_missing_reading_is_unknown() -> None:
    delta = run_server_delta(_BEFORE, None)

    assert set(delta["counters"]) == set(COUNTERS)
    assert set(delta["counters"].values()) == {"unknown"}
    assert set(delta["gauges"]) == set(GAUGES)
    assert set(delta["gauges"].values()) == {"unknown"}
    assert delta["e2e"] == "unknown"


def test_run_throughput_rates_and_percentiles() -> None:
    block = run_throughput(
        [4.0, 1.0, 3.0, 2.0],
        images_per_judgment=2,
        wall_seconds=2.0,
        concurrency=4,
        discarded=3,
        server=None,
    )

    assert block == {
        "concurrency": 4,
        "wall_seconds": 2.0,
        "judgments": 4,
        "images": 8,
        "judgments_per_second": 2.0,
        "images_per_second": 4.0,
        "latency_seconds": {"n": 4, "p50": 2.0, "p95": 4.0, "p99": 4.0},
        "discarded": 3,
        "server": None,
    }


def test_five_value_percentiles_use_the_ceiling_rank() -> None:
    # Ranks ceil(2.5) = 3, ceil(4.75) = 5, ceil(4.95) = 5.
    assert latency([5.0, 1.0, 4.0, 2.0, 3.0]) == {
        "n": 5,
        "p50": 3.0,
        "p95": 5.0,
        "p99": 5.0,
    }


@pytest.mark.parametrize("quantile", [0.5, 0.95, 0.99])
def test_collections_wrapper_matches_the_shared_nearest_rank(quantile: float) -> None:
    ordered = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert collections_metrics.nearest_rank(ordered, quantile) == nearest_rank(
        ordered, quantile
    )

"""Offline tests for the collections throughput runner ([#236][i236]).

The mock server answers scoring calls with the redacted vLLM v0.30.0 probe
``image_three_way`` and ``/tokenize`` with ``tokenize_ordinals``. ``/version``,
``/v1/models`` and ``/metrics`` replies are synthetic. A record whose state
holds ``FAIL_MARK`` gets an HTTP 500 on every scoring call. Records are
synthetic and use partner field names only.

Examples:
    ```bash
    uv run pytest -q evals/tests/unit/test_collections_throughput.py
    ```

See Also:
    - [typevet_evals.throughput.collections_throughput][]: the runner
    - [typevet_evals.throughput.collections_metrics][]: /metrics deltas

[i236]: https://github.com/Alberto-Codes/typevet/issues/236
"""

from __future__ import annotations

import copy
import json
import threading
import zlib
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx
import pytest

from evals.tests.unit.test_server_args import CACHE_CONFIG, CACHE_CONFIG_TEXT, STATED
from typevet.domain.judgment_questions import Choice, Noul
from typevet_evals.throughput.collections_metrics import (
    histogram_delta,
    prefix_hit_rate,
    read_metrics,
    server_delta,
    snapshot,
)
from typevet_evals.throughput.collections_throughput import (
    CAPS,
    LEVELS,
    METHOD,
    RUN_SECONDS,
    RunOptions,
    best_level,
    level_timing,
    remaining_run_seconds,
    run_throughput,
)
from typevet_evals.throughput.collections_workload import CollectionsRecord, ece
from typevet_evals.vllm_acceptance.core import CallCaps

pytestmark = pytest.mark.unit

_VLLM = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "vllm"
MODEL = "gemma-4-31b-it"
KEY = "sk-SENTINEL-236A2"
AGENT = "typevet-236-test/1.0"
FAIL_MARK = "FAIL-236"
QUESTIONS: dict[str, Noul | Choice] = {
    "will_engage": Noul(
        instructions="Will the customer engage?",
        criteria={"true": "Engages.", "false": "Does not."},
    ),
    "accepted_offer": Choice(
        instructions="Which offer?",
        criteria={"NONE": "None.", "PLAN_A": "Plan A."},
    ),
}
_BEFORE = """\
# HELP vllm:e2e_request_latency_seconds Histogram of e2e request latency in seconds.
# TYPE vllm:e2e_request_latency_seconds histogram
vllm:e2e_request_latency_seconds_bucket{engine="0",le="0.5",model_name="m"} 1.0
vllm:e2e_request_latency_seconds_bucket{engine="0",le="1.0",model_name="m"} 2.0
vllm:e2e_request_latency_seconds_bucket{engine="0",le="+Inf",model_name="m"} 2.0
vllm:e2e_request_latency_seconds_count{engine="0",model_name="m"} 2.0
vllm:e2e_request_latency_seconds_sum{engine="0",model_name="m"} 1.1
vllm:e2e_request_latency_seconds_created{engine="0",model_name="m"} 1.7e9
vllm:prefix_cache_queries_total{engine="0",model_name="m"} 100.0
vllm:prefix_cache_hits_total{engine="0",model_name="m"} 10.0
"""
_AFTER = """\
vllm:e2e_request_latency_seconds_bucket{engine="0",le="0.5",model_name="m"} 51.0
vllm:e2e_request_latency_seconds_bucket{engine="0",le="1.0",model_name="m"} 97.0
vllm:e2e_request_latency_seconds_bucket{engine="0",le="+Inf",model_name="m"} 102.0
vllm:e2e_request_latency_seconds_count{engine="0",model_name="m"} 102.0
vllm:e2e_request_latency_seconds_sum{engine="0",model_name="m"} 81.1
vllm:prefix_cache_queries_total{engine="0",model_name="m"} 1100.0
vllm:prefix_cache_hits_total{engine="0",model_name="m"} 810.0
"""


def _fixture(name: str) -> dict[str, Any]:
    return json.loads((_VLLM / f"{name}.json").read_text(encoding="utf-8"))


def env(**extra: str) -> dict[str, str]:
    """Return a vLLM environment with a key and a user agent."""
    return {
        "TYPEVET_BACKEND": "vllm",
        "TYPEVET_VLLM__BASE_URL": "http://vllm.test:8000",
        "TYPEVET_VLLM__MODEL": MODEL,
        "TYPEVET_VLLM__API_KEY": KEY,
        "TYPEVET_VLLM__USER_AGENT": AGENT,
        **extra,
    }


def records(count: int, *, failing: int = 0) -> list[CollectionsRecord]:
    """Build ``count`` records; the first ``failing`` get a failing state."""
    return [
        CollectionsRecord(
            state={
                "balance": 100 + i,
                "note": FAIL_MARK if i < failing else "ok",
                "proposed_action": {"action": "PLAN_A"},
            },
            label="engaged" if i % 2 == 0 else "not_engaged",
            accepted_offer="NONE",
        )
        for i in range(count)
    ]


class FakeVllm:
    """MockTransport handler for the endpoints the runner calls.

    ``hold`` blocks each scoring call until ``hold`` calls are in flight or
    one second passed, so the peak in-flight count is observable. A level
    reads ``/metrics`` three times (before, in flight, after), so every third
    read, starting with the first, returns the ``before`` text.
    """

    def __init__(self, *, hold: int = 0, echo_key: bool = False) -> None:
        """Start with no requests; ``echo_key`` puts the auth header in errors."""
        self.requests: list[httpx.Request] = []
        self.hold = hold
        self.echo_key = echo_key
        self.in_flight = 0
        self.peak = 0
        self.metrics_reads = 0
        self._cond = threading.Condition()
        self._tokens = _fixture("tokenize_ordinals")["responses"]
        self._scoring = _fixture("image_three_way")["response"]

    def __call__(self, request: httpx.Request) -> httpx.Response:
        with self._cond:
            self.requests.append(request)
        if request.method == "GET":
            return self._get(request.url.path)
        body = json.loads(request.content.decode())
        if request.url.path == "/tokenize":
            return httpx.Response(200, json=self._tokenize(body["prompt"]))
        return self._chat(request, body)

    def _get(self, path: str) -> httpx.Response:
        if path == "/version":
            return httpx.Response(200, json={"version": "0.30.0"})
        if path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": MODEL}]})
        with self._cond:
            self.metrics_reads += 1
            text = _BEFORE if self.metrics_reads % 3 == 1 else _AFTER
        text += CACHE_CONFIG_TEXT
        return httpx.Response(200, text=text + 'vllm:kv_cache_usage_perc{e="0"} 0.25\n')

    def _tokenize(self, prompt: str) -> dict[str, Any]:
        if prompt in self._tokens:
            return self._tokens[prompt]
        return {"count": 1, "tokens": [200000 + zlib.crc32(prompt.encode()) % 50000]}

    def _chat(self, request: httpx.Request, body: dict[str, Any]) -> httpx.Response:
        with self._cond:
            self.in_flight += 1
            self.peak = max(self.peak, self.in_flight)
            self._cond.notify_all()
            self._cond.wait_for(lambda: self.in_flight >= self.hold, timeout=1.0)
        try:
            if FAIL_MARK in json.dumps(body):
                auth = request.headers.get("authorization", "") if self.echo_key else ""
                return httpx.Response(500, json={"error": f"boom {auth}"})
            reply = copy.deepcopy(self._scoring)
            top = [
                {"token": f"token_id:{token}", "logprob": -0.5 - rank}
                for rank, token in enumerate(body["logprob_token_ids"])
            ]
            reply["choices"][0]["logprobs"]["content"][0]["top_logprobs"] = top
            reply["usage"] = {"prompt_tokens": 90, "completion_tokens": 1}
            return httpx.Response(200, json=reply)
        finally:
            with self._cond:
                self.in_flight -= 1


def run(
    server: FakeVllm,
    rows: list[CollectionsRecord],
    levels: tuple[int, ...] = (1,),
    caps: CallCaps = CAPS,
    options: RunOptions | None = None,
) -> dict[str, Any]:
    """Run the throughput sweep against ``server``."""
    return run_throughput(
        env(),
        rows,
        QUESTIONS,
        transport=httpx.MockTransport(server),
        levels=levels,
        caps=caps,
        options=options or RunOptions(),
    )


def test_defaults_match_the_contract() -> None:
    assert LEVELS == (1, 8, 32, 64)
    assert CallCaps(model=11000, tokenizer=256, metadata=40) == CAPS
    assert RunOptions().level_seconds == 900.0
    assert RunOptions().run_seconds == 2700.0


def test_histogram_and_prefix_deltas_from_fixed_metrics_text() -> None:
    delta = server_delta(_BEFORE, _AFTER)

    e2e = delta["e2e"]
    assert e2e["count"] == 100
    assert e2e["sum"] == pytest.approx(80.0)
    assert (e2e["p50"], e2e["p95"], e2e["p99"]) == (0.5, 1.0, "+Inf")
    assert e2e["percentile_basis"] == "bucket_upper_bound"
    assert delta["queue"] == "unknown"
    assert delta["prefill"] == "unknown"
    assert delta["prefix_cache_hit_rate"] == pytest.approx(0.8)


def test_absent_metrics_text_is_unknown() -> None:
    delta = server_delta(None, _AFTER)

    assert delta == {
        "e2e": "unknown",
        "queue": "unknown",
        "prefill": "unknown",
        "prefix_cache_hit_rate": "unknown",
    }


def test_bare_counters_bucket_mismatch_and_unreadable_values() -> None:
    before = snapshot("vllm:prefix_cache_hits 1\nvllm:prefix_cache_queries 4\nx abc\n")
    after = snapshot("vllm:prefix_cache_hits 4\nvllm:prefix_cache_queries 10\n")
    same = snapshot("vllm:prefix_cache_hits 4\nvllm:prefix_cache_queries 4\n")
    old = snapshot(_BEFORE)
    shifted = snapshot(_AFTER.replace('le="0.5"', 'le="0.25"'))

    assert ("x", None) not in before
    assert prefix_hit_rate(before, after) == pytest.approx(0.5)
    assert prefix_hit_rate(same, same) == "unknown"
    assert prefix_hit_rate(snapshot(""), after) == "unknown"
    assert histogram_delta(old, shifted, "vllm:e2e_request_latency_seconds") == (
        "unknown"
    )
    flat = histogram_delta(old, old, "vllm:e2e_request_latency_seconds")
    assert flat == {
        "count": 0.0,
        "sum": 0.0,
        "p50": None,
        "p95": None,
        "p99": None,
        "percentile_basis": "bucket_upper_bound",
    }


def test_failed_metrics_reads_are_unknown() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down", request=request)

    down = httpx.Client(transport=httpx.MockTransport(refuse), base_url="http://x")
    error = httpx.Client(
        transport=httpx.MockTransport(lambda _r: httpx.Response(500)),
        base_url="http://x",
    )

    assert read_metrics(down) is None
    assert read_metrics(error) is None


class _NoMetrics(FakeVllm):
    def _get(self, path: str) -> httpx.Response:
        if path == "/v1/models":
            return super()._get(path)
        return httpx.Response(500, text="not json")


def test_run_without_metrics_or_version_records_unknown() -> None:
    receipt = run(_NoMetrics(), records(2))

    (level,) = receipt["levels"]
    assert receipt["pins"]["version"] == "unknown"
    assert level["kv_cache_usage"] == "unknown"
    assert set(level["server"].values()) == {"unknown"}
    assert level["stopped"] is None


def test_level_stops_at_six_errors_of_592_and_no_higher_level_runs() -> None:
    server = FakeVllm()
    receipt = run(server, records(592, failing=6), levels=(1, 8))

    (level,) = receipt["levels"]
    assert level["errors"] == 6
    assert level["sent"] == 6
    assert level["stopped"] == "error rate above 0.01"
    assert receipt["stopped"] == "level 1: error rate above 0.01"
    assert receipt["best_level"] == "unknown"
    assert level["parity"]["failures"] == 592
    assert level["parity"]["meets_parity"] is False
    assert {e["type"] for e in level["error_rows"]} == {"BackendHttpError"}


def test_five_errors_of_592_do_not_stop_the_level() -> None:
    receipt = run(FakeVllm(), records(592, failing=5), levels=(8,))

    (level,) = receipt["levels"]
    assert level["stopped"] is None
    assert level["errors"] == 5
    assert level["answered"] == 587
    assert level["error_rate"] == pytest.approx(5 / 592)
    assert receipt["stopped"] is None
    assert receipt["best_level"] == 8


def test_model_call_cap_stop_still_returns_a_receipt() -> None:
    caps = CallCaps(model=3, tokenizer=256, metadata=40)
    receipt = run(FakeVllm(), records(4), levels=(1, 8), caps=caps)

    (level,) = receipt["levels"]
    assert level["stopped"] == "model call cap 3 reached"
    assert receipt["stopped"] == "level 1: model call cap 3 reached"
    assert receipt["calls"]["model"] == 3
    assert level["answered"] == 1


def test_metadata_cap_stop_still_returns_a_receipt() -> None:
    caps = CallCaps(model=100, tokenizer=256, metadata=1)
    receipt = run(FakeVllm(), records(2), caps=caps)

    assert receipt["stopped"] == "metadata call cap 1 reached"
    assert receipt["levels"] == []


def test_level_time_cap_stops_before_sending() -> None:
    receipt = run(
        FakeVllm(), records(3), levels=(1, 8), options=RunOptions(level_seconds=0.0)
    )

    (level,) = receipt["levels"]
    assert level["stopped"] == "time cap"
    assert level["sent"] == 0
    assert receipt["calls"]["model"] == 0


def test_run_time_cap_bounds_every_level() -> None:
    receipt = run(FakeVllm(), records(3), options=RunOptions(run_seconds=0.0))

    assert receipt["levels"][0]["stopped"] == "time cap"


def test_level_measures_and_parity_wiring() -> None:
    server = FakeVllm()
    receipt = run(server, records(1), levels=(1,))
    p = 1.0 - receipt["levels"][0]["parity"]["ece"]

    labels = [i < 3 for i in range(10)]
    rows = [
        replace(row, label="engaged" if y else "not_engaged")
        for row, y in zip(records(10), labels, strict=True)
    ]
    receipt = run(FakeVllm(), rows, levels=(1, 8))

    assert [lv["level"] for lv in receipt["levels"]] == [1, 8]
    level = receipt["levels"][1]
    assert level["parity"]["ece"] == pytest.approx(ece([p] * 10, labels))
    assert level["parity"]["base_rate"] == pytest.approx(0.3)
    assert level["parity"]["scored"] == 10
    assert level["scoring_calls"] == 20
    assert level["latency"]["n"] == 10
    assert level["latency"]["p99"] >= level["latency"]["p50"] > 0
    assert level["prompt_tokens"]["max"] == 180
    assert level["kv_cache_usage"] == 0.25
    assert level["server"]["e2e"]["count"] == 100
    assert level["server"]["prefix_cache_hit_rate"] == pytest.approx(0.8)
    assert level["records_per_second"] > 0
    assert receipt["pins"]["served_models"] == [MODEL]
    assert receipt["cold_start_seconds"] == "unknown"
    assert receipt["cost_per_1000"] == "unknown"
    assert receipt["gpu_memory"] == "unknown"


def test_supplied_hourly_rate_gives_cost_per_1000_at_best_level() -> None:
    options = RunOptions(supplied={"hourly_usd": 3.6, "gpu_memory": "80 GiB"})
    receipt = run(FakeVllm(), records(4), options=options)

    rps = receipt["levels"][0]["records_per_second"]
    assert receipt["hourly_usd"] == 3.6
    assert receipt["gpu_memory"] == "80 GiB"
    assert receipt["cold_start_seconds"] == "unknown"
    assert receipt["cost_per_1000"] == pytest.approx(3.6 / 3600 * 1000 / rps)


def test_non_vllm_backend_is_refused() -> None:
    with pytest.raises(ValueError, match="vllm"):
        run_throughput(
            {**env(), "TYPEVET_BACKEND": "llama_cpp"},
            records(1),
            QUESTIONS,
            transport=httpx.MockTransport(FakeVllm()),
        )


def _level(level: int, rps: float | None, rate: float, stopped: str | None = None):
    return {
        "level": level,
        "records_per_second": rps,
        "error_rate": rate,
        "stopped": stopped,
    }


def test_best_level_is_the_fastest_level_at_or_below_one_percent_errors() -> None:
    levels = [
        _level(1, 2.0, 0.0),
        _level(8, 9.0, 0.01),
        _level(32, 30.0, 0.02),
        _level(64, 40.0, 0.0, stopped="time cap"),
    ]

    assert best_level(levels) == 8
    assert best_level([_level(1, None, 0.0)]) == "unknown"
    assert best_level([]) == "unknown"


def test_wall_clock_is_last_end_minus_first_start_and_rate_counts_answers() -> None:
    rows = [
        {"p": 0.5, "started": 10.0, "ended": 13.0},
        {"error": {"type": "BackendHttpError"}, "started": 11.0, "ended": 12.0},
        {"p": 0.5, "started": 12.0, "ended": 15.0},
        {"p": 0.5, "started": 14.0, "ended": 18.0},
    ]

    timing = level_timing(rows)

    assert timing["wall_seconds"] == 8.0
    assert timing["records_per_second"] == pytest.approx(3 / 8.0)
    assert level_timing([]) == {"wall_seconds": 0.0, "records_per_second": None}


def test_second_run_gets_only_the_time_the_sweep_left() -> None:
    sweep = run(FakeVllm(), records(2))

    assert 0 < sweep["elapsed_seconds"] < RUN_SECONDS
    assert remaining_run_seconds(sweep) == pytest.approx(
        RUN_SECONDS - sweep["elapsed_seconds"]
    )
    assert remaining_run_seconds({"elapsed_seconds": RUN_SECONDS + 5}) == 0.0

    spent = RunOptions(run_seconds=remaining_run_seconds({"elapsed_seconds": 2700}))
    second = run(FakeVllm(), records(3), levels=(8,), options=spent)

    assert second["levels"][0]["stopped"] == "time cap"
    assert second["levels"][0]["sent"] == 0
    assert second["calls"]["model"] == 0


def test_receipt_states_its_method() -> None:
    receipt = run(FakeVllm(), records(1))

    assert receipt["method"] == dict(METHOD)
    assert "before each send" in METHOD["time_cap"]
    assert "failed" in METHOD["client_latency"]
    assert "+Inf" in METHOD["server_percentiles"]


def test_receipt_records_cache_config_and_stated_server_args() -> None:
    server = FakeVllm()
    receipt = run(server, records(2), options=RunOptions(caller_stated=STATED))

    assert receipt["server_args"] == {
        "cache_config": CACHE_CONFIG,
        "cache_config_source": "metrics",
        "caller_stated": STATED,
    }
    assert server.metrics_reads == 3


def test_stop_before_any_level_still_records_server_args() -> None:
    caps = CallCaps(model=100, tokenizer=256, metadata=1)
    receipt = run(FakeVllm(), records(2), caps=caps)

    assert receipt["server_args"] == {
        "cache_config": None,
        "cache_config_source": "metrics",
        "caller_stated": None,
    }

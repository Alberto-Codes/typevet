"""Offline tests for the ``concurrency`` option of the image runners (#334).

Each test runs the face, check and signature runners through one fake
judgment port. The port finds the request index from the image bytes, so it
can hold calls at a barrier, finish them in reverse order or fail one index.
Order and stop tests wait on ``threading`` events, not on timed pauses, so a
slow runner cannot change their result (#339). No test reads the network or a
real image. The ``throughput`` receipt tests
(#335) use a scripted clock and fixed ``/metrics`` text.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ALL_COMPLETED
from dataclasses import dataclass
from typing import Any

import pytest

from typevet.domain import (
    Choice,
    ChoiceAnswer,
    ImageInput,
    JudgmentResponse,
    Noul,
    NoulAnswer,
    Question,
    Score,
    ScoreAnswer,
    TokenUsage,
)
from typevet.domain.errors import TransportError
from typevet_evals.check_match import (
    build_check_match_receipt,
    build_check_match_request,
    check_cases,
    run_check_match,
)
from typevet_evals.datasets.cedar import CedarPair, CedarSignature, PairKind
from typevet_evals.datasets.lfw import LfwFace, LfwPair
from typevet_evals.face_match import (
    IMAGE_CONCURRENCY_ENV,
    build_face_match_receipt,
    build_face_match_request,
    image_concurrency,
    pool,
    run_face_match,
)
from typevet_evals.signature_match import (
    build_signature_match_receipt,
    build_signature_match_request,
    run_signature_match,
)

pytestmark = pytest.mark.unit

_BARRIER_TIMEOUT = 5.0


def _image(index: int) -> bytes:
    return f"synthetic image {index:03d}".encode()


def _index(media: tuple[ImageInput, ...] | None) -> int:
    assert media
    return int(media[0].data.decode().rsplit(" ", 1)[1])


def _answers(questions: Mapping[str, Question | Mapping[str, Any]]) -> dict:
    answers: dict[str, object] = {}
    for name, question in questions.items():
        if isinstance(question, Noul):
            answers[name] = NoulAnswer(noul=0.5)
        elif isinstance(question, Choice):
            labels = list(question.criteria)
            share = 1.0 / len(labels)
            answers[name] = ChoiceAnswer(
                choice=labels[0],
                confidence=share,
                probabilities=dict.fromkeys(labels, share),
            )
        else:
            assert isinstance(question, Score)
            levels = range(len(question.criteria))
            probabilities = dict.fromkeys(levels, 0.0)
            probabilities[0] = 1.0
            answers[name] = ScoreAnswer(
                score=0.0,
                confidence=1.0,
                legend={i: f"level {i}" for i in levels},
                probabilities=probabilities,
            )
    return answers


class _CallClock:
    """Clock that reads 0 except right after a call the port has timed.

    ``judge_in_order`` reads the clock on the worker thread before and after
    each call. The port sets the seconds of its call on that thread, so the
    reading after the call gives that latency exactly, whatever the thread
    schedule.
    """

    def __init__(self) -> None:
        self._local = threading.local()

    def set(self, seconds: float) -> None:
        """Make the next reading on this thread return ``seconds``.

        Args:
            seconds: The latency of the current call.
        """
        self._local.pending = seconds

    def __call__(self) -> float:
        """Return and clear this thread's pending reading, or 0.

        Returns:
            The seconds the port set on this thread, else 0.
        """
        pending = getattr(self._local, "pending", None)
        self._local.pending = None
        return 0.0 if pending is None else pending


class _IndexedPort:
    """Judgment port that acts on the request index found in the image.

    A call whose index is in ``hold`` waits for that event first. Each call
    then records its index in ``finished`` and sets its ``done`` event, before
    it returns or raises.
    """

    def __init__(
        self,
        *,
        barrier: threading.Barrier | None = None,
        hold: Mapping[int, threading.Event] | None = None,
        failing: frozenset[int] = frozenset(),
        clock: _CallClock | None = None,
        latency: Callable[[int], float] = lambda _: 0.0,
    ) -> None:
        self._barrier = barrier
        self.hold: dict[int, threading.Event] = dict(hold or {})
        self._failing = failing
        self._clock = clock
        self._latency = latency
        self._lock = threading.Lock()
        self._done: dict[int, threading.Event] = {}
        self.called: list[int] = []
        self.finished: list[int] = []

    def done(self, index: int) -> threading.Event:
        """Return the event that the call of ``index`` sets as it ends.

        Args:
            index: A request index.

        Returns:
            The event of that index.
        """
        with self._lock:
            return self._done.setdefault(index, threading.Event())

    def judge(
        self,
        state: str | dict[str, Any] | list[Any],
        questions: Mapping[str, Question | Mapping[str, Any]],
        model: str,
        *,
        media: tuple[ImageInput, ...] | None = None,
    ) -> JudgmentResponse:
        del state
        index = _index(media)
        with self._lock:
            self.called.append(index)
        if self._barrier is not None:
            self._barrier.wait()
        gate = self.hold.get(index)
        if gate is not None and not gate.wait(_BARRIER_TIMEOUT):
            msg = f"call {index} was never released"
            raise AssertionError(msg)
        if self._clock is not None:
            self._clock.set(self._latency(index))
        with self._lock:
            self.finished.append(index)
        self.done(index).set()
        if index in self._failing:
            msg = f"refused {index}"
            raise TransportError(msg)
        return JudgmentResponse(
            model=model, usage=TokenUsage(input_tokens=10), answers=_answers(questions)
        )


@pytest.fixture
def failure_seen(monkeypatch: pytest.MonkeyPatch) -> threading.Event:
    """Return an event the pooled loop sets once it has a failed call in hand.

    The fixture wraps the ``wait`` of ``judge_in_order``. The event is set
    after ``wait`` gives the loop a failed future and before the loop reads
    it, so a call held on this event can only finish after the loop has
    stopped new submissions.

    Args:
        monkeypatch: The pytest monkeypatch fixture.

    Returns:
        The event.
    """
    seen = threading.Event()
    real_wait = pool.wait

    def spy(fs: Any, timeout: float | None = None, return_when: str = ALL_COMPLETED):
        result = real_wait(fs, timeout=timeout, return_when=return_when)
        if any(f.exception() is not None for f in result.done):
            seen.set()
        return result

    monkeypatch.setattr(pool, "wait", spy)
    return seen


@dataclass(frozen=True)
class _Family:
    name: str
    id_key: str
    build: Callable[[int], list[Any]]
    run: Callable[..., Any]


def _face_requests(count: int) -> list[Any]:
    return [
        build_face_match_request(
            LfwPair(
                1,
                LfwFace("Alpha_Example", i + 1),
                LfwFace("Bravo_Example", i + 2),
                same_person=False,
            ),
            left_image=_image(i),
            right_image=_image(i),
        )
        for i in range(count)
    ]


def _check_requests(count: int) -> list[Any]:
    cases = check_cases(count=count)[:count]
    return [build_check_match_request(c, image=_image(i)) for i, c in enumerate(cases)]


def _signature_requests(count: int) -> list[Any]:
    return [
        build_signature_match_request(
            CedarPair(
                PairKind.GENUINE_GENUINE,
                CedarSignature(i + 1, 1, forged=False),
                CedarSignature(i + 1, 2, forged=False),
            ),
            reference_image=_image(i),
            questioned_image=_image(i),
        )
        for i in range(count)
    ]


FAMILIES = [
    _Family("face", "pair_id", _face_requests, run_face_match),
    _Family("check", "case_id", _check_requests, run_check_match),
    _Family("signature", "pair_id", _signature_requests, run_signature_match),
]
_RECEIPTS: dict[str, Callable[..., dict[str, Any]]] = {
    "face": build_face_match_receipt,
    "check": build_check_match_receipt,
    "signature": build_signature_match_receipt,
}
_IMAGES_PER_JUDGMENT = {"face": 2, "check": 1, "signature": 2}
_IDS = [f.name for f in FAMILIES]


def _ids(items: Sequence[Any], key: str) -> list[str]:
    return [getattr(item, key) for item in items]


@pytest.mark.parametrize("family", FAMILIES, ids=_IDS)
def test_concurrency_runs_that_many_judgments_at_once(family: _Family) -> None:
    # Four calls meet at a four-party barrier; one at a time breaks it.
    requests = family.build(8)
    port = _IndexedPort(barrier=threading.Barrier(4, timeout=_BARRIER_TIMEOUT))
    run = family.run(port, requests, "m", concurrency=4)
    assert run.stopped is None
    assert _ids(run.outcomes, family.id_key) == _ids(requests, family.id_key)
    assert sorted(port.called) == list(range(8))


@pytest.mark.parametrize("family", FAMILIES, ids=_IDS)
def test_outcomes_keep_slice_order_when_calls_finish_in_reverse(
    family: _Family,
) -> None:
    # Call i waits until call i + 1 has ended, so the calls end 5, 4, ..., 0.
    # Call i takes 6 - i seconds on the call clock.
    requests = family.build(6)
    clock = _CallClock()
    port = _IndexedPort(clock=clock, latency=lambda i: 6.0 - i)
    port.hold.update({i: port.done(i + 1) for i in range(5)})
    run = family.run(port, requests, "m", concurrency=6, clock=clock)
    assert run.stopped is None
    assert port.finished == [5, 4, 3, 2, 1, 0]
    assert _ids(run.outcomes, family.id_key) == _ids(requests, family.id_key)
    assert [o.latency_seconds for o in run.outcomes] == [6.0, 5.0, 4.0, 3.0, 2.0, 1.0]


@pytest.mark.parametrize("family", FAMILIES, ids=_IDS)
def test_failure_stops_new_submissions_and_is_recorded(
    family: _Family, failure_seen: threading.Event
) -> None:
    # Index 3 fails at once. Index 2 and each index after 3 are held until
    # the loop has the failure in hand, so index 2 is in flight when 3
    # fails. With two in flight, no index after 4 is ever sent.
    requests = family.build(10)
    held = (2, *range(4, 10))
    port = _IndexedPort(hold=dict.fromkeys(held, failure_seen), failing=frozenset({3}))
    run = family.run(port, requests, "m", concurrency=2)
    assert _ids(run.outcomes, family.id_key) == _ids(requests[:3], family.id_key)
    assert run.stopped == {
        "index": 3,
        family.id_key: getattr(requests[3], family.id_key),
        "error_class": "TransportError",
        "message": "refused 3",
        "discarded": 0,
    }
    assert max(port.called) <= 4


@pytest.mark.parametrize("family", FAMILIES, ids=_IDS)
def test_first_failure_in_slice_order_wins(family: _Family) -> None:
    # Index 2 waits until index 3 has failed, then fails; the record names
    # index 2, as a one-at-a-time run would.
    requests = family.build(4)
    port = _IndexedPort(failing=frozenset({2, 3}))
    port.hold[2] = port.done(3)
    run = family.run(port, requests, "m", concurrency=4)
    assert port.finished.index(3) < port.finished.index(2)
    assert _ids(run.outcomes, family.id_key) == _ids(requests[:2], family.id_key)
    assert run.stopped is not None
    assert run.stopped["index"] == 2
    assert run.stopped["message"] == "refused 2"


@pytest.mark.parametrize("family", FAMILIES, ids=_IDS)
@pytest.mark.parametrize("concurrency", [0, -1])
def test_concurrency_below_one_is_refused(family: _Family, concurrency: int) -> None:
    port = _IndexedPort()
    with pytest.raises(ValueError, match="concurrency"):
        family.run(port, family.build(1), "m", concurrency=concurrency)
    assert port.called == []


@pytest.mark.parametrize("family", FAMILIES, ids=_IDS)
def test_other_errors_still_propagate(family: _Family) -> None:
    class _Boom(RuntimeError):
        pass

    class _BoomPort(_IndexedPort):
        def judge(self, *args: Any, **kwargs: Any) -> JudgmentResponse:
            raise _Boom

    with pytest.raises(_Boom):
        family.run(_BoomPort(), family.build(3), "m", concurrency=2)


@pytest.mark.parametrize(
    ("raw", "expected"), [(None, 1), ("", 1), (" ", 1), ("1", 1), (" 16 ", 16)]
)
def test_image_concurrency_reads_the_variable(raw: str | None, expected: int) -> None:
    environ = {} if raw is None else {IMAGE_CONCURRENCY_ENV: raw}
    assert image_concurrency(environ) == expected


@pytest.mark.parametrize("raw", ["0", "-4", "four", "2.5"])
def test_image_concurrency_refuses_bad_values(raw: str) -> None:
    with pytest.raises(ValueError, match=IMAGE_CONCURRENCY_ENV):
        image_concurrency({IMAGE_CONCURRENCY_ENV: raw})


# --- throughput receipt fields (#335) -------------------------------------

# A permutation of 1..20 seconds: nearest rank gives p50 10, p95 19, p99 20.
_LATENCIES = [float(7 * i % 20 + 1) for i in range(20)]
_BEFORE_METRICS = """\
# TYPE vllm:prefix_cache_hits counter
vllm:prefix_cache_hits_total{engine="0",model_name="m"} 100.0
vllm:prefix_cache_queries_total{engine="0",model_name="m"} 400.0
vllm:prompt_tokens_total{engine="0",model_name="m"} 1000.0
vllm:generation_tokens_total{engine="0",model_name="m"} 10.0
vllm:request_success_total{engine="0",finished_reason="stop",model_name="m"} 5.0
vllm:request_success_total{engine="0",finished_reason="length",model_name="m"} 1.0
vllm:kv_cache_usage_perc{engine="0",model_name="m"} 0.0
"""
_AFTER_METRICS = """\
vllm:prefix_cache_hits_total{engine="0",model_name="m"} 1600.0
vllm:prefix_cache_queries_total{engine="0",model_name="m"} 2400.0
vllm:prompt_tokens_total{engine="0",model_name="m"} 3000.0
vllm:generation_tokens_total{engine="0",model_name="m"} 22.0
vllm:request_success_total{engine="0",finished_reason="stop",model_name="m"} 15.0
vllm:request_success_total{engine="0",finished_reason="length",model_name="m"} 3.0
vllm:kv_cache_usage_perc{engine="0",model_name="m"} 0.25
"""


def _scripted_clock(latencies: Sequence[float]) -> Callable[[], float]:
    """Clock for a one-at-a-time run: run start, then start and end per call."""
    ticks = [0.0]
    for latency in latencies:
        ticks += [ticks[-1], ticks[-1] + latency]
    ticks.append(ticks[-1])
    return iter(ticks).__next__


def _receipt(family: _Family, run: Any) -> dict[str, Any]:
    return _RECEIPTS[family.name](run, backend="vllm", model="m", pins={}, identity={})


@pytest.mark.parametrize("family", FAMILIES, ids=_IDS)
def test_receipt_states_throughput_with_hand_checked_percentiles(
    family: _Family,
) -> None:
    requests = family.build(20)
    clock = _scripted_clock(_LATENCIES)
    run = family.run(_IndexedPort(), requests, "m", clock=clock)
    receipt = _receipt(family, run)

    per = _IMAGES_PER_JUDGMENT[family.name]
    assert receipt["wall_seconds"] == 210.0
    assert receipt["throughput"] == {
        "concurrency": 1,
        "wall_seconds": 210.0,
        "judgments": 20,
        "images": 20 * per,
        "judgments_per_second": pytest.approx(20 / 210),
        "images_per_second": pytest.approx(20 * per / 210),
        "latency_seconds": {"n": 20, "p50": 10.0, "p95": 19.0, "p99": 20.0},
        "discarded": 0,
        "server": None,
    }


@pytest.mark.parametrize("family", FAMILIES, ids=_IDS)
def test_images_per_second_counts_the_images_of_each_set(family: _Family) -> None:
    run = family.run(
        _IndexedPort(), family.build(3), "m", clock=_scripted_clock([1.0] * 3)
    )
    block = _receipt(family, run)["throughput"]
    per = _IMAGES_PER_JUDGMENT[family.name]
    assert block["images"] == 3 * per
    assert block["images_per_second"] == pytest.approx(per)
    assert block["judgments_per_second"] == pytest.approx(1.0)


@pytest.mark.parametrize("family", FAMILIES, ids=_IDS)
def test_zero_wall_time_and_empty_run_have_no_rates(family: _Family) -> None:
    run = family.run(_IndexedPort(), [], "m", clock=lambda: 5.0)
    block = _receipt(family, run)["throughput"]
    assert block["judgments"] == 0
    assert block["judgments_per_second"] is None
    assert block["images_per_second"] is None
    assert block["latency_seconds"] == {"n": 0, "p50": None, "p95": None, "p99": None}


@pytest.mark.parametrize("family", FAMILIES, ids=_IDS)
def test_discarded_counts_completions_and_failures_after_the_first_failure(
    family: _Family, failure_seen: threading.Event
) -> None:
    # Four in flight at once: index 1 fails at once. Indices 0, 2 and 3 are
    # held until the loop has that failure in hand; then 3 fails and 0 and 2
    # finish. Index 2 and index 3 reached the server but are dropped by
    # design, so two are discarded.
    requests = family.build(8)
    port = _IndexedPort(
        hold=dict.fromkeys((0, 2, 3), failure_seen), failing=frozenset({1, 3})
    )
    run = family.run(port, requests, "m", concurrency=4)
    receipt = _receipt(family, run)

    assert sorted(port.called) == [0, 1, 2, 3]
    assert _ids(run.outcomes, family.id_key) == _ids(requests[:1], family.id_key)
    assert run.stopped is not None
    assert run.stopped["index"] == 1
    assert run.stopped["discarded"] == 2
    assert receipt["stopped"]["discarded"] == 2
    assert receipt["throughput"]["discarded"] == 2
    assert receipt["throughput"]["concurrency"] == 4
    assert receipt["throughput"]["judgments"] == 1


@pytest.mark.parametrize("family", FAMILIES, ids=_IDS)
def test_one_at_a_time_failure_discards_nothing(family: _Family) -> None:
    port = _IndexedPort(failing=frozenset({1}))
    run = family.run(port, family.build(4), "m")
    assert run.stopped is not None
    assert run.stopped["discarded"] == 0
    assert port.called == [0, 1]


@pytest.mark.parametrize("family", FAMILIES, ids=_IDS)
def test_vllm_metrics_deltas_wrap_the_run(family: _Family) -> None:
    readings = iter([_BEFORE_METRICS, _AFTER_METRICS])
    reads: list[int] = []

    def read() -> str:
        reads.append(1)
        return next(readings)

    run = family.run(_IndexedPort(), family.build(2), "m", server_metrics=read)
    server = _receipt(family, run)["throughput"]["server"]

    assert len(reads) == 2
    assert server["counters"]["prefix_cache_hits"] == 1500.0
    assert server["counters"]["prefix_cache_queries"] == 2000.0
    assert server["counters"]["prompt_tokens"] == 2000.0
    assert server["counters"]["generation_tokens"] == 12.0
    assert server["counters"]["request_success"] == 12.0
    assert server["prefix_cache_hit_rate"] == pytest.approx(0.75)
    assert server["gauges"]["kv_cache_usage_perc"] == {"before": 0.0, "after": 0.25}
    assert server["e2e"] == "unknown"


@pytest.mark.parametrize("family", FAMILIES, ids=_IDS)
def test_failed_metrics_read_is_unknown(family: _Family) -> None:
    run = family.run(_IndexedPort(), family.build(1), "m", server_metrics=lambda: None)
    server = _receipt(family, run)["throughput"]["server"]
    assert server["prefix_cache_hit_rate"] == "unknown"
    assert set(server["counters"].values()) == {"unknown"}
    assert set(server["gauges"].values()) == {"unknown"}

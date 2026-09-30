"""Offline tests for the ``concurrency`` option of the image runners (#334).

Each test runs the face, check and signature runners through one fake
judgment port. The port finds the request index from the image bytes, so it
can hold calls at a barrier, finish them in reverse order or fail one index.
No test reads the network or a real image.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping, Sequence
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
    build_check_match_request,
    check_cases,
    run_check_match,
)
from typevet_evals.datasets.cedar import CedarPair, CedarSignature, PairKind
from typevet_evals.datasets.lfw import LfwFace, LfwPair
from typevet_evals.face_match import (
    IMAGE_CONCURRENCY_ENV,
    build_face_match_request,
    image_concurrency,
    run_face_match,
)
from typevet_evals.signature_match import (
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


class _IndexedPort:
    """Judgment port that acts on the request index found in the image."""

    def __init__(
        self,
        *,
        barrier: threading.Barrier | None = None,
        delay: Callable[[int], float] = lambda _: 0.0,
        failing: frozenset[int] = frozenset(),
    ) -> None:
        self._barrier = barrier
        self._delay = delay
        self._failing = failing
        self._lock = threading.Lock()
        self.called: list[int] = []

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
        time.sleep(self._delay(index))
        if index in self._failing:
            msg = f"refused {index}"
            raise TransportError(msg)
        return JudgmentResponse(
            model=model, usage=TokenUsage(input_tokens=10), answers=_answers(questions)
        )


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
    requests = family.build(6)
    port = _IndexedPort(delay=lambda i: 0.02 * (6 - i))
    run = family.run(port, requests, "m", concurrency=6)
    assert run.stopped is None
    assert _ids(run.outcomes, family.id_key) == _ids(requests, family.id_key)
    latencies = [o.latency_seconds for o in run.outcomes]
    assert latencies == sorted(latencies, reverse=True)


@pytest.mark.parametrize("family", FAMILIES, ids=_IDS)
def test_failure_stops_new_submissions_and_is_recorded(family: _Family) -> None:
    # Index 3 fails at once while the others take 50 ms. With two in flight,
    # no index after 4 is ever sent.
    requests = family.build(10)
    port = _IndexedPort(delay=lambda i: 0.0 if i == 3 else 0.05, failing=frozenset({3}))
    run = family.run(port, requests, "m", concurrency=2)
    assert _ids(run.outcomes, family.id_key) == _ids(requests[:3], family.id_key)
    assert run.stopped == {
        "index": 3,
        family.id_key: getattr(requests[3], family.id_key),
        "error_class": "TransportError",
        "message": "refused 3",
    }
    assert max(port.called) <= 4


@pytest.mark.parametrize("family", FAMILIES, ids=_IDS)
def test_first_failure_in_slice_order_wins(family: _Family) -> None:
    # Index 3 fails after index 2 fails; the record names index 2, as a
    # one-at-a-time run would.
    requests = family.build(4)
    port = _IndexedPort(
        delay=lambda i: 0.05 if i == 2 else 0.0, failing=frozenset({2, 3})
    )
    run = family.run(port, requests, "m", concurrency=4)
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

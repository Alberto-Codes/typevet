"""Unit tests: refresh the llama.cpp media marker after a model reload ([#322][i322]).

The llama.cpp router gives each model load a new random media marker. A
stub router behind ``httpx.MockTransport`` rejects a ``/completion`` prompt
that holds a stale marker with the HTTP 400 body that llama.cpp returns
("Failed to tokenize prompt"). The scoring adapter then reads ``/props``
once more, rebuilds the prompt and sends it once more. No request leaves
the host. The edge cases from [#323][i323] cover several images, a swap to
a text model and back, a failed refresh probe, an early close on the retry
and concurrent stale requests.

Examples:
    ```bash
    uv run pytest -q tests/unit/test_llama_cpp_media_marker_refresh.py
    ```

See Also:
    - [typevet.adapters.outbound.llama_cpp.scoring][]: Scoring adapter
    - [typevet.adapters.outbound.llama_cpp.multimodal][]: Media probe

[i322]: https://github.com/Alberto-Codes/typevet/issues/322
[i323]: https://github.com/Alberto-Codes/typevet/issues/323
"""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import httpx
import pytest

from typevet.adapters.outbound.llama_cpp.multimodal import MediaCapability
from typevet.adapters.outbound.llama_cpp.scoring import (
    LlamaCppCandidateScoringAdapter,
)
from typevet.domain.candidate_scoring_request import (
    CandidateScoringRequest,
    CandidateTokenSpec,
)
from typevet.domain.errors import (
    BackendHttpError,
    ScoringUnsupportedCapabilityError,
    TransportError,
)
from typevet.domain.media import MEDIA_MARKER, ImageInput

pytestmark = pytest.mark.unit

_MODEL = "gemma-mm"
# A caller vocabulary size skips the ``/v1/models`` read (#321), so the stub
# sees only the ``/props`` and ``/completion`` requests under test.
_N_VOCAB = 262144
_OLD = "<__media_old__>"
_NEW = "<__media_new__>"
_PNG = ImageInput(data=b"\x89PNG\r\n\x1a\nstub", mime_type="image/png")
_TOKENIZE_ERROR = {
    "error": {
        "code": 400,
        "message": "Failed to tokenize prompt",
        "type": "invalid_request_error",
    }
}
_COMPLETION = {
    "completion_probabilities": [
        {"top_logprobs": [{"id": 101, "logprob": -0.5}, {"id": 202, "logprob": -1.2}]}
    ]
}


class _ReloadingRouter:
    """Stub router whose marker is the one of the current model load.

    Attributes:
        marker (str): Marker ``/props`` reports and ``/completion`` accepts.
        reject_all (bool): Reject every media prompt, whatever the marker.
        vision (bool): Whether the loaded model declares image input.
        props_status (int): HTTP status of every ``/props`` answer.
        close_sends (set[int]): Zero-based ``/completion`` sends that end in
            an early close before a response head.
        stale_barrier (threading.Barrier | None): Holds each stale-marker
            send until every concurrent caller has sent one.
        props_calls (int): ``/props`` requests the stub answered.
        completion_bodies (list[dict[str, Any]]): Every ``/completion`` body.
        sends (int): Every ``/completion`` send, early closes included.
        paths (list[str]): Every request path, in arrival order.
    """

    def __init__(self, marker: str, *, reject_all: bool = False) -> None:
        self.marker = marker
        self.reject_all = reject_all
        self.vision = True
        self.props_status = 200
        self.close_sends: set[int] = set()
        self.stale_barrier: threading.Barrier | None = None
        self.props_calls = 0
        self.completion_bodies: list[dict[str, Any]] = []
        self.sends = 0
        self.paths: list[str] = []
        self._lock = threading.Lock()

    def handle(self, request: httpx.Request) -> httpx.Response:
        with self._lock:
            self.paths.append(request.url.path)
        if request.url.path.endswith("/props"):
            with self._lock:
                self.props_calls += 1
            if self.props_status != 200:
                return httpx.Response(self.props_status, text="loading model")
            props: dict[str, Any] = {"modalities": {"vision": self.vision}}
            if self.vision:
                props["media_marker"] = self.marker
            return httpx.Response(200, json=props)
        with self._lock:
            send = self.sends
            self.sends += 1
        if send in self.close_sends:
            msg = "Server disconnected without sending a response."
            raise httpx.RemoteProtocolError(msg, request=request)
        body = json.loads(request.content.decode())
        with self._lock:
            self.completion_bodies.append(body)
        prompt = body["prompt"]
        if isinstance(prompt, str) or not self.vision:
            return httpx.Response(400, json=_TOKENIZE_ERROR)
        stale = self.marker not in prompt["prompt_string"]
        if stale and self.stale_barrier is not None:
            self.stale_barrier.wait(timeout=10)
        if self.reject_all or stale:
            return httpx.Response(400, json=_TOKENIZE_ERROR)
        return httpx.Response(200, json=_COMPLETION)

    def adapter(self, marker: str) -> LlamaCppCandidateScoringAdapter:
        client = httpx.Client(transport=httpx.MockTransport(self.handle))
        return LlamaCppCandidateScoringAdapter(
            "http://offline-router",
            client=client,
            n_vocab=_N_VOCAB,
            media_capabilities={_MODEL: MediaCapability(vision=True, marker=marker)},
        )


def _request(*, media: tuple[ImageInput, ...] = (_PNG,)) -> CandidateScoringRequest:
    return CandidateScoringRequest(
        model=_MODEL,
        prefix=f"{MEDIA_MARKER * len(media)}Answer:",
        candidates=(
            CandidateTokenSpec("yes", (101,)),
            CandidateTokenSpec("no", (202,)),
        ),
        media=media,
    )


def test_stale_marker_refreshes_once_and_retries_once() -> None:
    """A reload between calls costs one ``/props`` read and one retry."""
    router = _ReloadingRouter(_OLD)
    adapter = router.adapter(_OLD)
    adapter.score_candidates(_request())
    router.marker = _NEW
    result = adapter.score_candidates(_request())
    assert [c.logprob for c in result.candidates] == [-0.5, -1.2]
    assert router.props_calls == 1
    prompts = [b["prompt"]["prompt_string"] for b in router.completion_bodies]
    assert prompts == [f"{_OLD}Answer:", f"{_OLD}Answer:", f"{_NEW}Answer:"]
    adapter.score_candidates(_request())
    assert router.props_calls == 1
    assert len(router.completion_bodies) == 4


def test_success_path_sends_no_extra_request() -> None:
    """A valid marker costs one ``/completion`` and no ``/props`` read."""
    router = _ReloadingRouter(_OLD)
    router.adapter(_OLD).score_candidates(_request())
    assert router.props_calls == 0
    assert len(router.completion_bodies) == 1


def test_persistent_tokenize_error_raises_after_one_refresh() -> None:
    """A second tokenize failure raises the usual ``BackendHttpError``."""
    router = _ReloadingRouter(_NEW, reject_all=True)
    with pytest.raises(BackendHttpError) as info:
        router.adapter(_OLD).score_candidates(_request())
    assert info.value.status_code == 400
    assert "Failed to tokenize prompt" in str(info.value)
    assert router.props_calls == 1
    assert len(router.completion_bodies) == 2


def test_text_request_tokenize_error_does_not_refresh() -> None:
    """A text-only request keeps the one-send error path."""
    router = _ReloadingRouter(_OLD)
    with pytest.raises(BackendHttpError):
        router.adapter(_OLD).score_candidates(_request(media=()))
    assert router.props_calls == 0
    assert len(router.completion_bodies) == 1


@pytest.mark.parametrize(
    ("status", "text"),
    [(400, "Failed to load image"), (500, "Failed to tokenize prompt")],
)
def test_other_errors_do_not_refresh(status: int, text: str) -> None:
    """Another 400 message, or the message with another status, is not retried."""
    props_calls = 0
    sends = 0

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal props_calls, sends
        if request.url.path.endswith("/props"):
            props_calls += 1
            return httpx.Response(200, json={"modalities": {"vision": True}})
        sends += 1
        return httpx.Response(status, json={"error": {"message": text}})

    adapter = LlamaCppCandidateScoringAdapter(
        "http://offline-router",
        client=httpx.Client(transport=httpx.MockTransport(handle)),
        n_vocab=_N_VOCAB,
        media_capabilities={_MODEL: MediaCapability(vision=True, marker=_OLD)},
    )
    with pytest.raises(BackendHttpError) as info:
        adapter.score_candidates(_request())
    assert info.value.status_code == status
    assert (props_calls, sends) == (0, 1)


def test_several_images_replace_every_stale_marker() -> None:
    """The retry carries the new marker once per image and every image."""
    router = _ReloadingRouter(_NEW)
    media = (_PNG, _PNG, _PNG)
    router.adapter(_OLD).score_candidates(_request(media=media))
    retried = router.completion_bodies[-1]["prompt"]
    assert retried["prompt_string"] == f"{_NEW * 3}Answer:"
    assert len(retried["multimodal_data"]) == 3
    assert (router.props_calls, len(router.completion_bodies)) == (1, 2)


def test_swap_to_text_model_then_back_probes_again() -> None:
    """A no-vision refresh is not cached; the next image request probes again."""
    router = _ReloadingRouter(_NEW)
    router.vision = False
    adapter = router.adapter(_OLD)
    with pytest.raises(ScoringUnsupportedCapabilityError, match="image input"):
        adapter.score_candidates(_request())
    assert (router.props_calls, len(router.completion_bodies)) == (1, 1)
    router.vision = True
    del router.paths[:]
    result = adapter.score_candidates(_request())
    assert router.paths == ["/props", "/completion"]
    assert [c.logprob for c in result.candidates] == [-0.5, -1.2]
    assert router.props_calls == 2
    retried = router.completion_bodies[-1]["prompt"]["prompt_string"]
    assert retried == f"{_NEW}Answer:"
    adapter.score_candidates(_request())
    assert router.props_calls == 2


def test_failed_refresh_probe_raises_the_original_tokenize_error() -> None:
    """A ``/props`` failure keeps the tokenize 400 and chains the probe error."""
    router = _ReloadingRouter(_NEW)
    router.props_status = 503
    with pytest.raises(BackendHttpError) as info:
        router.adapter(_OLD).score_candidates(_request())
    assert info.value.status_code == 400
    assert "Failed to tokenize prompt" in str(info.value)
    cause = info.value.__cause__
    assert isinstance(cause, BackendHttpError)
    assert cause.status_code == 503
    assert (router.props_calls, len(router.completion_bodies)) == (1, 1)


def test_refresh_probe_transport_failure_is_chained() -> None:
    """A transport failure on the refresh probe is the cause of the 400."""
    router = _ReloadingRouter(_NEW)

    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/props"):
            msg = "connection refused"
            raise httpx.ConnectError(msg, request=request)
        return router.handle(request)

    adapter = LlamaCppCandidateScoringAdapter(
        "http://offline-router",
        client=httpx.Client(transport=httpx.MockTransport(handle)),
        n_vocab=_N_VOCAB,
        media_capabilities={_MODEL: MediaCapability(vision=True, marker=_OLD)},
    )
    with pytest.raises(BackendHttpError) as info:
        adapter.score_candidates(_request())
    assert info.value.status_code == 400
    assert isinstance(info.value.__cause__, TransportError)


def test_early_close_on_the_retry_is_sent_once_more() -> None:
    """The retried send keeps the one early-close retry and does not loop."""
    router = _ReloadingRouter(_NEW)
    router.close_sends = {1}
    result = router.adapter(_OLD).score_candidates(_request())
    assert [c.logprob for c in result.candidates] == [-0.5, -1.2]
    assert (router.props_calls, router.sends) == (1, 3)


def test_second_early_close_on_the_retry_raises_transport_error() -> None:
    """Two early closes on the retried send raise and send nothing more."""
    router = _ReloadingRouter(_NEW)
    router.close_sends = {1, 2}
    with pytest.raises(TransportError):
        router.adapter(_OLD).score_candidates(_request())
    assert (router.props_calls, router.sends) == (1, 3)


def test_concurrent_stale_requests_all_recover() -> None:
    """Concurrent stale requests succeed with at most one probe each."""
    callers = 4
    router = _ReloadingRouter(_NEW)
    router.stale_barrier = threading.Barrier(callers)
    adapter = router.adapter(_OLD)
    with ThreadPoolExecutor(max_workers=callers) as pool:
        results = list(
            pool.map(lambda _: adapter.score_candidates(_request()), range(callers))
        )
    assert all([c.logprob for c in r.candidates] == [-0.5, -1.2] for r in results)
    assert 1 <= router.props_calls <= callers
    assert len(router.completion_bodies) == 2 * callers
    router.stale_barrier = None
    adapter.score_candidates(_request())
    assert router.props_calls <= callers
    assert len(router.completion_bodies) == 2 * callers + 1

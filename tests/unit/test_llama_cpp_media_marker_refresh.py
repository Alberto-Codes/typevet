"""Unit tests: refresh the llama.cpp media marker after a model reload ([#322][i322]).

The llama.cpp router gives each model load a new random media marker. A
stub router behind ``httpx.MockTransport`` rejects a ``/completion`` prompt
that holds a stale marker with the HTTP 400 body that llama.cpp returns
("Failed to tokenize prompt"). The scoring adapter then reads ``/props``
once more, rebuilds the prompt and sends it once more. No request leaves
the host.

Examples:
    ```bash
    uv run pytest -q tests/unit/test_llama_cpp_media_marker_refresh.py
    ```

See Also:
    - [typevet.adapters.outbound.llama_cpp.scoring][]: Scoring adapter
    - [typevet.adapters.outbound.llama_cpp.multimodal][]: Media probe

[i322]: https://github.com/Alberto-Codes/typevet/issues/322
"""

from __future__ import annotations

import json
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
from typevet.domain.errors import BackendHttpError
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
        props_calls (int): ``/props`` requests the stub answered.
        completion_bodies (list[dict[str, Any]]): Every ``/completion`` body.
    """

    def __init__(self, marker: str, *, reject_all: bool = False) -> None:
        self.marker = marker
        self.reject_all = reject_all
        self.props_calls = 0
        self.completion_bodies: list[dict[str, Any]] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/props"):
            self.props_calls += 1
            return httpx.Response(
                200,
                json={"modalities": {"vision": True}, "media_marker": self.marker},
            )
        body = json.loads(request.content.decode())
        self.completion_bodies.append(body)
        prompt = body["prompt"]
        if isinstance(prompt, str):
            return httpx.Response(400, json=_TOKENIZE_ERROR)
        if self.reject_all or self.marker not in prompt["prompt_string"]:
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

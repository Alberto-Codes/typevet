"""Contract tests for vLLM scoring error mapping and fail-closed rules (#107C).

Fixtures are redacted vLLM v0.30.0 HTTP 400 responses from the #168 probe:
``http400_out_of_vocab`` (P6), ``http400_over_128`` (P7) and
``http400_logprobs_false`` (P9). The duplicate-entry case mutates P4
(``image_three_way``).
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from tests.fixtures.synthetic_images import solid_png
from typevet.adapters import outbound
from typevet.adapters.outbound.vllm.scoring import (
    ChatContentFraming,
    VllmCandidateScoringAdapter,
)
from typevet.domain.candidate_scoring_request import (
    CandidateScoringRequest,
    CandidateTokenSpec,
)
from typevet.domain.errors import (
    BackendHttpError,
    GenerationError,
    ScoringUnsupportedCapabilityError,
    ScoringValidationError,
    TransportError,
)
from typevet.domain.media import MEDIA_MARKER, ImageInput

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "vllm"
_MODEL = "gemma-4-31b-it"
_P4_IDS = (236771, 236770, 236778)
_HEADER_VALUE = "Bearer do-not-echo-this-value"


def _fixture(name: str) -> dict[str, Any]:
    return json.loads((_FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


def _request(
    ids: tuple[int, ...] = _P4_IDS,
    *,
    prefix: str = "Answer:",
    media: tuple[ImageInput, ...] = (),
) -> CandidateScoringRequest:
    return CandidateScoringRequest(
        model=_MODEL,
        prefix=prefix,
        candidates=tuple(CandidateTokenSpec(f"id{i}", (i,)) for i in ids),
        media=media,
    )


class _Recorder:
    """MockTransport handler that records bodies and replies with one response."""

    def __init__(self, *, status: int = 200, payload: object = None, text: str = ""):
        self.bodies: list[dict[str, Any]] = []
        self._status = status
        self._payload = payload
        self._text = text

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.bodies.append(json.loads(request.content.decode()))
        if self._payload is None:
            return httpx.Response(self._status, text=self._text)
        return httpx.Response(self._status, json=self._payload)


def _adapter(handler: Any) -> VllmCandidateScoringAdapter:
    client = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="http://test",
        headers={"Authorization": _HEADER_VALUE},
    )
    return VllmCandidateScoringAdapter(base_url="http://test", client=client)


def _p4_payload() -> dict[str, Any]:
    return _fixture("image_three_way")["response"]


def _top(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return payload["choices"][0]["logprobs"]["content"][0]["top_logprobs"]


@pytest.mark.contract
@pytest.mark.parametrize(
    "name",
    ["http400_out_of_vocab", "http400_over_128", "http400_logprobs_false"],
    ids=["P6", "P7", "P9"],
)
def test_vllm_scoring_http_error_status_raises_backend_http_error(name: str) -> None:
    probe = _fixture(name)
    recorder = _Recorder(status=probe["http_status"], payload=probe["response"])

    with pytest.raises(BackendHttpError) as caught:
        _adapter(recorder).score_candidates(_request())

    exc = caught.value
    assert exc.status_code == 400
    assert probe["response"]["error"]["message"] in exc.body_snippet
    assert str(exc).startswith("vLLM HTTP 400: ")
    assert len(recorder.bodies) == 1


@pytest.mark.contract
def test_vllm_scoring_http_error_snippet_is_bounded() -> None:
    recorder = _Recorder(status=502, text="<html>" + "x" * 2000)

    with pytest.raises(BackendHttpError) as caught:
        _adapter(recorder).score_candidates(_request())

    assert caught.value.status_code == 502
    assert len(caught.value.body_snippet) == 500


@pytest.mark.contract
def test_vllm_scoring_http_error_does_not_copy_request_headers() -> None:
    seen: list[str | None] = []

    def reject(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("Authorization"))
        return httpx.Response(401, text="bad key")

    with pytest.raises(BackendHttpError) as caught:
        _adapter(reject).score_candidates(_request())

    assert seen == [_HEADER_VALUE]
    assert caught.value.body_snippet == "bad key"
    assert _HEADER_VALUE not in str(caught.value)
    assert _HEADER_VALUE not in caught.value.body_snippet


@pytest.mark.contract
def test_vllm_scoring_transport_failure_raises_transport_error() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with pytest.raises(TransportError, match=r"^vLLM request failed: ") as caught:
        _adapter(refuse).score_candidates(_request())

    assert _HEADER_VALUE not in str(caught.value)


@pytest.mark.contract
def test_vllm_scoring_non_json_success_body_raises_generation_error() -> None:
    recorder = _Recorder(status=200, text="not json")
    with pytest.raises(GenerationError, match="vLLM") as caught:
        _adapter(recorder).score_candidates(_request())
    assert not isinstance(caught.value, BackendHttpError)


@pytest.mark.contract
def test_vllm_scoring_refuses_more_than_128_ids_before_post() -> None:
    recorder = _Recorder(payload=_p4_payload())
    ids = tuple(range(1000, 1129))
    assert len(ids) == 129

    with pytest.raises(ScoringUnsupportedCapabilityError, match="128"):
        _adapter(recorder).score_candidates(_request(ids))

    assert recorder.bodies == []


@pytest.mark.contract
def test_vllm_scoring_sends_exactly_128_ids() -> None:
    ids = tuple(range(1000, 1128))
    payload = _p4_payload()
    payload["choices"][0]["logprobs"]["content"][0]["top_logprobs"] = [
        {"token": f"token_id:{i}", "logprob": -1.0} for i in ids
    ]
    recorder = _Recorder(payload=payload)

    result = _adapter(recorder).score_candidates(_request(ids))

    assert recorder.bodies[0]["logprob_token_ids"] == list(ids)
    assert len(result.candidates) == 128


@pytest.mark.contract
def test_vllm_scoring_rejects_duplicate_token_id_entry() -> None:
    payload = _p4_payload()
    entries = _top(payload)
    duplicate = dict(entries[0])
    duplicate["logprob"] = -0.001
    entries.append(duplicate)
    recorder = _Recorder(payload=payload)

    with pytest.raises(ScoringValidationError, match="duplicate"):
        _adapter(recorder).score_candidates(_request())


@pytest.mark.contract
@pytest.mark.parametrize(
    "value", [True, False, "-0.5", None], ids=["true", "false", "string", "null"]
)
def test_vllm_scoring_rejects_non_number_logprob(value: object) -> None:
    payload = _p4_payload()
    _top(payload)[0]["logprob"] = value
    recorder = _Recorder(payload=payload)

    with pytest.raises(GenerationError, match="logprob"):
        _adapter(recorder).score_candidates(_request())


@pytest.mark.contract
def test_vllm_scoring_empty_candidates_fail_in_domain_without_post() -> None:
    """Characterization: green when written; the domain already refuses this."""
    recorder = _Recorder(payload=_p4_payload())
    adapter = _adapter(recorder)

    with pytest.raises(ScoringValidationError):
        adapter.score_candidates(_request(()))

    assert recorder.bodies == []


@pytest.mark.contract
def test_vllm_media_data_uri_takes_mime_type_from_image_input() -> None:
    data = solid_png((10, 20, 30))
    image = ImageInput(data=data, mime_type="image/jpeg")
    recorder = _Recorder(payload=_p4_payload())

    _adapter(recorder).score_candidates(
        _request(prefix=f"{MEDIA_MARKER}\nWhich colour?", media=(image,))
    )

    blocks = recorder.bodies[0]["messages"][0]["content"]
    encoded = base64.b64encode(data).decode("ascii")
    assert blocks[0] == {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
    }


@pytest.mark.contract
def test_chat_content_framing_is_exported_from_outbound_package() -> None:
    assert getattr(outbound, "ChatContentFraming", None) is ChatContentFraming
    assert "ChatContentFraming" in outbound.__all__

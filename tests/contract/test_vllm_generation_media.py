"""Contract tests for image-conditioned typed generation (#169B).

The ``generation_enum_image`` fixture is the redacted vLLM v0.30.0 P11
response from the #168 probe: an enum schema with one image returned
``{"verdict": "0"}``. The llama.cpp generation adapters must refuse media
before any POST, so an image is never dropped silently.
"""

from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from typevet import domain
from typevet.adapters.inbound import api
from typevet.adapters.outbound.async_llama_cpp import AsyncLlamaCppGenerationAdapter
from typevet.adapters.outbound.llama_cpp import LlamaCppGenerationAdapter
from typevet.adapters.outbound.vllm_generation import VllmGenerationAdapter
from typevet.domain import errors as domain_errors
from typevet.domain.media import MEDIA_MARKER, ImageInput
from typevet.domain.models import GenerationRequest

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "vllm"
_PNG = ImageInput(data=b"\x89PNG-first", mime_type="image/png")
_JPEG = ImageInput(data=b"\xff\xd8JPEG-second", mime_type="image/jpeg")
_SCHEMA = {
    "type": "object",
    "properties": {"n": {"type": "integer"}},
    "required": ["n"],
    "additionalProperties": False,
}


def _probe(name: str) -> dict[str, Any]:
    return json.loads((_FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


def _data_uri(image: ImageInput) -> str:
    encoded = base64.b64encode(image.data).decode("ascii")
    return f"data:{image.mime_type};base64,{encoded}"


class _Recorder:
    """MockTransport handler that records each body and replies with a payload."""

    def __init__(self, payload: object) -> None:
        self.bodies: list[dict[str, Any]] = []
        self._payload = payload

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.bodies.append(json.loads(request.content.decode()))
        return httpx.Response(200, json=self._payload)


def _vllm(recorder: _Recorder) -> VllmGenerationAdapter:
    client = httpx.Client(transport=httpx.MockTransport(recorder))
    return VllmGenerationAdapter("http://vllm.test:8000/", client=client)


def _reply(content: str) -> dict[str, Any]:
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


@pytest.mark.contract
def test_p11_image_generation_returns_the_probe_verdict() -> None:
    probe = _probe("generation_enum_image")
    sent = probe["request"]
    text = sent["messages"][0]["content"][1]["text"]
    recorder = _Recorder(probe["response"])
    result = api.generate(
        _vllm(recorder),
        prompt=f"{MEDIA_MARKER}\n{text}",
        schema=sent["structured_outputs"]["json"],
        model=sent["model"],
        media=(_PNG,),
    )
    assert result.value == {"verdict": "0"}
    assert len(recorder.bodies) == 1
    body = recorder.bodies[0]
    assert body["structured_outputs"] == sent["structured_outputs"]
    assert body["messages"] == [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": _data_uri(_PNG)}},
                {"type": "text", "text": text},
            ],
        }
    ]


@pytest.mark.contract
def test_two_images_keep_marker_order() -> None:
    recorder = _Recorder(_reply('{"n": 2}'))
    api.generate(
        _vllm(recorder),
        prompt=f"First {MEDIA_MARKER}\nthen {MEDIA_MARKER}\ncount them.",
        schema=_SCHEMA,
        model="served-model",
        media=(_PNG, _JPEG),
    )
    assert recorder.bodies[0]["messages"][0]["content"] == [
        {"type": "text", "text": "First "},
        {"type": "image_url", "image_url": {"url": _data_uri(_PNG)}},
        {"type": "text", "text": "then "},
        {"type": "image_url", "image_url": {"url": _data_uri(_JPEG)}},
        {"type": "text", "text": "count them."},
    ]


@pytest.mark.contract
def test_text_only_request_keeps_string_content() -> None:
    recorder = _Recorder(_reply('{"n": 1}'))
    api.generate(
        _vllm(recorder), prompt="one", schema=_SCHEMA, model="served-model", media=()
    )
    assert recorder.bodies[0]["messages"][0]["content"] == "one"


@pytest.mark.contract
@pytest.mark.parametrize(
    ("prompt", "media"),
    [
        ("no marker here", (_PNG,)),
        (f"{MEDIA_MARKER} one marker", ()),
        (f"{MEDIA_MARKER} {MEDIA_MARKER}", (_PNG,)),
    ],
)
def test_marker_mismatch_is_rejected_before_any_post(
    prompt: str, media: tuple[ImageInput, ...]
) -> None:
    recorder = _Recorder(_reply('{"n": 1}'))
    with pytest.raises(ValueError, match="media marker"):
        api.generate(
            _vllm(recorder),
            prompt=prompt,
            schema=_SCHEMA,
            model="served-model",
            media=media,
        )
    assert recorder.bodies == []


def _image_request() -> GenerationRequest:
    return GenerationRequest(
        prompt=f"{MEDIA_MARKER}\nCount.",
        schema=_SCHEMA,
        model="m",
        media=(_PNG,),
    )


@pytest.mark.contract
def test_sync_llama_cpp_refuses_media_without_a_post() -> None:
    request = _image_request()
    recorder = _Recorder(_reply('{"n": 1}'))
    client = httpx.Client(transport=httpx.MockTransport(recorder))
    adapter = LlamaCppGenerationAdapter(base_url="http://llama.test", client=client)
    with pytest.raises(domain_errors.GenerationUnsupportedCapabilityError):
        adapter.generate(request)
    assert recorder.bodies == []


@pytest.mark.contract
def test_async_llama_cpp_refuses_media_without_a_post() -> None:
    request = _image_request()
    recorder = _Recorder(_reply('{"n": 1}'))
    client = httpx.AsyncClient(transport=httpx.MockTransport(recorder))
    adapter = AsyncLlamaCppGenerationAdapter(
        base_url="http://llama.test", client=client
    )
    with pytest.raises(domain_errors.GenerationUnsupportedCapabilityError):
        asyncio.run(adapter.generate(request))
    assert recorder.bodies == []


@pytest.mark.contract
def test_unsupported_capability_error_is_an_exported_generation_error() -> None:
    error_type = domain_errors.GenerationUnsupportedCapabilityError
    assert issubclass(error_type, domain_errors.GenerationError)
    assert "GenerationUnsupportedCapabilityError" in domain.__all__
    assert domain.GenerationUnsupportedCapabilityError is error_type
    assert _image_request().media == (_PNG,)

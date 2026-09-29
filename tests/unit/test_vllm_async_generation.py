"""Unit tests for the async vLLM generation adapter (#169E).

Handlers gate on ``asyncio.Event`` so the tests observe how many POSTs are in
flight without a wall-clock sleep. ``asyncio.sleep(0)`` only yields to the
event loop.
"""

from __future__ import annotations

import asyncio
import base64
import json
from typing import Any

import httpx
import pytest

from typevet.adapters import outbound
from typevet.adapters.inbound.backend_settings import VllmSettings, load_vllm_settings
from typevet.adapters.outbound.vllm_generation_async import AsyncVllmGenerationAdapter
from typevet.domain.errors import GenerationError, TransportError
from typevet.domain.media import MEDIA_MARKER, ImageInput
from typevet.domain.models import GenerationRequest
from typevet.ports.async_generation import AsyncGenerationPort

_SCHEMA = {
    "type": "object",
    "properties": {"n": {"type": "integer"}},
    "required": ["n"],
    "additionalProperties": False,
}
_MODEL = "served-model"
_VAR = "TYPEVET_VLLM__MAX_CONCURRENCY"


def _reply(content: str = '{"n": 1}') -> httpx.Response:
    return httpx.Response(
        200, json={"choices": [{"message": {"role": "assistant", "content": content}}]}
    )


def _request(prompt: str = "count", media: tuple[ImageInput, ...] = ()) -> Any:
    return GenerationRequest(prompt=prompt, schema=_SCHEMA, model=_MODEL, media=media)


def _adapter(handler: Any, *, max_concurrency: int = 1) -> AsyncVllmGenerationAdapter:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return AsyncVllmGenerationAdapter(
        "http://vllm.test:8000/", client=client, max_concurrency=max_concurrency
    )


async def _yield_loop(times: int = 20) -> None:
    for _ in range(times):
        await asyncio.sleep(0)


class _Gate:
    """Async handler that holds every POST until ``release`` is set."""

    def __init__(self, *, limit_seen: int) -> None:
        self.in_flight = 0
        self.peak = 0
        self.started = 0
        self.limit_seen = limit_seen
        self.reached = asyncio.Event()
        self.release = asyncio.Event()

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        self.started += 1
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        if self.in_flight >= self.limit_seen:
            self.reached.set()
        try:
            await self.release.wait()
        finally:
            self.in_flight -= 1
        return _reply()


@pytest.mark.unit
def test_async_vllm_adapter_is_exported_and_satisfies_port() -> None:
    assert outbound.AsyncVllmGenerationAdapter is AsyncVllmGenerationAdapter
    assert "AsyncVllmGenerationAdapter" in outbound.__all__
    port: AsyncGenerationPort = AsyncVllmGenerationAdapter("http://x")
    assert callable(port.generate)


@pytest.mark.unit
@pytest.mark.parametrize("bad", [0, -1])
def test_async_vllm_adapter_rejects_limit_below_one(bad: int) -> None:
    with pytest.raises(ValueError, match="max_concurrency must be at least 1"):
        AsyncVllmGenerationAdapter("http://x", max_concurrency=bad)


@pytest.mark.unit
def test_five_gathered_calls_under_limit_two_peak_at_two_in_flight() -> None:
    async def run() -> None:
        gate = _Gate(limit_seen=2)
        adapter = _adapter(gate, max_concurrency=2)
        tasks = [asyncio.create_task(adapter.generate(_request())) for _ in range(5)]
        await gate.reached.wait()
        await _yield_loop()
        assert gate.in_flight == 2
        assert gate.started == 2
        gate.release.set()
        results = await asyncio.gather(*tasks)
        assert [result.value for result in results] == [{"n": 1}] * 5
        assert gate.started == 5
        assert gate.peak == 2

    asyncio.run(run())


@pytest.mark.unit
def test_queued_call_starts_after_in_flight_call_is_cancelled() -> None:
    async def run() -> None:
        gate = _Gate(limit_seen=1)
        adapter = _adapter(gate, max_concurrency=1)
        first = asyncio.create_task(adapter.generate(_request()))
        await gate.reached.wait()
        second = asyncio.create_task(adapter.generate(_request()))
        await _yield_loop()
        assert gate.started == 1
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        await _yield_loop()
        assert gate.started == 2
        assert gate.in_flight == 1
        gate.release.set()
        result = await second
        assert result.value == {"n": 1}

    asyncio.run(run())


@pytest.mark.unit
def test_timeout_raises_transport_error_and_frees_the_slot() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            raise httpx.ReadTimeout("slow", request=request)
        return _reply()

    async def run() -> None:
        adapter = _adapter(handler)
        with pytest.raises(TransportError, match="vLLM request failed") as caught:
            await adapter.generate(_request())
        assert isinstance(caught.value.__cause__, httpx.TimeoutException)
        assert not adapter._slots.locked()
        result = await adapter.generate(_request())
        assert result.value == {"n": 1}

    asyncio.run(run())


@pytest.mark.unit
def test_async_body_matches_sync_contract_and_sends_images() -> None:
    bodies: list[dict[str, Any]] = []
    image = ImageInput(data=b"\x89PNG-one", mime_type="image/png")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        bodies.append(json.loads(request.content.decode()))
        return _reply()

    async def run() -> None:
        adapter = _adapter(handler)
        await adapter.generate(_request())
        await adapter.generate(_request(f"{MEDIA_MARKER}\nlook", (image,)))

    asyncio.run(run())
    text_body, image_body = bodies
    assert text_body == {
        "model": _MODEL,
        "messages": [{"role": "user", "content": "count"}],
        "temperature": 0,
        "structured_outputs": {"json": _SCHEMA},
        "add_generation_prompt": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    blocks = image_body["messages"][0]["content"]
    encoded = base64.b64encode(image.data).decode("ascii")
    assert {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{encoded}"},
    } in blocks
    assert {key: val for key, val in image_body.items() if key != "messages"} == {
        key: val for key, val in text_body.items() if key != "messages"
    }


@pytest.mark.unit
def test_async_bad_content_raises_generation_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _reply("not json")

    with pytest.raises(GenerationError, match="not valid JSON"):
        asyncio.run(_adapter(handler).generate(_request()))


@pytest.mark.unit
def test_owned_client_closes_on_exit_and_injected_client_stays_open() -> None:
    async def run() -> None:
        async with AsyncVllmGenerationAdapter("http://vllm.test") as owned:
            client = owned._ensure_client()
            assert owned._ensure_client() is client
        assert client.is_closed
        injected = httpx.AsyncClient()
        async with AsyncVllmGenerationAdapter("http://vllm.test", client=injected):
            pass
        assert not injected.is_closed
        await injected.aclose()

    asyncio.run(run())


def _env(**extra: str) -> dict[str, str]:
    return {
        "TYPEVET_VLLM__BASE_URL": "http://vllm.test",
        "TYPEVET_VLLM__MODEL": "m",
        **extra,
    }


@pytest.mark.unit
def test_vllm_settings_max_concurrency_defaults_to_one_and_reads_env() -> None:
    assert load_vllm_settings(_env()).max_concurrency == 1
    assert load_vllm_settings(_env(**{_VAR: ""})).max_concurrency == 1
    assert load_vllm_settings(_env(**{_VAR: " 4 "})).max_concurrency == 4
    assert VllmSettings(base_url="u", model="m").max_concurrency == 1


@pytest.mark.unit
@pytest.mark.parametrize("raw", ["0", "-2", "1.5", "many9731"])
def test_vllm_settings_rejects_bad_max_concurrency_without_echoing(raw: str) -> None:
    with pytest.raises(
        ValueError, match=f"{_VAR} must be a positive integer"
    ) as caught:
        load_vllm_settings(_env(**{_VAR: raw}))
    assert raw not in str(caught.value)
    assert caught.value.__cause__ is None or raw not in str(caught.value.__cause__)

"""Unit tests: ``TYPEVET_LLAMA__AUTOLOAD=false`` never evicts another model (#461).

A stateful fake llama.cpp router follows the ``--models-max`` rules of
``tools/server/server-models.cpp``: a request that names a model the router is
not running, and that carries ``?autoload=false``, fails at once with HTTP 400
``model is not loaded``. Without the parameter, or with ``autoload=true``, the
router loads the model and evicts another one when the slots are full.

With the setting off, the adapters send ``autoload=false`` on every request the
router proxies to a model. After the 400 they read ``GET /models`` once. When
no other model holds a slot, or the router ``/props`` reports room for one
more, the request is sent once more with ``autoload=true``. Otherwise they
raise ``BackendBusyError``, which the judgevet bridge maps to
``ProviderBusyError``. With the default setting, no URL carries the parameter.

Examples:
    ```bash
    uv run pytest -q tests/unit/test_llama_cpp_autoload.py
    ```

See Also:
    - [typevet.adapters.inbound.settings][]: ``LlamaSettings.autoload``
    - [typevet.adapters.outbound.llama_cpp.generation][]: Sync adapter
    - [typevet.adapters.outbound.llama_cpp.generation_async][]: Async adapter
    - [typevet.adapters.outbound.llama_cpp.scoring][]: Scoring adapter
    - [typevet.adapters.inbound.judgevet][]: Bridge that maps the busy error
"""

from __future__ import annotations

import asyncio
import json
import math

import httpx
import pytest
from judgevet.domain.questions import Noul as JevNoul
from judgevet.providers import ProviderBusyError

from typevet.adapters.inbound.backend_settings import open_judgment
from typevet.adapters.inbound.judgevet import provider_factory
from typevet.adapters.inbound.settings import (
    LlamaSettings,
    llama_cpp_adapter,
    load_llama_settings,
)
from typevet.adapters.outbound.llama_cpp.generation import LlamaCppGenerationAdapter
from typevet.adapters.outbound.llama_cpp.generation_async import (
    AsyncLlamaCppGenerationAdapter,
)
from typevet.adapters.outbound.llama_cpp.scoring import (
    LlamaCppCandidateScoringAdapter,
)
from typevet.domain.candidate_scoring_request import (
    CandidateScoringRequest,
    CandidateTokenSpec,
)
from typevet.domain.errors import BackendBusyError, BackendHttpError
from typevet.domain.models import GenerationRequest

pytestmark = pytest.mark.unit

_JUDGE = "gemma-4-31b-kv9-text"
_OTHER = "coder-74b"
_RUNNING = frozenset({"loaded", "loading", "sleeping"})
_NOT_LOADED = {
    "error": {
        "code": 400,
        "message": "model is not loaded",
        "type": "invalid_request_error",
    }
}
_GEMMA4_RENDERED = "<|turn>user\nhello<turn|>\n<|turn>model\n"
_SCHEMA = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}},
    "required": ["ok"],
}
_N_VOCAB = 4
_ENVIRON = {
    "TYPEVET_BACKEND": "llama_cpp",
    "TYPEVET_LLAMA__BASE_URL": "http://test",
    "TYPEVET_LLAMA__MULTIMODAL_MODEL": _JUDGE,
    "TYPEVET_LLAMA__AUTOLOAD": "false",
}

CHAT = "/v1/chat/completions"
OFF = "autoload=false"
ON = "autoload=true"


class _Router:
    """Fake router with one model slot per ``max_instances`` (0 is unlimited).

    Attributes:
        status (dict[str, str]): ``status.value`` per model id.
        max_instances (int): Router ``--models-max``.
        requests (list[tuple[str, str]]): ``(path, query)`` in call order.
        evictions (int): Models the router unloaded to make room.

    Examples:
        ```python
        router = _Router({_OTHER: "loaded"})
        assert router.requests == []
        ```
    """

    def __init__(self, status: dict[str, str], *, max_instances: int = 1) -> None:
        self.status = {_JUDGE: "unloaded", _OTHER: "unloaded", **status}
        self.max_instances = max_instances
        self.requests: list[tuple[str, str]] = []
        self.evictions = 0
        self.refuse_load = False
        self.local: dict[str, httpx.Response] = {}

    def running(self) -> list[str]:
        """Return the model ids whose status counts as running."""
        return [m for m, s in self.status.items() if s in _RUNNING]

    def handle(self, request: httpx.Request) -> httpx.Response:
        """Serve the router-local routes, then validate and proxy the rest."""
        path = request.url.path
        params = dict(request.url.params)
        self.requests.append((path, request.url.query.decode()))
        if path in ("/models", "/v1/models"):
            data = [{"id": m, "status": {"value": s}} for m, s in self.status.items()]
            return self.local.get("/models", httpx.Response(200, json={"data": data}))
        if path == "/props" and "model" not in params:
            body = {"role": "router", "max_instances": self.max_instances}
            return self.local.get("/props", httpx.Response(200, json=body))
        model = params.get("model") or json.loads(request.content)["model"]
        if model not in self.status:
            message = f"model '{model}' not found"
            error = {"code": 400, "message": message, "type": "invalid_request_error"}
            return httpx.Response(400, json={"error": error})
        if self.status[model] not in _RUNNING:
            if self.refuse_load or params.get("autoload", "true") not in ("true", "1"):
                return httpx.Response(400, json=_NOT_LOADED)
            self._load(model)
        return self._proxied(path, request)

    def _load(self, model: str) -> None:
        running = self.running()
        if self.max_instances and len(running) >= self.max_instances:
            self.status[running[0]] = "unloaded"
            self.evictions += 1
        self.status[model] = "loaded"

    def _proxied(self, path: str, request: httpx.Request) -> httpx.Response:
        if path == "/props":
            return httpx.Response(200, json={"modalities": {"vision": False}})
        if path == "/apply-template":
            return httpx.Response(200, json={"prompt": _GEMMA4_RENDERED})
        if path == "/tokenize":
            content = json.loads(request.content)["content"]
            return httpx.Response(200, json={"tokens": [sum(map(ord, content)) % 4]})
        if path == "/completion":
            probs = (0.5, 0.3, 0.1, 0.1)
            top = [{"id": i, "logprob": math.log(p)} for i, p in enumerate(probs)]
            body = {"completion_probabilities": [{"top_logprobs": top}]}
            return httpx.Response(200, json=body)
        if path == CHAT:
            content = json.dumps({"ok": True})
            body = {"choices": [{"message": {"content": content}}]}
            return httpx.Response(200, json=body)
        return httpx.Response(404)

    def client(self) -> httpx.Client:
        return httpx.Client(
            transport=httpx.MockTransport(self.handle), base_url="http://test"
        )

    def async_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            transport=httpx.MockTransport(self.handle), base_url="http://test"
        )


def _generation() -> GenerationRequest:
    return GenerationRequest(prompt="Is it fine?", schema=_SCHEMA, model=_JUDGE)


def _scoring() -> CandidateScoringRequest:
    return CandidateScoringRequest(
        model=_JUDGE,
        prefix="Answer:",
        candidates=(CandidateTokenSpec("a", (0,)), CandidateTokenSpec("b", (1,))),
    )


def _after_first_400(router: _Router) -> list[tuple[str, str]]:
    """Return the requests after the first ``autoload=false`` request."""
    return router.requests[1:]


# --- settings -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("", True),
        ("true", True),
        ("1", True),
        ("false", False),
        ("0", False),
        (" False ", False),
    ],
)
def test_autoload_setting_reads_the_variable(raw: str, expected: bool) -> None:
    assert LlamaSettings().autoload is True
    assert load_llama_settings({"TYPEVET_LLAMA__AUTOLOAD": raw}).autoload is expected


def test_autoload_setting_rejects_another_value() -> None:
    with pytest.raises(ValueError, match="TYPEVET_LLAMA__AUTOLOAD"):
        load_llama_settings({"TYPEVET_LLAMA__AUTOLOAD": "maybe"})


# --- default: no parameter -----------------------------------------------------


def test_default_setting_sends_no_autoload_parameter() -> None:
    router = _Router({})
    settings = LlamaSettings(base_url="http://test")
    with llama_cpp_adapter(
        settings, transport=httpx.MockTransport(router.handle)
    ) as port:
        assert port.generate(_generation()).value == {"ok": True}
    scoring = LlamaCppCandidateScoringAdapter(
        base_url="http://test", client=router.client(), n_vocab=_N_VOCAB
    )
    scoring.score_candidates(_scoring())
    assert [q for _, q in router.requests] == ["", ""]
    assert [p for p, _ in router.requests] == [CHAT, "/completion"]


# --- sync generation -----------------------------------------------------------


def test_generation_busy_router_raises_busy_error_and_sends_no_retry() -> None:
    router = _Router({_OTHER: "loaded"})
    adapter = LlamaCppGenerationAdapter(
        base_url="http://test", client=router.client(), autoload=False
    )
    with pytest.raises(BackendBusyError) as caught:
        adapter.generate(_generation())
    assert caught.value.status_code == 400
    assert router.requests == [(CHAT, OFF), ("/models", ""), ("/props", "")]
    assert router.evictions == 0
    assert router.status[_OTHER] == "loaded"


def test_generation_idle_router_retries_once_with_autoload_on() -> None:
    router = _Router({})
    adapter = LlamaCppGenerationAdapter(
        base_url="http://test", client=router.client(), autoload=False
    )
    assert adapter.generate(_generation()).value == {"ok": True}
    assert router.requests == [(CHAT, OFF), ("/models", ""), (CHAT, ON)]
    assert router.status[_JUDGE] == "loaded"


@pytest.mark.parametrize("max_instances", [2, 0], ids=["two-slots", "unlimited"])
def test_generation_retries_when_the_router_has_a_free_slot(max_instances: int) -> None:
    router = _Router({_OTHER: "loaded"}, max_instances=max_instances)
    adapter = LlamaCppGenerationAdapter(
        base_url="http://test", client=router.client(), autoload=False
    )
    assert adapter.generate(_generation()).value == {"ok": True}
    assert router.requests == [(CHAT, OFF), ("/models", ""), ("/props", ""), (CHAT, ON)]
    assert router.evictions == 0


@pytest.mark.parametrize("status", ["loading", "sleeping"])
def test_generation_counts_a_loading_or_sleeping_model_as_running(status: str) -> None:
    router = _Router({_OTHER: status})
    adapter = LlamaCppGenerationAdapter(
        base_url="http://test", client=router.client(), autoload=False
    )
    with pytest.raises(BackendBusyError):
        adapter.generate(_generation())
    assert CHAT not in [p for p, _ in _after_first_400(router)]


def test_another_400_keeps_the_backend_http_error_and_reads_nothing() -> None:
    router = _Router({})
    adapter = LlamaCppGenerationAdapter(
        base_url="http://test", client=router.client(), autoload=False
    )
    request = GenerationRequest(prompt="x", schema=_SCHEMA, model="no-such-model")
    with pytest.raises(BackendHttpError) as caught:
        adapter.generate(request)
    assert not isinstance(caught.value, BackendBusyError)
    assert "not found" in caught.value.body_snippet
    assert router.requests == [(CHAT, OFF)]


def test_retry_that_is_still_not_loaded_raises_busy_error_once() -> None:
    router = _Router({})
    router.refuse_load = True
    adapter = LlamaCppGenerationAdapter(
        base_url="http://test", client=router.client(), autoload=False
    )
    with pytest.raises(BackendBusyError):
        adapter.generate(_generation())
    assert router.requests == [(CHAT, OFF), ("/models", ""), (CHAT, ON)]


@pytest.mark.parametrize(
    ("path", "response"),
    [
        ("/models", httpx.Response(500, text="down")),
        ("/models", httpx.Response(200, json={"data": "x"})),
        ("/models", httpx.Response(200, text="not json")),
        ("/props", httpx.Response(500, text="down")),
        ("/props", httpx.Response(200, json={"max_instances": True})),
    ],
    ids=["models-500", "models-shape", "models-text", "props-500", "props-shape"],
)
def test_failed_read_raises_busy_error_and_never_loads(
    path: str, response: httpx.Response
) -> None:
    router = _Router({_OTHER: "loaded"})
    router.local[path] = response
    adapter = LlamaCppGenerationAdapter(
        base_url="http://test", client=router.client(), autoload=False
    )
    with pytest.raises(BackendBusyError):
        adapter.generate(_generation())
    assert ON not in [q for _, q in router.requests]
    assert router.evictions == 0


# --- async generation ----------------------------------------------------------


def test_async_generation_busy_router_raises_busy_error() -> None:
    router = _Router({_OTHER: "loaded"})
    adapter = AsyncLlamaCppGenerationAdapter(
        base_url="http://test", client=router.async_client(), autoload=False
    )
    with pytest.raises(BackendBusyError):
        asyncio.run(adapter.generate(_generation()))
    assert router.requests == [(CHAT, OFF), ("/models", ""), ("/props", "")]
    assert router.evictions == 0


def test_async_generation_idle_router_retries_once_with_autoload_on() -> None:
    router = _Router({})
    adapter = AsyncLlamaCppGenerationAdapter(
        base_url="http://test", client=router.async_client(), autoload=False
    )
    result = asyncio.run(adapter.generate(_generation()))
    assert result.value == {"ok": True}
    assert router.requests == [(CHAT, OFF), ("/models", ""), (CHAT, ON)]


# --- scoring -------------------------------------------------------------------


def test_scoring_busy_router_raises_busy_error() -> None:
    router = _Router({_OTHER: "loaded"})
    adapter = LlamaCppCandidateScoringAdapter(
        base_url="http://test", client=router.client(), n_vocab=_N_VOCAB, autoload=False
    )
    with pytest.raises(BackendBusyError):
        adapter.score_candidates(_scoring())
    assert router.requests == [("/completion", OFF), ("/models", ""), ("/props", "")]
    assert router.evictions == 0


def test_scoring_idle_router_retries_once_with_autoload_on() -> None:
    router = _Router({})
    adapter = LlamaCppCandidateScoringAdapter(
        base_url="http://test", client=router.client(), n_vocab=_N_VOCAB, autoload=False
    )
    result = adapter.score_candidates(_scoring())
    assert {c.label for c in result.candidates} == {"a", "b"}
    assert router.requests == [
        ("/completion", OFF),
        ("/models", ""),
        ("/completion", ON),
    ]


# --- the judgevet bridge over open_judgment -----------------------------------


def test_bridge_raises_provider_busy_error_and_never_reaches_completion() -> None:
    router = _Router({_OTHER: "loaded"})
    factory = provider_factory(
        lambda: open_judgment(_ENVIRON, transport=httpx.MockTransport(router.handle))
    )
    with pytest.raises(ProviderBusyError), factory() as port:
        port.system_one("Charged twice.", {"q": JevNoul()}, _JUDGE)
    paths = [p for p, _ in router.requests]
    assert "/completion" not in paths
    assert OFF in router.requests[0][1]
    assert [p for p, _ in _after_first_400(router)] == ["/models", "/props"]
    assert router.evictions == 0
    assert router.status[_OTHER] == "loaded"


def test_bridge_loads_the_judge_on_an_idle_router_and_answers() -> None:
    router = _Router({})
    factory = provider_factory(
        lambda: open_judgment(_ENVIRON, transport=httpx.MockTransport(router.handle))
    )
    with factory() as port:
        response = port.system_one("Charged twice.", {"q": JevNoul()}, _JUDGE)
    assert set(response.answers) == {"q"}
    assert router.status[_JUDGE] == "loaded"
    assert router.evictions == 0
    local = ("/models", "/props")
    assert [p for p, _ in router.requests].count("/models") == 1
    assert OFF in router.requests[0][1]
    assert router.requests[1] == ("/models", "")
    assert ON in router.requests[2][1]
    assert all(OFF in q for p, q in router.requests[3:] if p not in local)
    assert ("/completion", OFF) in router.requests

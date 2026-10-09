"""Unit tests: the llama.cpp judgment names its model on a template mismatch ([#425][i425]).

``open_judgment`` with ``TYPEVET_BACKEND=llama_cpp`` probes the model that
``TYPEVET_LLAMA__MULTIMODAL_MODEL`` names. A fake transport serves vision on
``/props`` and a Gemma 3 template on ``/apply-template``. The ``ValueError``
then names the probed model id, ``settings.multimodal_model`` with the variable
behind it, and the ``<|turn>`` requirement. An explicit ``model`` argument is
named as the source instead. No request leaves the process.

Examples:
    ```bash
    uv run pytest -q tests/unit/test_open_judgment_template_error.py
    ```

See Also:
    - [typevet.adapters.inbound.backend_settings][]: ``open_judgment``
    - [typevet.adapters.outbound.llama_cpp.gemma_native_vision_factory][]: Probe
    - [typevet.adapters.inbound.settings][]: ``load_llama_settings``

[i425]: https://github.com/Alberto-Codes/typevet/issues/425
"""

from __future__ import annotations

import json

import httpx
import pytest

from typevet.adapters.inbound.backend_settings import open_judgment
from typevet.adapters.inbound.settings import load_llama_settings
from typevet.adapters.outbound.llama_cpp.gemma_native_vision_factory import (
    open_gemma_native_vision_judgment,
)

pytestmark = pytest.mark.unit

_GEMMA3_RENDERED = "<start_of_turn>user\nhello<end_of_turn>\n<start_of_turn>model\n"
_PROPS = {"modalities": {"vision": True}, "media_marker": "<__media__>"}
_VARIABLE = "TYPEVET_LLAMA__MULTIMODAL_MODEL"
_SETTINGS_LABEL = f"from settings.multimodal_model ({_VARIABLE})"
_MISMATCH = r"^expected NATIVE_GEMMA4_TURN, got"


def _gemma3_router(request: httpx.Request) -> httpx.Response:
    """Serve vision props and a Gemma 3 rendered template.

    Args:
        request: The request the client sent.

    Returns:
        A JSON response for ``/props`` or ``/apply-template``, else 404.
    """
    if request.url.path == "/props":
        return httpx.Response(200, json=_PROPS)
    if request.url.path == "/apply-template":
        return httpx.Response(200, content=json.dumps({"prompt": _GEMMA3_RENDERED}))
    return httpx.Response(404)


def _mismatch_message(environ: dict[str, str]) -> str:
    """Open the llama.cpp judgment and return the mismatch message.

    Args:
        environ: Variables for ``open_judgment``.

    Returns:
        The ``ValueError`` text.
    """
    transport = httpx.MockTransport(_gemma3_router)
    with (
        pytest.raises(ValueError, match=_MISMATCH) as info,
        open_judgment(environ, transport=transport),
    ):
        pass
    return str(info.value)


def test_mismatch_names_the_configured_model_and_variable() -> None:
    """A configured Gemma 3 id is named with the variable that chose it."""
    message = _mismatch_message(
        {"TYPEVET_BACKEND": "llama_cpp", _VARIABLE: "gemma-3-custom-mm"}
    )
    assert "got native_gemma3_turn" in message
    assert "gemma-3-custom-mm" in message
    assert _SETTINGS_LABEL in message
    assert "<|turn>" in message


def test_mismatch_names_the_default_model_when_the_variable_is_unset() -> None:
    """The default Gemma 3 id fails the check and the message says so."""
    message = _mismatch_message({"TYPEVET_BACKEND": "llama_cpp"})
    assert "gemma-3-4b-it-q4km-mm" in message
    assert _SETTINGS_LABEL in message


def test_mismatch_names_the_model_argument_when_one_is_passed() -> None:
    """An explicit ``model`` id is named with the argument as its source."""
    settings = load_llama_settings({_VARIABLE: "gemma-3-unused-mm"})
    with (
        httpx.Client(
            base_url="http://router.test", transport=httpx.MockTransport(_gemma3_router)
        ) as client,
        pytest.raises(ValueError, match=_MISMATCH) as info,
        open_gemma_native_vision_judgment(
            settings=settings, model="gemma-3-explicit-mm", http_client=client
        ),
    ):
        pass
    message = str(info.value)
    assert "for model 'gemma-3-explicit-mm' from the model argument" in message
    assert "settings.multimodal_model" not in message
    assert "gemma-3-unused-mm" not in message

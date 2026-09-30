"""Probe the served chat template and require native Gemma 4 framing (#327).

#324 found that the text wording runs framed Gemma differently on the two
backends. The llama.cpp decoder alias serves no native template, so typevet
wrote a ChatML prefix; vLLM applied the model's own Gemma 4 template. The
receipt recorded the constant ``vllm_chat`` instead of a probe.

These probes ask each server how it renders one user turn and classify the
result. Both probes send ``add_generation_prompt`` and thinking off
(``chat_template_kwargs``), the settings of the scoring requests.
``probe_llama_template`` posts to llama.cpp ``/apply-template``.
``probe_vllm_template`` posts the same turn to vLLM ``/tokenize`` with
``return_token_strs`` and classifies the joined token strings.
vLLM ``TokenizeChatRequest`` and ``TokenizeResponse`` define these fields.
``require_native_template`` refuses any family other than native Gemma 4
unless ``ALLOW_DEGRADED_ENV`` is ``1``.

Attributes:
    ALLOW_DEGRADED_ENV (str): The variable that allows a non-native template.
    TEXT_JUDGE (str): The router alias that serves the native Gemma 4
        template for text requests.

Examples:
    ```python
    served = require_native_template(probe_llama_template(client, TEXT_JUDGE), {})
    ```

See Also:
    - [typevet.adapters.outbound.gemma.served_template][]: the classifier
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

import httpx

from typevet.adapters.outbound.gemma import (
    ServedTemplateClass,
    classify_served_template,
)

ALLOW_DEGRADED_ENV: Final[str] = "TYPEVET_WORDING_ALLOW_DEGRADED_TEMPLATE"
TEXT_JUDGE: Final[str] = "gemma-4-31b-kv9-q4km-mm"
_PROBE_MESSAGES: Final[tuple[dict[str, str], ...]] = (
    {"role": "user", "content": "hello"},
)
_THINKING_OFF: Final[dict[str, bool]] = {"enable_thinking": False}


def probe_llama_template(client: httpx.Client, model: str) -> ServedTemplateClass:
    """Classify the template the llama.cpp router serves for ``model``.

    Args:
        client: A client whose base URL is the router root.
        model: The router alias.

    Returns:
        The served-template family of the ``/apply-template`` prompt.
    """
    rendered = client.post(
        "/apply-template",
        json={
            "model": model,
            "messages": list(_PROBE_MESSAGES),
            "add_generation_prompt": True,
            "chat_template_kwargs": dict(_THINKING_OFF),
        },
    )
    return classify_served_template(str(rendered.raise_for_status().json()["prompt"]))


def probe_vllm_template(client: httpx.Client, model: str) -> ServedTemplateClass:
    """Classify the chat template the vLLM server applies for ``model``.

    Args:
        client: A client whose base URL is the server root.
        model: The served model name.

    Returns:
        The served-template family of the joined ``/tokenize`` token strings.

    Raises:
        TypeError: When the reply holds no list of token strings.
    """
    reply = client.post(
        "/tokenize",
        json={
            "model": model,
            "messages": list(_PROBE_MESSAGES),
            "add_generation_prompt": True,
            "chat_template_kwargs": dict(_THINKING_OFF),
            "return_token_strs": True,
        },
    )
    strings = reply.raise_for_status().json().get("token_strs")
    if not isinstance(strings, list):
        msg = "vLLM /tokenize reply holds no token_strs list"
        raise TypeError(msg)
    return classify_served_template("".join(str(s) for s in strings))


def require_native_template(
    served: ServedTemplateClass, environ: Mapping[str, str]
) -> ServedTemplateClass:
    """Return ``served`` when it is native Gemma 4 or the override is set.

    Args:
        served: The probed served-template family.
        environ: The variables to read ``ALLOW_DEGRADED_ENV`` from.

    Returns:
        ``served``, unchanged.

    Raises:
        ValueError: When ``served`` is not native Gemma 4 and
            ``ALLOW_DEGRADED_ENV`` is not ``1``.
    """
    if served is ServedTemplateClass.NATIVE_GEMMA4_TURN:
        return served
    if environ.get(ALLOW_DEGRADED_ENV, "").strip() == "1":
        return served
    msg = (
        f"the server serves {served.value}, not native_gemma4_turn; "
        f"set {ALLOW_DEGRADED_ENV}=1 to run anyway"
    )
    raise ValueError(msg)

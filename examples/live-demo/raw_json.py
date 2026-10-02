"""Raw JSON trimmers for the live demo's behind-the-scenes panel.

The panel shows the raw ``/completion`` request and reply. The reply holds a
log probability for every vocabulary entry and the request holds the base64
image, so these helpers cut long lists and elide image payloads before the
page shows them. They change only what the page shows, not what typevet sends.

``bts.py`` imports this module from the same directory.

Examples:
    ```python
    shown = trim_completion(reply)
    ```

See Also:
    - [typevet.runtime.open_gemma_native_vision_judgment][]: Judgment session.
    - examples/live-demo/README.md: How to run the live demo.
"""

from __future__ import annotations

from typing import Any

TOP_N = 8
MAX_LIST = 12


def trim(obj: Any) -> Any:
    """Shorten long lists recursively so the raw JSON stays readable.

    Args:
        obj: Parsed JSON value.

    Returns:
        The value with each list cut to ``MAX_LIST`` items and a note.
    """
    if isinstance(obj, list):
        head = [trim(x) for x in obj[:MAX_LIST]]
        if len(obj) > MAX_LIST:
            head.append(f"... (+{len(obj) - MAX_LIST} more, trimmed by demo)")
        return head
    if isinstance(obj, dict):
        return {k: trim(v) for k, v in obj.items()}
    return obj


def sanitize_request(body: Any) -> Any:
    """Replace base64 image payloads with a length placeholder.

    Args:
        body: Parsed ``/completion`` request body.

    Returns:
        The body with each image payload replaced by its length.
    """
    if isinstance(body, dict) and isinstance(body.get("prompt"), dict):
        prompt = dict(body["prompt"])
        data = prompt.get("multimodal_data")
        if isinstance(data, list):
            prompt["multimodal_data"] = [
                f"<base64 image data elided: {len(d)} chars>" for d in data
            ]
        body = {**body, "prompt": prompt}
    return body


def trim_completion(resp: Any) -> Any:
    """Keep only the top entries of the huge n_vocab logprob list.

    Args:
        resp: Parsed ``/completion`` reply.

    Returns:
        The reply with the logprob lists cut to ``TOP_N`` entries.
    """
    if not isinstance(resp, dict):
        return resp
    out = dict(resp)
    cps = out.get("completion_probabilities")
    if isinstance(cps, list) and cps and isinstance(cps[0], dict):
        first = dict(cps[0])
        tops = first.get("top_logprobs")
        if isinstance(tops, list):
            first["top_logprobs"] = tops[:TOP_N] + (
                [f"... (+{len(tops) - TOP_N} more vocabulary entries, trimmed by demo)"]
                if len(tops) > TOP_N
                else []
            )
        out["completion_probabilities"] = [first] + (
            [f"... (+{len(cps) - 1} more)"] if len(cps) > 1 else []
        )
    return trim(out)

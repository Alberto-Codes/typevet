"""Live web demo: typevet typed judgments on local Gemma 4, text and image.

The page asks one typed Choice about an expense claim and a receipt image, and
three typed questions (Noul, Choice and Score) about a customer message. A
"behind the scenes" panel shows the exact prompt, the candidate tokens, their
raw log probabilities and the router calls (see ``bts.py``).

This file is the entry point. ``demo.py`` holds the warm judgment session,
``judge.py`` the typed questions and the judgment payloads, and ``routes.py``
the HTTP handlers.

Run the command from the repository root, because the gallery reads the CORD
receipts from ``tests/fixtures/cord/expense_smoke/``:

    uv run python examples/live-demo/server.py

The server binds 127.0.0.1 only. ``LIVE_UI_PORT`` sets the port (default
8765). Model calls go to the llama.cpp router that ``TYPEVET_LLAMA__BASE_URL``
names (default ``http://127.0.0.1:8090``). ``TYPEVET_LLAMA__MULTIMODAL_MODEL``
names the model (default ``gemma-4-31b-kv9-q4km-mm``). Each judgment writes a
JSON receipt under ``typevet-receipts/``.

Examples:
    ```bash
    LIVE_UI_PORT=8766 uv run python examples/live-demo/server.py
    ```

See Also:
    - [typevet.runtime.open_gemma_native_vision_judgment][]: Judgment session.
    - examples/live-demo/README.md: Prerequisites and run of show.
"""

from __future__ import annotations

import os
import sys
from http.server import ThreadingHTTPServer

import httpx
from demo import Demo
from routes import Handler, one_line

from typevet.domain import GenerationError

HOST = "127.0.0.1"
PORT = int(os.environ.get("LIVE_UI_PORT", "8765"))


def main() -> int:
    """Warm the session, then serve the page until Ctrl+C.

    Returns:
        0 after a clean stop, 1 when startup fails.
    """
    try:
        demo = Demo()
    except (httpx.HTTPError, GenerationError, ValueError, OSError) as exc:
        print(f"startup failed: {one_line(exc)}", file=sys.stderr)
        return 1
    Handler.demo = demo
    httpd = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"ready  http://{HOST}:{PORT}/", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
        demo.stack.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

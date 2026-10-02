"""HTTP handlers of the live demo web page.

``Handler`` serves the page, the status, the gallery images and thumbnails,
and the two judgment endpoints ``POST /api/image`` and ``POST /api/text``.
It turns each error into a one-line JSON error with the matching HTTP status,
and never sends a stack trace to the page.

``server.py`` imports this module from the same directory.

Examples:
    ```python
    Handler.demo = Demo()
    ThreadingHTTPServer(("127.0.0.1", 8765), Handler).serve_forever()
    ```

See Also:
    - [typevet.domain.ScoringValidationError][]: Typed refusal of an input.
    - examples/live-demo/README.md: Prerequisites and run of show.
"""

from __future__ import annotations

import json
import logging
import sys
import threading
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler
from io import BytesIO
from pathlib import Path
from typing import Any, ClassVar

import httpx
from demo import CORD, RECEIPT_IDS, Demo, DemoError
from PIL import Image, ImageOps

from typevet.domain import (
    GenerationError,
    JudgmentValidationError,
    ScoringValidationError,
)

HERE = Path(__file__).resolve().parent
THUMB_WIDTH = 360
MAX_BODY = 25 * 1024 * 1024

LOG = logging.getLogger(__name__)
THUMBS: dict[str, bytes] = {}
THUMB_LOCK = threading.Lock()


def thumbnail(rid: str) -> bytes | None:
    """Return a cached 360 px wide JPEG of a gallery receipt (display only).

    Pillow builds the thumbnail in memory. The model always gets the original
    PNG bytes.

    Args:
        rid: Gallery receipt id, such as ``R01``.

    Returns:
        The JPEG bytes, or None when Pillow cannot read the image.
    """
    with THUMB_LOCK:
        if rid not in THUMBS:
            try:
                with Image.open(CORD / f"{rid}.png") as img:
                    shown = ImageOps.exif_transpose(img).convert("RGB")
                    height = max(1, round(shown.height * THUMB_WIDTH / shown.width))
                    small = shown.resize((THUMB_WIDTH, height))
                    out = BytesIO()
                    small.save(out, format="JPEG", quality=82)
            except OSError:
                return None
            THUMBS[rid] = out.getvalue()
        return THUMBS[rid]


def one_line(exc: BaseException) -> str:
    """Return the error message on one line, at most 300 characters.

    Args:
        exc: Error to show.

    Returns:
        The shortened message, or the error type name.
    """
    text = " ".join(str(exc).split()) or type(exc).__name__
    return text[:300]


class Handler(BaseHTTPRequestHandler):
    """Serves the page, the gallery images and the two judgment endpoints.

    Attributes:
        demo (Demo): The warm demo that ``main`` sets before serving.
        server_version (str): Value of the ``Server`` header.

    Examples:
        ```python
        Handler.demo = Demo()
        ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
        ```
    """

    demo: ClassVar[Demo]
    server_version = "typevet-live-ui/1"

    def log_message(self, format: str, *args: Any) -> None:
        """Write one access log line to stderr with the local time.

        The base class passes a printf-style format and the values that fill it.
        """
        stamp = datetime.now(UTC).astimezone()
        sys.stderr.write(f"[{stamp:%H:%M:%S}] {format % args}\n")

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        """Send one response with a cache header.

        Args:
            code: HTTP status.
            body: Response bytes.
            ctype: Content type.
        """
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header(
            "Cache-Control",
            "no-store" if ctype.startswith("application/json") else "max-age=3600",
        )
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj: Any) -> None:
        """Send one JSON response.

        Args:
            code: HTTP status.
            obj: Value to encode.
        """
        self._send(code, json.dumps(obj).encode(), "application/json; charset=utf-8")

    def do_GET(self) -> None:
        """Serve the page, the status, a gallery image or a thumbnail."""
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            self._send(
                200, (HERE / "index.html").read_bytes(), "text/html; charset=utf-8"
            )
        elif path == "/api/status":
            self._json(200, self.demo.status())
        elif path.startswith("/img/"):
            name = path[len("/img/") :]
            if name.removesuffix(".png") in RECEIPT_IDS and name.endswith(".png"):
                self._send(200, (CORD / name).read_bytes(), "image/png")
            else:
                self._json(404, {"error": "not found"})
        elif path.startswith("/thumb/"):
            name = path[len("/thumb/") :]
            if name.removesuffix(".jpg") in RECEIPT_IDS and name.endswith(".jpg"):
                thumb = thumbnail(name.removesuffix(".jpg"))
                if thumb is not None:
                    self._send(200, thumb, "image/jpeg")
                else:
                    self._send(
                        200,
                        (CORD / f"{name.removesuffix('.jpg')}.png").read_bytes(),
                        "image/png",
                    )
            else:
                self._json(404, {"error": "not found"})
        else:
            self._json(404, {"error": "not found"})

    def _read_json(self) -> dict[str, Any]:
        """Read and parse the request body.

        Returns:
            The JSON object.

        Raises:
            DemoError: When the body is missing, too large, not JSON or not an
                object.
        """
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > MAX_BODY:
            raise DemoError("request body missing or too large")
        try:
            req = json.loads(self.rfile.read(length))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise DemoError("request body is not JSON") from exc
        if not isinstance(req, dict):
            raise DemoError("request body must be a JSON object")
        return req

    def do_POST(self) -> None:
        """Run an image or a text judgment and send the result as JSON."""
        path = self.path.split("?", 1)[0]
        try:
            req = self._read_json()
            if path == "/api/image":
                out = self.demo.judge_image(req)
                self._json(422 if out.get("rejected") else 200, out)
            elif path == "/api/text":
                self._json(200, self.demo.judge_text(req))
            else:
                self._json(404, {"error": "not found"})
        except DemoError as exc:
            self._json(400, {"error": one_line(exc)})
        except (ScoringValidationError, JudgmentValidationError) as exc:
            self._json(422, {"error": f"{type(exc).__name__}: {one_line(exc)}"})
        except (GenerationError, httpx.HTTPError) as exc:
            self._json(502, {"error": f"model call failed: {one_line(exc)}"})
        except Exception as exc:  # demo server: never leak a stack trace to the page
            LOG.exception("unexpected error")
            self._json(500, {"error": f"unexpected error: {one_line(exc)}"})

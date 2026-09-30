"""Serve judgevet's MCP stdio entry point on the typevet bridge (#284).

Usage: ``python -m tests.fixtures.judgevet_mcp_app``. The server uses the
offline typevet session from ``tests.fixtures.judgevet_bridge``, so it makes
no network call. Protocol frames use stdout.
"""

from __future__ import annotations

from judgevet.adapters.inbound.mcp_entrypoint import main as serve

from tests.fixtures.judgevet_bridge import FAKE_MODEL, open_fake_session
from typevet.adapters.inbound.judgevet import provider_factory


def main() -> int:
    """Serve MCP stdio with the bridge's owning provider factory.

    Returns:
        The entry point's exit status.
    """
    return serve(provider_factory=provider_factory(open_fake_session), model=FAKE_MODEL)


if __name__ == "__main__":
    raise SystemExit(main())

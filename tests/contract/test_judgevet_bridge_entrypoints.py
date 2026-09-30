"""judgevet's CLI and MCP entry points run one judgment on the bridge (#284).

Both run offline over typevet fakes: the CLI through ``create_cli_app(port=)``
and the MCP stdio server through ``main(provider_factory=)`` in a child
process that an MCP SDK client drives.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import mcp
import pytest
from judgevet.adapters.inbound.cli import create_cli_app
from typer.testing import CliRunner

from tests.fixtures.judgevet_bridge import FAKE_MODEL, STATE, judgment_port
from typevet.adapters.inbound.judgevet import TypevetSystemOnePort

pytestmark = pytest.mark.contract

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_cli_runs_one_judgment_on_a_borrowed_bridge_port() -> None:
    questions = {
        "route": {
            "type": "choice",
            "instructions": "Which team should handle the state?",
            "criteria": {"billing": "A payment problem", "technical": "A fault"},
        }
    }
    port = TypevetSystemOnePort(judgment_port())
    result = CliRunner().invoke(
        create_cli_app(port=port),
        [STATE, json.dumps(questions), "--model", FAKE_MODEL, "--json"],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)
    assert data["model"] == FAKE_MODEL
    assert data["answers"]["route"]["choice"] == "billing"
    assert data["answers"]["route"]["probabilities"] == pytest.approx(
        {"billing": 0.75, "technical": 0.25}
    )


async def _ask_noul() -> mcp.types.CallToolResult:
    server = mcp.StdioServerParameters(
        command=sys.executable,
        args=["-m", "tests.fixtures.judgevet_mcp_app"],
        cwd=REPO_ROOT,
    )
    async with (
        mcp.stdio_client(server) as (read, write),
        mcp.ClientSession(read, write) as session,
    ):
        await session.initialize()
        return await session.call_tool(
            "ask_noul",
            {
                "state": STATE,
                "instruction": "Does the state report a duplicate charge?",
            },
        )


def test_mcp_serves_one_judgment_through_the_bridge_factory() -> None:
    result = asyncio.run(asyncio.wait_for(_ask_noul(), timeout=60))
    assert not result.is_error, result.content
    data = result.structured_content
    assert data is not None
    assert data["model"] == FAKE_MODEL
    assert data["noul"] == pytest.approx(0.8)

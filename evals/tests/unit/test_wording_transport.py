"""Unit checks for the wording transport's helpers and construction (#306).

Examples:
    ```bash
    uv run pytest -q evals/tests/unit/test_wording_transport.py
    ```

See Also:
    - [typevet_evals.wording.transport][]: the transport module
"""

from __future__ import annotations

import pytest
from google.adk.models import LlmRequest
from google.genai import types
from judgevet import Usage
from judgevet.domain.questions import Noul
from judgevet.testing import FakeSystemOnePort

from typevet_evals.wording import WordingTransport, last_user_text, usage_metadata

pytestmark = pytest.mark.unit

SEED = Noul(instructions="Is this message a scam?")


def _content(role: str, text: str) -> types.Content:
    """Return one content with one text part."""
    return types.Content(role=role, parts=[types.Part.from_text(text=text)])


def test_last_user_text_reads_the_last_user_turn() -> None:
    request = LlmRequest(
        contents=[_content("user", "a"), _content("user", "b"), _content("model", "c")]
    )

    assert last_user_text(request) == "b"


def test_a_request_without_user_text_is_refused() -> None:
    request = LlmRequest(contents=[_content("model", "x")])

    with pytest.raises(ValueError, match="no user text"):
        last_user_text(request)


def test_usage_metadata_sums_the_counts() -> None:
    usage = usage_metadata(Usage(input_tokens=7, output_tokens=1))

    assert usage is not None
    assert (usage.prompt_token_count, usage.candidates_token_count) == (7, 1)
    assert usage.total_token_count == 8


@pytest.mark.parametrize("usage", [None, Usage()])
def test_no_counts_give_no_usage_metadata(usage: Usage | None) -> None:
    assert usage_metadata(usage) is None


def test_a_missing_mapping_key_is_refused_at_construction() -> None:
    with pytest.raises(ValueError, match="'is_scam' is not in the wording mapping"):
        WordingTransport(
            port=FakeSystemOnePort(),
            mapping={"other": "text"},
            key="is_scam",
            seed=SEED,
            judge_model="m",
        )


def test_the_mapping_is_held_by_reference() -> None:
    mapping = {"is_scam": "text"}
    transport = WordingTransport(
        port=FakeSystemOnePort(),
        mapping=mapping,
        key="is_scam",
        seed=SEED,
        judge_model="m",
    )

    assert transport.mapping is mapping
    assert transport.capabilities.output_schema_and_tools is False

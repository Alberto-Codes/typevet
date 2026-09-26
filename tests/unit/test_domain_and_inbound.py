"""Unit tests for domain and inbound API."""

from __future__ import annotations

import pytest

from typevet.adapters.inbound import generate
from typevet.domain.models import GenerationRequest
from typevet.testing import StaticGenerationFake

SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "ok": {"type": "boolean"},
    },
    "required": ["name", "ok"],
    "additionalProperties": False,
}


@pytest.mark.unit
def test_generation_request_rejects_empty_prompt() -> None:
    with pytest.raises(ValueError, match="prompt"):
        GenerationRequest(prompt="  ", schema=SCHEMA, model="m")


@pytest.mark.unit
def test_generation_request_rejects_non_object_schema_type() -> None:
    with pytest.raises(ValueError, match="object"):
        GenerationRequest(
            prompt="hi",
            schema={"type": "array"},
            model="m",
        )


@pytest.mark.unit
def test_inbound_generate_uses_port() -> None:
    fake = StaticGenerationFake({"name": "ada", "ok": True})
    result = generate(
        fake,
        prompt="Return a person.",
        schema=SCHEMA,
        model="fake-model",
    )
    assert result.value["name"] == "ada"
    assert result.model == "fake-model"

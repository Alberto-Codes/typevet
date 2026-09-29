"""Unit tests for the shared generation helpers (#214).

The chat completion parser, the schema check, the request schema check (#226),
the HTTP error constants and the vLLM body builder live in neutral modules.
Each backend keeps its own error messages and its own JSON-body parse exception class.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

import httpx
import jsonschema
import pytest

from typevet.adapters.outbound import llama_cpp_http, vllm_http
from typevet.adapters.outbound.chat_completion import (
    check_request_schema,
    extract_content,
    validated_value,
)
from typevet.adapters.outbound.http_errors import (
    BODY_SNIPPET_MAX,
    HTTP_ERROR_STATUS,
    body_snippet,
)
from typevet.adapters.outbound.vllm_generation import generation_body
from typevet.domain.errors import GenerationError, SchemaValidationError
from typevet.domain.media import MEDIA_MARKER, ImageInput
from typevet.domain.models import GenerationRequest

_SCHEMA = {
    "type": "object",
    "properties": {"n": {"type": "integer"}},
    "required": ["n"],
    "additionalProperties": False,
}


def _reply(content: object) -> dict[str, Any]:
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


@pytest.mark.unit
def test_http_error_constants_are_shared_and_reexported_by_llama_cpp_http() -> None:
    assert HTTP_ERROR_STATUS == 400
    assert BODY_SNIPPET_MAX == 500
    assert body_snippet("x" * 600) == "x" * 500
    assert llama_cpp_http.HTTP_ERROR_STATUS is HTTP_ERROR_STATUS
    assert llama_cpp_http.BODY_SNIPPET_MAX is BODY_SNIPPET_MAX
    assert llama_cpp_http.body_snippet is body_snippet


@pytest.mark.unit
def test_vllm_http_does_not_import_llama_cpp_http() -> None:
    source = vllm_http.__file__
    assert source is not None
    with open(source, encoding="utf-8") as handle:
        imports = [line for line in handle if line.startswith(("from ", "import "))]
    assert imports
    assert not [line for line in imports if "llama_cpp" in line]


@pytest.mark.unit
@pytest.mark.parametrize("backend", ["llama.cpp", "vLLM"])
def test_extract_content_names_the_backend(backend: str) -> None:
    assert extract_content(_reply('{"n": 1}'), backend) == '{"n": 1}'
    with pytest.raises(GenerationError, match=f"^{backend} response missing choices"):
        extract_content({"choices": []}, backend)
    with pytest.raises(GenerationError, match=f"^{backend} returned empty message"):
        extract_content(_reply("  "), backend)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("raw", "error", "message"),
    [
        ("not json", GenerationError, "not valid JSON"),
        ("[1]", SchemaValidationError, "root must be an object"),
        ('{"n": NaN}', SchemaValidationError, "non-finite"),
        ('{"n": "one"}', SchemaValidationError, "output failed schema"),
    ],
)
def test_validated_value_rejects_bad_content(
    raw: str, error: type[GenerationError], message: str
) -> None:
    with pytest.raises(error, match=message) as caught:
        validated_value(raw, _SCHEMA)
    assert type(caught.value) is error


@pytest.mark.unit
def test_validated_value_returns_the_object() -> None:
    assert validated_value('{"n": 3}', _SCHEMA) == {"n": 3}


@pytest.mark.unit
def test_backends_keep_their_own_json_body_parse_exception_class() -> None:
    # A UnicodeDecodeError is a ValueError, not a JSONDecodeError.
    bad_bytes = httpx.Response(200, content=b"\xff\xfe\xfa")
    with pytest.raises(GenerationError, match="vLLM returned non-JSON"):
        vllm_http.parse_json_response(bad_bytes)
    with pytest.raises(UnicodeDecodeError):
        llama_cpp_http.parse_json_response(bad_bytes)


@pytest.mark.unit
def test_generation_body_text_and_image() -> None:
    text = generation_body(
        GenerationRequest(prompt="count", schema=_SCHEMA, model="m"), _SCHEMA
    )
    assert text == {
        "model": "m",
        "messages": [{"role": "user", "content": "count"}],
        "temperature": 0,
        "structured_outputs": {"json": _SCHEMA},
        "add_generation_prompt": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    image = ImageInput(data=b"\x89PNG", mime_type="image/png")
    with_image = generation_body(
        GenerationRequest(
            prompt=f"{MEDIA_MARKER}\nlook", schema=_SCHEMA, model="m", media=(image,)
        ),
        _SCHEMA,
    )
    content = with_image["messages"][0]["content"]
    assert isinstance(content, list)
    assert json.dumps(content).count("image_url") >= 1


def _bad_schema(title: str) -> dict[str, Any]:
    return {
        "type": "object",
        "title": title,
        "properties": {"n": {"type": "not-a-type"}},
    }


@pytest.mark.unit
def test_check_request_schema_raises_value_error_on_a_malformed_schema() -> None:
    with pytest.raises(
        ValueError, match=r"^schema is not a valid JSON Schema: "
    ) as caught:
        check_request_schema(_bad_schema("malformed"))
    assert type(caught.value) is ValueError
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__ is True


@pytest.mark.unit
def test_check_request_schema_rejects_a_schema_json_cannot_encode() -> None:
    schema = {"type": "object", "title": "not-json", "default": {1, 2}}
    with pytest.raises(ValueError, match=r"^schema is not a valid JSON Schema: "):
        check_request_schema(schema)


@pytest.mark.unit
def test_check_request_schema_runs_the_meta_schema_check_once_per_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    real = jsonschema.Draft202012Validator.check_schema

    def spy(schema: Any, *args: Any, **kwargs: Any) -> None:
        calls.append(schema)
        real(schema, *args, **kwargs)

    monkeypatch.setattr(jsonschema.Draft202012Validator, "check_schema", spy)
    # A fresh title keeps an earlier in-process run from filling the cache.
    schema = {**_SCHEMA, "title": f"spy-once-{uuid.uuid4()}"}
    check_request_schema(schema)
    check_request_schema(dict(reversed(list(schema.items()))))
    assert len(calls) == 1


@pytest.mark.unit
def test_check_request_schema_accepts_number_bounds_and_multiple_of() -> None:
    schema = {
        "type": "object",
        "title": "bounds",
        "properties": {
            "n": {"type": "integer", "minimum": 0, "maximum": 9, "multipleOf": 3},
            "x": {"type": "number", "exclusiveMinimum": 0.5, "multipleOf": 0.25},
        },
    }
    check_request_schema(schema)

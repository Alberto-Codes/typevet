"""Unit tests for structured generation error types."""

from __future__ import annotations

import httpx
import pytest

from typevet.adapters.outbound.llama_cpp.http_mapping import (
    BODY_SNIPPET_MAX,
    map_http_status,
    map_transport_error,
)
from typevet.domain.errors import (
    BackendHttpError,
    GenerationError,
    SchemaValidationError,
    TransportError,
)


@pytest.mark.unit
def test_transport_error_metadata() -> None:
    err = TransportError("llama.cpp request failed: down")
    assert err.status_code is None
    assert err.body_snippet is None
    assert isinstance(err, GenerationError)


@pytest.mark.unit
def test_backend_http_error_metadata() -> None:
    err = BackendHttpError(
        "llama.cpp HTTP 502: bad gateway",
        status_code=502,
        body_snippet="bad gateway",
    )
    assert err.status_code == 502
    assert err.body_snippet == "bad gateway"
    assert isinstance(err, GenerationError)


@pytest.mark.unit
def test_map_transport_error_from_httpx() -> None:
    exc = httpx.ConnectError("refused")
    err = map_transport_error(exc)
    assert isinstance(err, TransportError)
    assert "request failed" in str(err)


@pytest.mark.unit
def test_map_http_status_truncates_body() -> None:
    long_body = "x" * (BODY_SNIPPET_MAX + 50)
    response = httpx.Response(500, text=long_body)
    err = map_http_status(response)
    assert isinstance(err, BackendHttpError)
    assert err.status_code == 500
    assert len(err.body_snippet) == BODY_SNIPPET_MAX
    assert err.body_snippet == long_body[:BODY_SNIPPET_MAX]


@pytest.mark.unit
def test_schema_validation_still_subclasses_generation() -> None:
    err = SchemaValidationError("bad", payload={"a": 1})
    assert isinstance(err, GenerationError)
    assert err.payload == {"a": 1}

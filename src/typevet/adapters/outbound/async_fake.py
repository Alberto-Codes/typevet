"""Offline async fake generation adapter for unit and contract tests.

Examples:
    ```python
    from typevet.adapters.outbound.async_fake import AsyncFakeGenerationAdapter
    from typevet.domain.models import GenerationRequest

    import asyncio

    asyncio.run(
        AsyncFakeGenerationAdapter(value={"answer": 1}).generate(
            GenerationRequest(
                prompt="n",
                schema={
                    "type": "object",
                    "properties": {"answer": {"type": "integer"}},
                    "required": ["answer"],
                    "additionalProperties": False,
                },
                model="fake",
            )
        )
    )
    ```

See Also:
    - [typevet.adapters.outbound.generation_finite][]: Non-finite float guard
    - [typevet.adapters.outbound.fake][]: Sync fake
    - [typevet.adapters.outbound.async_llama_cpp][]: Live llama.cpp adapter
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import jsonschema

from typevet.adapters.outbound.generation_finite import reject_non_finite_numbers
from typevet.domain.errors import GenerationError, SchemaValidationError
from typevet.domain.models import GenerationRequest, GenerationResult

AsyncResponder = Callable[[GenerationRequest], Mapping[str, Any]]


class AsyncFakeGenerationAdapter:
    """Return a fixed or callable mapping and validate it against the schema.

    Attributes:
        _value (Mapping[str, Any] | None): Fixed mapping when set.
        _responder (AsyncResponder | None): Callable producer when set.
        _fail (Exception | None): Exception raised instead of a value.

    Examples:
        ```python
        from typevet.adapters.outbound.async_fake import AsyncFakeGenerationAdapter

        AsyncFakeGenerationAdapter(value={"answer": 1})
        ```
    """

    def __init__(
        self,
        value: Mapping[str, Any] | None = None,
        *,
        responder: AsyncResponder | None = None,
        fail: Exception | None = None,
    ) -> None:
        """Configure the fake.

        Args:
            value: Fixed mapping returned for every call (validated).
            responder: Optional callable that builds the mapping from the request.
            fail: When set, ``generate`` raises this exception instead.

        Raises:
            ValueError: When value, responder and fail are all omitted.
        """
        if value is None and responder is None and fail is None:
            msg = "provide value, responder, or fail"
            raise ValueError(msg)
        self._value = value
        self._responder = responder
        self._fail = fail

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """Return a validated mapping or raise the configured failure.

        Args:
            request: Typed generation request.

        Returns:
            GenerationResult with the fake value.

        Raises:
            Exception: The configured ``fail`` value when set.
            GenerationError: When the fake has no value source.
            SchemaValidationError: When the fake value is non-finite or fails schema.
        """
        if self._fail is not None:
            raise self._fail
        if self._responder is not None:
            raw: Mapping[str, Any] = self._responder(request)
        elif self._value is not None:
            raw = self._value
        else:
            msg = "fake has no value"
            raise GenerationError(msg)

        schema_obj = dict(request.schema)
        instance = dict(raw)
        reject_non_finite_numbers(instance)
        try:
            jsonschema.validate(instance=instance, schema=schema_obj)
        except jsonschema.ValidationError as exc:
            raise SchemaValidationError(
                f"fake output failed schema: {exc.message}",
                payload=raw,
            ) from exc
        return GenerationResult(value=instance, model=request.model, raw_text=None)

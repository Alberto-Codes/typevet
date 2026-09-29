"""Request and result values for typed generation.

Examples:
    ```python
    from typevet.domain.models import GenerationRequest, GenerationResult

    req = GenerationRequest(
        prompt="hi",
        schema={"type": "object", "additionalProperties": False},
        model="m",
    )
    result = GenerationResult(value={"ok": True}, model=req.model)
    assert result.model == "m"
    ```

See Also:
    - [typevet.domain.errors][]: Failures raised around these values
    - [typevet.domain.media][]: ImageInput and the media marker
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from typevet.domain.media import MEDIA_MARKER, ImageInput, count_media_markers


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    """One typed generation ask.

    Attributes:
        prompt (str): Natural-language instruction for the model.
        schema (Mapping[str, Any]): JSON Schema object as a mapping.
        model (str): Backend model id or alias.
        media (tuple[ImageInput, ...]): Images the prompt marks, one
            ``MEDIA_MARKER`` each, in marker order. Empty for a text ask.

    Examples:
        ```python
        from typevet.domain.models import GenerationRequest

        GenerationRequest(
            prompt="Return JSON.",
            schema={"type": "object", "additionalProperties": False},
            model="fake",
        )
        ```
    """

    prompt: str
    schema: Mapping[str, Any]
    model: str
    media: tuple[ImageInput, ...] = ()

    def __post_init__(self) -> None:
        """Reject a blank field, a non-object schema root or a marker mismatch.

        Stores ``media`` as a tuple, so a list input becomes immutable.

        Raises:
            ValueError: When prompt or model is blank, schema type is not
                object, or the media marker count differs from ``len(media)``.
            TypeError: When schema is not a mapping, or a media item is not
                an ``ImageInput``.
        """
        if not self.prompt.strip():
            msg = "prompt must be non-empty"
            raise ValueError(msg)
        if not self.model.strip():
            msg = "model must be non-empty"
            raise ValueError(msg)
        if not isinstance(self.schema, Mapping):
            msg = "schema must be a mapping"
            raise TypeError(msg)
        schema_type = self.schema.get("type")
        if schema_type is not None and schema_type != "object":
            msg = "schema root type must be object when set"
            raise ValueError(msg)
        object.__setattr__(self, "media", tuple(self.media))
        for index, item in enumerate(self.media):
            if not isinstance(item, ImageInput):
                msg = f"media[{index}] must be ImageInput, got {type(item).__name__}"
                raise TypeError(msg)
        markers = count_media_markers(self.prompt)
        if markers != len(self.media):
            msg = (
                f"prompt holds {markers} {MEDIA_MARKER} media marker(s) "
                f"but media has {len(self.media)} image(s)"
            )
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """A value that validated against the request schema.

    Attributes:
        value (Mapping[str, Any]): Validated JSON-compatible mapping.
        model (str): Model id that produced the value.
        raw_text (str | None): Optional raw model text before parse.

    Examples:
        ```python
        from typevet.domain.models import GenerationResult

        GenerationResult(value={"ok": True}, model="fake")
        ```
    """

    value: Mapping[str, Any]
    model: str
    raw_text: str | None = None

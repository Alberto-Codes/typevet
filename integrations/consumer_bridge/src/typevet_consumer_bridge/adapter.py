"""Synchronous consumer port composed through the public runtime factory.

See Also:
    - [typevet_consumer_bridge.settings][]: Validated connection settings.


Examples:
    ```python
    from typevet_consumer_bridge import BridgeSettings

    settings = BridgeSettings("http://localhost:8080", 30.0, "served-model")
    ```
"""

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from json import JSONDecodeError
from types import TracebackType
from typing import Self

import httpx
from judgevet import SystemOneResponse

from typevet.domain import (
    BackendHttpError,
    GemmaTemplateError,
    GenerationError,
    JudgmentValidationError,
    SchemaValidationError,
    ScoringUnsupportedCapabilityError,
    ScoringValidationError,
    TransportError,
)
from typevet.runtime import GemmaNativeVisionSession, open_gemma_native_vision_judgment
from typevet_consumer_bridge.errors import (
    BridgeCapabilityError,
    BridgeRequestError,
    BridgeResponseError,
    BridgeTransportError,
    BridgeUnavailableError,
)
from typevet_consumer_bridge.questions import convert_questions, snapshot_state
from typevet_consumer_bridge.responses import convert_response
from typevet_consumer_bridge.settings import BridgeSettings


class TypevetSystemOneAdapter:
    """Borrow a runtime session and expose the consumer synchronous port.

    The adapter never closes its borrowed session or client.

    Attributes:
        _session (GemmaNativeVisionSession): Borrowed public runtime session.
        _closed (bool): Whether access through this adapter has ended.

    Examples:
        ```python
        adapter = TypevetSystemOneAdapter(session)
        adapter.close()
        ```
    """

    def __init__(self, session: GemmaNativeVisionSession) -> None:
        """Retain a borrowed, already open session."""
        self._session = session
        self._closed = False

    def system_one(
        self, state: object, questions: object, model: str
    ) -> SystemOneResponse:
        """Validate a whole request and return consumer-owned answers.

        Args:
            state: Text or JSON object/array.
            questions: Consumer questions or supported raw mappings.
            model: Required identity, exactly equal to the session model.

        Returns:
            New consumer response with preserved runtime values.

        Raises:
            BridgeUnavailableError: If this adapter is closed.
            BridgeRequestError: If input or engine judgment validation fails.
            BridgeCapabilityError: If the engine lacks a required capability.
            BridgeTransportError: If a known transport operation fails.
            BridgeResponseError: If output validation fails.
        """
        if self._closed:
            raise BridgeUnavailableError("Bridge adapter is closed.")
        if not isinstance(model, str) or model != self._session.model:
            raise BridgeRequestError("Request model does not match the session.")
        snapshot = snapshot_state(state)
        converted = convert_questions(questions)
        try:
            response = self._session.port.judge(snapshot, converted, model)
        except JudgmentValidationError:
            raise BridgeRequestError("Runtime rejected the request.") from None
        except (ScoringUnsupportedCapabilityError, GemmaTemplateError):
            raise BridgeCapabilityError("Runtime capability is unavailable.") from None
        except (ScoringValidationError, SchemaValidationError):
            raise BridgeResponseError("Invalid runtime response.") from None
        except (TransportError, BackendHttpError, httpx.HTTPError):
            raise BridgeTransportError("Runtime transport failed.") from None
        return convert_response(response, converted, self._session.model)

    def close(self) -> None:
        """Invalidate this adapter without closing borrowed resources."""
        self._closed = True

    def __enter__(self) -> Self:
        """Return this adapter for a borrowed context.

        Returns:
            This adapter instance.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close this adapter without suppressing a caller exception."""
        self.close()


_FACTORY_MODULE = "typevet.adapters.outbound.gemma_native_vision_factory"


def _error_frames(error: Exception) -> list[tuple[str, str]]:
    frames = []
    traceback = error.__traceback__
    while traceback is not None:
        frame = traceback.tb_frame
        frames.append((frame.f_globals.get("__name__", ""), frame.f_code.co_name))
        traceback = traceback.tb_next
    return frames


def _metadata_failure(error: Exception) -> bool:
    # These origins belong to the hash-pinned engine artifact. Never infer a
    # metadata fault merely because a caller callback raised the same class.
    frames = _error_frames(error)
    if isinstance(error, JSONDecodeError):
        while frames and frames[-1][0] in {"json", "json.decoder"}:
            frames.pop()
        return frames[-2:] == [
            (_FACTORY_MODULE, "_classify_native_template"),
            ("httpx._models", "json"),
        ]
    if isinstance(error, (KeyError, TypeError)):
        return frames[-1:] in (
            [(_FACTORY_MODULE, "_classify_native_template")],
            [
                (
                    "typevet.adapters.outbound.gemma.served_template",
                    "classify_served_template",
                )
            ],
        )
    return type(error) is GenerationError and frames[-1:] in (
        [("typevet.adapters.outbound.llama_cpp_multimodal", "_capability_from_props")],
        [("typevet.adapters.outbound.llama_cpp_http", "parse_json_response")],
    )


def _enter_runtime(
    stack: ExitStack,
    settings: BridgeSettings,
    model: str,
    http_client: httpx.Client | None,
) -> GemmaNativeVisionSession:
    try:
        return stack.enter_context(
            open_gemma_native_vision_judgment(
                settings=settings, model=model, http_client=http_client
            )
        )
    except (TransportError, BackendHttpError, httpx.HTTPError):
        raise BridgeTransportError("Runtime transport failed.") from None
    except (JSONDecodeError, KeyError, TypeError, GenerationError) as error:
        if not _metadata_failure(error):
            raise
        raise BridgeResponseError("Invalid runtime metadata.") from None
    except ValueError as error:
        if _error_frames(error)[-1:] not in (
            [(_FACTORY_MODULE, "_session")],
            [(_FACTORY_MODULE, "_classify_native_template")],
        ) or not str(error).startswith(
            (
                "expected NATIVE_GEMMA4_TURN, got ",
                "unsupported served template for native vision: ",
                "model reports text-only input modalities",
            )
        ):
            raise
        raise BridgeCapabilityError("Runtime capability is unavailable.") from None


@contextmanager
def open_typevet_system_one(
    *,
    settings: BridgeSettings,
    model: str | None = None,
    http_client: httpx.Client | None = None,
) -> Iterator[TypevetSystemOneAdapter]:
    """Own one runtime context while preserving a supplied client's ownership.

    Args:
        settings: Locally validated connection settings.
        model: Explicit identity or the settings identity when omitted.
        http_client: Optional caller-owned client.

    Yields:
        A text adapter invalidated on every exit path.

    Raises:
        BridgeRequestError: If settings or the selected model is invalid.
    """
    if not isinstance(settings, BridgeSettings):
        raise BridgeRequestError("Invalid bridge settings.")
    selected = settings.multimodal_model if model is None else model
    if not isinstance(selected, str) or not selected.strip():
        raise BridgeRequestError("Invalid request model.")
    with ExitStack() as stack:
        session = _enter_runtime(stack, settings, selected, http_client)
        adapter = TypevetSystemOneAdapter(session)
        try:
            yield adapter
        finally:
            adapter.close()

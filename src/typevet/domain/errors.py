"""Domain failures for typed generation.

Examples:
    ```python
    from typevet.domain.errors import SchemaValidationError

    err = SchemaValidationError("bad field", payload={"x": 1})
    assert err.payload == {"x": 1}
    ```

See Also:
    - [typevet.domain.models][]: Request and result types
    - [typevet.adapters.outbound.llama_cpp.http_mapping][]: httpx to domain error mapping

Attributes:
    BackendHttpError (type): Backend HTTP status 400 or above, or a redirect,
        with the ``Retry-After`` wait and rate-limit headers when present.
    GenerationError (type): Base failure for a generation call.
    GenerationUnsupportedCapabilityError (type): Backend cannot honor the ask.
    JudgmentError (type): Base failure for a judgment call.
    JudgmentValidationError (type): Answer failed judgment shape rules.
    CalibrationMapError (type): Calibration map is malformed or refused.
    CalibrationDigestError (type): Map file digest is missing, malformed or wrong.
    CalibrationTaskMismatchError (type): Map was fitted for another task.
    CalibrationModelMismatchError (type): Map was fitted for another model or backend.
    CalibrationTargetError (type): Map targets a Choice or Score question.
    GemmaTemplateError (type): Gemma served-template or answer-prefix violation.
    DecisionExecutionError (type): Categorical decision execute rejected inputs.
    ScoringError (type): Base failure for a candidate scoring call.
    ScoringValidationError (type): Score coverage or value failed validation.
    ScoringUnsupportedCapabilityError (type): Backend cannot honor the stage.
    SchemaValidationError (type): Output failed the requested schema.
    TransportError (type): HTTP client failure before a response.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType


class GenerationError(Exception):
    """A typed generation call failed before a valid result existed.

    Examples:
        ```python
        from typevet.domain.errors import GenerationError

        raise GenerationError("transport failed")
        ```
    """


class TransportError(GenerationError):
    """The HTTP client failed before a usable llama.cpp response arrived.

    Attributes:
        status_code (None): Always ``None`` for transport failures.
        body_snippet (None): Always ``None`` when no response body was read.

    Examples:
        ```python
        from typevet.domain.errors import TransportError

        raise TransportError("llama.cpp request failed: connection refused")
        ```
    """

    status_code: None
    body_snippet: None

    def __init__(self, message: str) -> None:
        """Record a transport failure message.

        Args:
            message: Human-readable summary of the client failure.
        """
        super().__init__(message)
        self.status_code = None
        self.body_snippet = None


class BackendHttpError(GenerationError):
    """The backend returned an HTTP error status (400 or above) or a redirect.

    The vLLM clients that ``backend_settings`` builds do not follow redirects;
    a 3xx raises this error with the 3xx status and no ``Location`` value.

    Attributes:
        status_code (int): HTTP status from the router or the gateway.
        body_snippet (str): Truncated response body text for diagnostics.
            Empty when the body is HTML, which a gateway error page often is.
        retry_after_seconds (float | None): Wait that ``Retry-After`` asks
            for, in seconds. ``None`` when the header is absent or invalid,
            and always ``None`` for llama.cpp. typevet does not retry.
        rate_limit (Mapping[str, str]): Read-only ``x-ratelimit-*`` and
            ``ratelimit-*`` response headers, with lowercase names and
            verbatim values. Empty when there are none, and for llama.cpp.

    Examples:
        ```python
        from typevet.domain.errors import BackendHttpError

        raise BackendHttpError(
            "llama.cpp HTTP 500: internal",
            status_code=500,
            body_snippet="internal",
        )
        ```
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        body_snippet: str,
        retry_after_seconds: float | None = None,
        rate_limit: Mapping[str, str] | None = None,
    ) -> None:
        """Record an HTTP error status, response snippet and retry hints.

        Args:
            message: Human-readable summary including status and snippet.
            status_code: HTTP status from the backend or the gateway.
            body_snippet: Truncated response body text.
            retry_after_seconds: Parsed ``Retry-After`` wait, or ``None``.
            rate_limit: Rate-limit response headers; stored as a read-only copy.
        """
        super().__init__(message)
        self.status_code = status_code
        self.body_snippet = body_snippet
        self.retry_after_seconds = retry_after_seconds
        self.rate_limit: Mapping[str, str] = MappingProxyType(dict(rate_limit or {}))


class JudgmentError(GenerationError):
    """A typed judgment call failed before a valid response existed.

    Examples:
        ```python
        from typevet.domain.errors import JudgmentError

        raise JudgmentError("judgment failed")
        ```
    """


class JudgmentValidationError(JudgmentError):
    """An answer or option list failed judgment validation rules.

    Examples:
        ```python
        from typevet.domain.errors import JudgmentValidationError

        raise JudgmentValidationError("choice not in criteria")
        ```
    """


class CalibrationMapError(JudgmentError):
    """A calibration map is malformed or cannot apply to this judgment.

    Messages name fields and rules only. They never carry a probability,
    a digest, a task id, a model id or a backend name.

    Examples:
        ```python
        from typevet.domain.errors import CalibrationMapError

        raise CalibrationMapError("calibration map field 'method' is invalid")
        ```
    """


class CalibrationDigestError(CalibrationMapError):
    """The sha256 of the map file is missing, malformed or does not match.

    Examples:
        ```python
        from typevet.domain.errors import CalibrationDigestError

        raise CalibrationDigestError("calibration map digest does not match")
        ```
    """


class CalibrationTaskMismatchError(CalibrationMapError):
    """The map was fitted for a task other than the wrapper task id.

    Examples:
        ```python
        from typevet.domain.errors import CalibrationTaskMismatchError

        raise CalibrationTaskMismatchError("calibration map task does not match")
        ```
    """


class CalibrationModelMismatchError(CalibrationMapError):
    """The map was fitted on another model or another serving backend.

    Examples:
        ```python
        from typevet.domain.errors import CalibrationModelMismatchError

        raise CalibrationModelMismatchError("calibration map model does not match")
        ```
    """


class CalibrationTargetError(CalibrationMapError):
    """A map names a question that is not a Noul question.

    Examples:
        ```python
        from typevet.domain.errors import CalibrationTargetError

        raise CalibrationTargetError("calibration maps apply to Noul questions only")
        ```
    """


class GemmaTemplateError(JudgmentError):
    """Gemma served-template or answer-prefix rules were violated.

    Examples:
        ```python
        from typevet.domain.errors import GemmaTemplateError

        raise GemmaTemplateError("unsupported served template family")
        ```
    """


class DecisionExecutionError(GenerationError):
    """A categorical decision could not be executed in-domain.

    Examples:
        ```python
        from typevet.domain.errors import DecisionExecutionError

        raise DecisionExecutionError("nullable categorical decisions are unsupported")
        ```
    """


class GenerationUnsupportedCapabilityError(GenerationError):
    """The generation backend cannot honor a part of the request.

    An adapter raises it before any HTTP call, for example when a request
    carries images the backend adapter cannot send. The adapter never drops
    the unsupported part silently.

    Examples:
        ```python
        from typevet.domain.errors import GenerationUnsupportedCapabilityError

        raise GenerationUnsupportedCapabilityError("images are not supported")
        ```
    """


class ScoringError(GenerationError):
    """A candidate scoring call failed before a valid result existed.

    Examples:
        ```python
        from typevet.domain.errors import ScoringError

        raise ScoringError("scoring failed")
        ```
    """


class ScoringValidationError(ScoringError):
    """Scores failed fail-closed coverage or finiteness rules.

    Examples:
        ```python
        from typevet.domain.errors import ScoringValidationError

        raise ScoringValidationError("missing scores for requested candidates")
        ```
    """


class ScoringUnsupportedCapabilityError(ScoringError):
    """The backend cannot honor the requested score stage or capability.

    Examples:
        ```python
        from typevet.domain.errors import ScoringUnsupportedCapabilityError

        raise ScoringUnsupportedCapabilityError("unsupported score stage")
        ```
    """


class SchemaValidationError(GenerationError):
    """The model output did not validate against the requested schema.

    Attributes:
        payload (object | None): Parsed value that failed validation, when set.
        args (tuple): Standard exception args (message first).

    Examples:
        ```python
        from typevet.domain.errors import SchemaValidationError

        raise SchemaValidationError("missing key", payload={})
        ```
    """

    def __init__(self, message: str, *, payload: object | None = None) -> None:
        """Record the failure and the optional rejected payload.

        Args:
            message: Human-readable validation summary.
            payload: Parsed value that failed validation, when available.
        """
        super().__init__(message)
        self.payload = payload

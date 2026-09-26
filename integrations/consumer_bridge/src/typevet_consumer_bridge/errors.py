"""Safe bridge errors derived from the consumer error base.

See Also:
    - [typevet_consumer_bridge.adapter][]: Named exception boundaries.


Examples:
    ```python
    from typevet_consumer_bridge import BridgeSettings

    settings = BridgeSettings("http://localhost:8080", 30.0, "served-model")
    ```
"""

from judgevet import JudgevetError


class BridgeError(JudgevetError):
    """Base error for the private bridge.

    Examples:
        ```python
        error = BridgeError("Safe message.")
        ```
    """


class BridgeRequestError(BridgeError):
    """The caller request is outside the supported subset.

    Examples:
        ```python
        error = BridgeRequestError("Safe message.")
        ```
    """


class BridgeCapabilityError(BridgeError):
    """The runtime cannot serve this text judgment.

    Examples:
        ```python
        error = BridgeCapabilityError("Safe message.")
        ```
    """


class BridgeTransportError(BridgeError):
    """A known runtime transport operation failed.

    Examples:
        ```python
        error = BridgeTransportError("Safe message.")
        ```
    """


class BridgeResponseError(BridgeError):
    """Runtime metadata or answers violate the bridge contract.

    Examples:
        ```python
        error = BridgeResponseError("Safe message.")
        ```
    """


class BridgeUnavailableError(BridgeError):
    """The adapter is closed.

    Examples:
        ```python
        error = BridgeUnavailableError("Safe message.")
        ```
    """

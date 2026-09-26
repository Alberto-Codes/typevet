"""Private synchronous bridge from consumer types to the public Typevet runtime.

Attributes:
    __all__ (list[str]): Public settings, adapter, context manager and errors.

See Also:
    - [typevet_consumer_bridge.adapter][]: Borrowed and owned session lifetimes.
    - [typevet_consumer_bridge.settings][]: Connection settings.


Examples:
    ```python
    from typevet_consumer_bridge import BridgeSettings

    settings = BridgeSettings("http://localhost:8080", 30.0, "served-model")
    ```
"""

from typevet_consumer_bridge.adapter import (
    TypevetSystemOneAdapter,
    open_typevet_system_one,
)
from typevet_consumer_bridge.errors import (
    BridgeCapabilityError,
    BridgeError,
    BridgeRequestError,
    BridgeResponseError,
    BridgeTransportError,
    BridgeUnavailableError,
)
from typevet_consumer_bridge.settings import BridgeSettings

__all__ = [
    "BridgeCapabilityError",
    "BridgeError",
    "BridgeRequestError",
    "BridgeResponseError",
    "BridgeSettings",
    "BridgeTransportError",
    "BridgeUnavailableError",
    "TypevetSystemOneAdapter",
    "open_typevet_system_one",
]

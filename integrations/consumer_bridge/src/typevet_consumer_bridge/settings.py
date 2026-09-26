"""Validated connection settings for the private text bridge.

See Also:
    - [typevet_consumer_bridge.adapter][]: Runtime composition.


Examples:
    ```python
    from typevet_consumer_bridge import BridgeSettings

    settings = BridgeSettings("http://localhost:8080", 30.0, "served-model")
    ```
"""

from dataclasses import dataclass
from math import isfinite

from typevet_consumer_bridge.errors import BridgeRequestError


@dataclass(frozen=True)
class BridgeSettings:
    """Connection values passed to the public runtime factory.

    Attributes:
        base_url (str): Nonblank runtime URL.
        timeout (float): Finite positive timeout in seconds.
        multimodal_model (str): Nonblank requested model identity.

    Examples:
        ```python
        settings = BridgeSettings("http://localhost:8080", 30.0, "served-model")
        ```
    """

    base_url: str
    timeout: float
    multimodal_model: str

    def __post_init__(self) -> None:
        """Validate settings before any runtime operation.

        Raises:
            BridgeRequestError: If a setting has an invalid value.
        """
        strings = (self.base_url, self.multimodal_model)
        if any(not isinstance(value, str) or not value.strip() for value in strings):
            raise BridgeRequestError("Invalid bridge settings.")
        if (
            isinstance(self.timeout, bool)
            or not isinstance(self.timeout, (int, float))
            or not isfinite(self.timeout)
            or self.timeout <= 0
        ):
            raise BridgeRequestError("Invalid bridge settings.")

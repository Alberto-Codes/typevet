"""Read one calibration map file and check its digest (#352).

The caller supplies the sha256 of the file bytes. The reader refuses the file
before it parses the JSON when the digest is missing, malformed or different.
``fitted_on.receipt_sha256`` inside the map is for traceability only; this
reader does not check it.

Examples:
    ```python
    from typevet.adapters.inbound import load_calibration_map

    cmap = load_calibration_map("maps/fraud.json", sha256=digest)
    ```

See Also:
    - [typevet.domain.calibration][]: Map types and the validator
    - [typevet.runtime.calibrated_judgment][]: The judgment port wrapper
"""

from __future__ import annotations

import hashlib
import hmac
import json
from os import PathLike
from pathlib import Path

from typevet.domain.calibration import (
    CalibrationMap,
    calibration_map_from_mapping,
    normalize_sha256,
)
from typevet.domain.errors import CalibrationDigestError, CalibrationMapError


def load_calibration_map(path: str | PathLike[str], *, sha256: str) -> CalibrationMap:
    """Read, verify and validate one calibration map file.

    Args:
        path: The map file.
        sha256: The expected sha256 hex digest of the file bytes.

    Returns:
        The validated map, with ``sha256`` set to the verified digest.

    Raises:
        CalibrationDigestError: ``sha256`` is missing or malformed, or the
            file bytes have a different digest.
        CalibrationMapError: The file is not UTF-8 JSON, or the document
            fails validation.
        OSError: The file cannot be read.
    """
    expected = normalize_sha256(sha256)
    raw = Path(path).read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if not hmac.compare_digest(actual, expected):
        msg = "calibration map file digest does not match the supplied sha256"
        raise CalibrationDigestError(msg)
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise CalibrationMapError("calibration map file is not UTF-8 JSON") from None
    return calibration_map_from_mapping(document, sha256=expected)

"""Calibrate: map a raw Noul probability through a Platt calibration map.

Scenario: a fraud screen asks one typed Noul whether a short message is a fraud
attempt. A calibration map, fitted for this task, changes the raw probability
into a calibrated probability. ``CalibratedJudgment`` wraps the session port and
applies the map. The script prints the raw and the calibrated probability, then
the winner on the last line: ``yes`` when the calibrated probability is above
0.5, otherwise ``no``.

Run with: `uv run python examples/calibrate/run.py`

Run the command from the repository root, because the map comes from
``tests/fixtures/calibration/platt_noul_map.json``. ``TYPEVET_BACKEND`` selects
the backend (default ``llama_cpp``). The wrapper refuses a map whose task,
backend or model differs from the call. This example reads the task, backend
and model from the fixture map, so the checks pass by construction. In your own
code pass your task id, ``load_backend()`` and ``session.model``. The fixture
map is synthetic and names the fake model, so run it with
``TYPEVET_BACKEND=fake``. For a live run, use a map fitted on your own model and
backend.

Examples:
    ```bash
    TYPEVET_BACKEND=fake uv run python examples/calibrate/run.py
    ```

See Also:
    - [typevet.runtime.CalibratedJudgment][]: Port wrapper that applies the map.
    - [typevet.adapters.inbound.load_calibration_map][]: The map file reader.
    - examples/README.md: Every example and its command.
"""

from __future__ import annotations

from pathlib import Path

from typevet.adapters.inbound import load_calibration_map, open_judgment
from typevet.domain import CalibrationMap, Noul
from typevet.runtime import CalibratedJudgment

# An edited map file has another digest and raises CalibrationDigestError.
MAP_SHA256 = "78e8d6ad792402e72c4178d7d331a5d209bb4d8839b86b1530934a70d222cb6a"
THRESHOLD = 0.5
MESSAGE = "Your account is locked. Reply with your card PIN to unlock it today."
FRAUD = Noul(
    instructions="Is this message a fraud attempt?",
    criteria={"true": "The message tries to deceive", "false": "The message is honest"},
)


def load_fraud_map() -> CalibrationMap:
    """Read the Platt fixture map for the fraud task. Repo-only: needs a checkout.

    Replace this function with your own loader in another project. Pin the
    sha256 digest of your own map file as a literal, as ``MAP_SHA256`` does.

    Returns:
        The validated calibration map.
    """
    path = Path("tests/fixtures/calibration/platt_noul_map.json")
    return load_calibration_map(path, sha256=MAP_SHA256)


def main() -> None:
    """Judge one message, calibrate the Noul answer and print both values."""
    # 1. Load the message and the calibration map.
    cmap = load_fraud_map()
    print(f"message: {MESSAGE}")

    # 2. Open the judgment session that TYPEVET_BACKEND selects.
    with open_judgment() as session:
        # 3. Wrap the session port with the map, then ask the Noul. This
        #    example reads the task, backend and model from the fixture map,
        #    so the checks pass by construction. In your own code pass your
        #    task id, load_backend() and session.model.
        fitted = cmap.fitted_on
        port = CalibratedJudgment(
            session.port,
            {"fraud": cmap},
            task_id=fitted.task_id,
            backend=fitted.backend,
        )
        response = port.judge(MESSAGE, {"fraud": FRAUD}, fitted.model)

    # 4. Print the raw and calibrated probabilities, then the winner last.
    record = response.calibration["fraud"]
    print(f"raw: {record.raw:.4f}")
    print(f"calibrated: {record.calibrated:.4f}")
    print(f"winner: {'yes' if record.calibrated > THRESHOLD else 'no'}")


if __name__ == "__main__":
    main()

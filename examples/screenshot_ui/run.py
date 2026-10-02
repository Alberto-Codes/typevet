"""Screenshot UI: which website does a browser screenshot show?

Scenario: a computer-use agent takes a screenshot of the browser. One typed
Choice asks which website chrome (logo, header and navigation) the screenshot
shows. The answer is ``fox_news``, ``home_depot``, ``squarespace`` or ``other``.
The script prints each option with its probability, then the winner on the last
line.

Run with: `uv run python examples/screenshot_ui/run.py`

Run the command from the repository root, because the screenshot comes from
``tests/fixtures/psai/vision_smoke/``. The loader checks the SHA-256 digest
that the manifest records for the PNG file. ``TYPEVET_BACKEND`` selects the
backend (default ``llama_cpp``). For a live llama.cpp run, set
``TYPEVET_LLAMA__MULTIMODAL_MODEL`` to a Gemma 4 vision model id.
``TYPEVET_BACKEND=fake`` runs offline.

Examples:
    ```bash
    TYPEVET_BACKEND=fake uv run python examples/screenshot_ui/run.py
    ```

See Also:
    - [typevet.adapters.inbound.open_judgment][]: Session that the backend selects.
    - examples/README.md: Every example and its command.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from typevet.adapters.inbound import open_judgment
from typevet.domain import Choice, ImageInput

STATE = "This is a screenshot of a web browser."
SITE = Choice(
    instructions="Which website chrome (logo, header and navigation) does it show?",
    criteria={
        "fox_news": "The Fox News website",
        "home_depot": "The Home Depot website",
        "squarespace": "The Squarespace website",
        "other": "Another website, or no website chrome is visible",
    },
)


def load_screenshot(unique_data_id: str = "cmcc8u6yc00va1p1ydsdu52zy") -> ImageInput:
    """Read one PSAI screenshot and check its digest. Repo-only: needs a checkout.

    Replace this function with your own loader in another project.

    Args:
        unique_data_id: Row id in the vision smoke manifest.

    Returns:
        The screenshot image.

    Raises:
        ValueError: When the PNG bytes do not match the manifest sha256.
    """
    psai = Path("tests/fixtures/psai/vision_smoke")
    manifest = json.loads((psai / "manifest.json").read_text())
    row = next(r for r in manifest["rows"] if r["unique_data_id"] == unique_data_id)
    screenshot = row["screenshot"]
    data = (psai / screenshot["file_name"]).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != screenshot["sha256"]:
        msg = f"{screenshot['file_name']} sha256 {digest} != manifest value"
        raise ValueError(msg)
    return ImageInput(data=data, mime_type=screenshot["mime_type"])


def main() -> None:
    """Ask which website the screenshot shows and print the answer."""
    # 1. Load the screenshot.
    image = load_screenshot()
    print(f"state: {STATE}")

    # 2. Open the judgment session that TYPEVET_BACKEND selects.
    with open_judgment() as session:
        # 3. Ask the Choice about the state, conditioned on the screenshot.
        response = session.port.judge(
            STATE, {"site": SITE}, session.model, media=(image,)
        )

    # 4. Print every option with its probability, then the winner last.
    answer = response.choices["site"]
    for label in SITE.criteria:
        print(f"{label}: {answer.probabilities[label]:.4f}")
    print(f"winner: {answer.choice}")


if __name__ == "__main__":
    main()

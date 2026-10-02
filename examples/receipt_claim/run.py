"""Receipt claim: does a receipt image support an expense claim?

Scenario: an employee claims an expense and attaches a photo of the receipt.
One typed Choice asks whether the image supports the claimed total. The answer
is ``supported``, ``contradicted`` or ``insufficient_evidence``. The script
prints each option with its probability, then the winner on the last line.

Run with: `uv run python examples/receipt_claim/run.py`

Run the command from the repository root, because the receipt comes from
``tests/fixtures/cord/expense_smoke/``. ``TYPEVET_BACKEND`` selects the backend
(default ``llama_cpp``). For a live llama.cpp run, set
``TYPEVET_LLAMA__MULTIMODAL_MODEL`` to a Gemma 4 vision model id.
``TYPEVET_BACKEND=fake`` runs offline.

Examples:
    ```bash
    TYPEVET_BACKEND=fake uv run python examples/receipt_claim/run.py
    ```

See Also:
    - [typevet.adapters.inbound.open_judgment][]: Session that the backend selects.
    - examples/README.md: Every example and its command.
"""

from __future__ import annotations

import json
from pathlib import Path

from typevet.adapters.inbound import open_judgment
from typevet.domain import Choice, ImageInput

VERDICT = Choice(
    instructions="Look at the receipt image. Does it support the expense claim?",
    criteria={
        "supported": "The receipt image shows this total",
        "contradicted": "The receipt image shows a different total",
        "insufficient_evidence": "The image does not show enough to decide",
    },
)


def load_receipt_case(receipt_id: str = "R01") -> tuple[str, ImageInput]:
    """Read one CORD receipt and its supported claim. Repo-only: needs a checkout.

    Replace this function with your own loader in another project.

    Args:
        receipt_id: Receipt id in the expense smoke manifest.

    Returns:
        The claim statement and the receipt image.
    """
    cord = Path("tests/fixtures/cord/expense_smoke")
    manifest = json.loads((cord / "manifest.json").read_text())
    receipt = next(r for r in manifest["receipts"] if r["receipt_id"] == receipt_id)
    claim = next(c for c in receipt["claims"] if c["expected_verdict"] == "supported")
    image = ImageInput(
        data=(cord / receipt["image"]["file_name"]).read_bytes(),
        mime_type=receipt["image"]["mime_type"],
    )
    return claim["statement"], image


def main() -> None:
    """Judge one expense claim against its receipt image and print the answer."""
    # 1. Load the claim text and the receipt image.
    claim, image = load_receipt_case()
    print(f"claim: {claim}")

    # 2. Open the judgment session that TYPEVET_BACKEND selects.
    with open_judgment() as session:
        # 3. Ask the Choice about the claim, conditioned on the image.
        response = session.port.judge(
            claim, {"verdict": VERDICT}, session.model, media=(image,)
        )

    # 4. Print every option with its probability, then the winner last.
    answer = response.choices["verdict"]
    for label in VERDICT.criteria:
        print(f"{label}: {answer.probabilities[label]:.4f}")
    print(f"winner: {answer.choice}")


if __name__ == "__main__":
    main()

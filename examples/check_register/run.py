"""Check register: does a cheque image match its register row?

Scenario: a bookkeeper compares a cheque image against one row of the cheque
register. One typed Choice asks whether the cheque shows the payee and the
amount of that row. The answer is ``matches``, ``differs`` or ``unreadable``.
The script prints each option with its probability, then the winner on the
last line.

Run with: `uv run python examples/check_register/run.py`

Run the command from the repository root, because the cheque comes from the
``typevet_evals`` workspace member. The cheque is a synthetic SPECIMEN with an
invented payee. ``TYPEVET_BACKEND`` selects the backend (default
``llama_cpp``). For a live llama.cpp run, set
``TYPEVET_LLAMA__MULTIMODAL_MODEL`` to a Gemma 4 vision model id.
``TYPEVET_BACKEND=fake`` runs offline.

Examples:
    ```bash
    TYPEVET_BACKEND=fake uv run python examples/check_register/run.py
    ```

See Also:
    - [typevet.adapters.inbound.open_judgment][]: Session that the backend selects.
    - examples/README.md: Every example and its command.
"""

from __future__ import annotations

from typevet.adapters.inbound import open_judgment
from typevet.domain import Choice, ImageInput

# Repo-only: the loader below draws the cheque from the evals workspace member.
from typevet_evals.check_match.cases import check_cases
from typevet_evals.check_match.render import render_check

VERDICT = Choice(
    instructions=(
        "Look at the cheque image. Does it show the payee and the amount of "
        "the register row?"
    ),
    criteria={
        "matches": "The cheque shows the same payee and the same amount",
        "differs": "The cheque shows a different payee or a different amount",
        "unreadable": "The image is not clear enough to decide",
    },
)


def load_check_case(seed: int = 0) -> tuple[str, ImageInput]:
    """Render one SPECIMEN cheque and its register row. Repo-only: needs a checkout.

    Replace this function with your own loader in another project.

    Args:
        seed: Seed of the synthetic register.

    Returns:
        The register row text (payee and amount) and the cheque image.
    """
    case = check_cases(seed=seed, count=1)[0]
    whole, cents = divmod(case.row.amount_cents, 100)
    row = f"Register row: payee {case.row.payee}; amount ${whole:,}.{cents:02d}"
    return row, ImageInput(data=render_check(case), mime_type="image/png")


def main() -> None:
    """Judge one cheque image against its register row and print the answer."""
    # 1. Load the register row text and the cheque image.
    row, image = load_check_case()
    print(row)

    # 2. Open the judgment session that TYPEVET_BACKEND selects.
    with open_judgment() as session:
        # 3. Ask the Choice about the register row, conditioned on the image.
        response = session.port.judge(
            row, {"verdict": VERDICT}, session.model, media=(image,)
        )

    # 4. Print every option with its probability, then the winner last.
    answer = response.choices["verdict"]
    for label in VERDICT.criteria:
        print(f"{label}: {answer.probabilities[label]:.4f}")
    print(f"winner: {answer.choice}")


if __name__ == "__main__":
    main()

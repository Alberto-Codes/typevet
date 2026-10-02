"""Scam message: is a text message a scam?

Scenario: a phone user receives a text message. One typed Noul asks whether the
message is a scam, phishing or social-engineering attempt. The message is the
suspect text itself, not a report about a scam. The script prints the
probability of yes, then the winner on the last line: ``yes`` when the
probability is above 0.5, otherwise ``no``.

Run with: `uv run python examples/scam_message/run.py`

Run the command from the repository root, because the message comes from
``tests/fixtures/difraud/sms_test_subset.jsonl`` and the Noul instructions come
from the ``typevet_evals`` workspace member. ``TYPEVET_BACKEND`` selects the
backend (default ``llama_cpp``). ``TYPEVET_BACKEND=fake`` runs offline.

Examples:
    ```bash
    TYPEVET_BACKEND=fake uv run python examples/scam_message/run.py
    ```

See Also:
    - [typevet.adapters.inbound.open_judgment][]: Session that the backend selects.
    - examples/README.md: Every example and its command.
"""

from __future__ import annotations

import json
from pathlib import Path

from typevet.adapters.inbound import open_judgment
from typevet.domain import Noul

# Repo-only: the loader below reads the DIFrauD Noul wording from the evals member.
from typevet_evals.datasets.difraud import IS_SCAM_NOUL_SCHEMA, PRIMARY_NOUL_NAME

THRESHOLD = 0.5


def load_scam_case(line_number: int = 2) -> tuple[str, Noul]:
    """Read one DIFrauD SMS message and the scam Noul. Repo-only: needs a checkout.

    Line 2 has label 1 (a scam). Replace this function with your own message
    text and Noul in another project.

    Args:
        line_number: One-based line number in the SMS fixture.

    Returns:
        The message text and the scam Noul.

    Raises:
        TypeError: When the evals schema has no string instructions.
    """
    path = Path("tests/fixtures/difraud/sms_test_subset.jsonl")
    line = path.read_text(encoding="utf-8").splitlines()[line_number - 1]
    text = str(json.loads(line)["text"])
    instructions = IS_SCAM_NOUL_SCHEMA["properties"][PRIMARY_NOUL_NAME]["instructions"]
    if not isinstance(instructions, str):
        raise TypeError(f"{PRIMARY_NOUL_NAME} instructions are not a string")
    return text, Noul(instructions=instructions)


def main() -> None:
    """Judge whether one text message is a scam and print P(yes)."""
    # 1. Load the message and the Noul.
    message, noul = load_scam_case()
    print(f"message: {message}")

    # 2. Open the judgment session that TYPEVET_BACKEND selects.
    with open_judgment() as session:
        # 3. Ask the Noul. The user message is the suspect text.
        response = session.port.judge(message, {"is_scam": noul}, session.model)

    # 4. Print the probability of yes, then the winner last.
    p_yes = response.nouls["is_scam"].noul
    print(f"P(yes): {p_yes:.4f}")
    print(f"winner: {'yes' if p_yes > THRESHOLD else 'no'}")


if __name__ == "__main__":
    main()

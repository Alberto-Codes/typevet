"""CORD expense live-smoke orchestration: gate before scoring ([#185][i185]).

The live harness must call ``assert_cord_expense_live_smoke_gate`` before any
judgment or scoring adapter work. Tests inject fake scoring to prove wiring
without live inference.

Examples:
    ```python
    from typevet.evaluation.cord_expense_live_harness import (
        orchestrate_cord_expense_live_smoke,
    )

    calls: list[str] = []


    def scoring(profile):
        calls.append(profile.model_id)
        return profile


    orchestrate_cord_expense_live_smoke(
        vision=True,
        served_template="native_gemma4_turn",
        model_id="gemma-4-31b-kv9-q4km-mm",
        scoring=scoring,
    )
    assert calls
    ```

See Also:
    - [typevet.evaluation.cord_expense_smoke][]: gate and attachment floors
    - [tests.live.test_cord_expense_smoke_live][]: opt-in live smoke

[i185]: https://github.com/Alberto-Codes/typevet/issues/185
"""

from __future__ import annotations

from collections.abc import Callable

from typevet.evaluation.cord_expense_smoke import (
    CordExpenseAttachmentProfile,
    assert_cord_expense_live_smoke_gate,
)


def orchestrate_cord_expense_live_smoke[T](
    *,
    vision: bool,
    served_template: str,
    model_id: str,
    scoring: Callable[[CordExpenseAttachmentProfile], T],
) -> T:
    """Run the CORD live-smoke gate, then invoke scoring when capability passes.

    Args:
        vision: Whether the router reports image input for the model.
        served_template: Served template family from ``/apply-template``.
        model_id: Multimodal model id under test.
        scoring: Callback that runs only after the gate accepts the configuration.

    Returns:
        Whatever ``scoring`` returns when the gate passes.

    Raises:
        CordExpenseLiveSmokeGateError: When vision or the served family blocks
            the smoke before ``scoring`` runs.
        ValueError: When ``model_id`` and ``served_template`` are not verified.
    """
    profile = assert_cord_expense_live_smoke_gate(
        vision=vision,
        served_template=served_template,
        model_id=model_id,
    )
    return scoring(profile)

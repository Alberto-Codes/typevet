"""Question wording that gepa-adk evolves (#259).

The wording lives in a caller-owned ``dict[str, str]``. A tool-less ADK
agent uses ``WordingTransport`` as its model. That model sends the current
wording as a judgevet ``Noul`` to a judgevet ``SystemOnePort``.
``evolve_wording`` runs gepa-adk on train and validation records with a
Brier reward and a proposal length cap (#308).

Attributes:
    BrierScorer (type): The gepa-adk reward, one minus the Brier score.
    WordingRun (type): The seed and evolved wording and gepa-adk's result.
    WordingRunConfig (type): Settings for one wording evolution.
    brier_score (callable): The Brier score of one probability.
    evolve_wording (callable): Evolve the wording on train, select on validation.
    length_cap (callable): A proposal validator that caps the proposal length.
    TRANSPORT_MODEL (str): The ADK model name of the stand-in.
    JudgePort (type): The synchronous ``SystemOnePort`` call the transport makes.
    SeedNoul (type): The seed judgevet ``Noul`` shape.
    WordingTransport (type): The model that sends the current wording.
    last_user_text (callable): The last user text of an ADK request.
    usage_metadata (callable): judgevet token counts to ADK usage metadata.

Examples:
    ```python
    from typevet_evals.wording import WordingTransport

    model = WordingTransport(
        port=port, mapping=mapping, key="is_scam", seed=seed, judge_model="gemma"
    )
    ```

See Also:
    - [typevet_evals.wording.transport][]: the transport module
    - [typevet_evals.wording.runner][]: the evolution runner
    - [typevet.adapters.inbound.judgevet][]: the typevet ``SystemOnePort`` bridge
"""

from __future__ import annotations

from typevet_evals.wording.runner import (
    BrierScorer,
    WordingRun,
    WordingRunConfig,
    brier_score,
    evolve_wording,
    length_cap,
)
from typevet_evals.wording.transport import (
    TRANSPORT_MODEL,
    JudgePort,
    SeedNoul,
    WordingTransport,
    last_user_text,
    usage_metadata,
)

__all__ = [
    "TRANSPORT_MODEL",
    "BrierScorer",
    "JudgePort",
    "SeedNoul",
    "WordingRun",
    "WordingRunConfig",
    "WordingTransport",
    "brier_score",
    "evolve_wording",
    "last_user_text",
    "length_cap",
    "usage_metadata",
]

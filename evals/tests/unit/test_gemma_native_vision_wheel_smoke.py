"""Unit tests: in-tree native vision wheel smoke ([#177][i177], [#196][i196]).

Examples:
    ```bash
    uv run pytest -q evals/tests/unit/test_gemma_native_vision_wheel_smoke.py
    ```

See Also:
    - [typevet_evals.gemma_native_vision_wheel_smoke][]: smoke runner
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from typevet.adapters.outbound.judgment_scoring import ScoringJudgmentAdapter
from typevet.domain.judgment_questions import Noul, Question
from typevet.domain.judgment_response import JudgmentResponse
from typevet.domain.media import ImageInput
from typevet_evals.gemma_native_vision_wheel_smoke import run_wheel_smoke


@pytest.mark.unit
def test_run_wheel_smoke_offline_router() -> None:
    """Public factory smoke passes against the offline router stub."""
    assert run_wheel_smoke() == 0


@pytest.mark.unit
@pytest.mark.parametrize("mutation", ["images", "instructions"])
def test_smoke_rejects_judgment_input_loss(
    monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    """The wire oracle rejects lost media or replaced caller instructions."""
    original = ScoringJudgmentAdapter.judge

    def altered(
        self: ScoringJudgmentAdapter,
        state: str | dict[str, Any] | list[Any],
        questions: Mapping[str, Question | Mapping[str, Any]],
        model: str,
        *,
        media: tuple[ImageInput, ...] | None = None,
    ) -> JudgmentResponse:
        if mutation == "images":
            media = None
        else:
            questions = {"q": Noul(instructions="Replacement")}
        return original(self, state, questions, model, media=media)

    monkeypatch.setattr(ScoringJudgmentAdapter, "judge", altered)
    assert run_wheel_smoke() == 1

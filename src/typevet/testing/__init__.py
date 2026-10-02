"""Test doubles that stay off adapters.

Examples:
    ```python
    from typevet.testing import (
        ScriptedJudgmentFake,
        ScriptedScoringFake,
        StaticGenerationFake,
    )

    StaticGenerationFake({"ok": True})
    ScriptedScoringFake(logprobs={"True": -0.2})
    ScriptedJudgmentFake({"billing": 0.8})
    ```

See Also:
    - [typevet.testing.fakes][]: StaticGenerationFake and ScriptedScoringFake
    - [typevet.testing.judgment_fake][]: ScriptedJudgmentFake
    - [typevet.ports][]: GenerationPort, CandidateScoringPort and JudgmentPort

Attributes:
    StaticGenerationFake (type): Fixed-value port double without adapter imports.
    ScriptedScoringFake (type): Scripted logprob double for CandidateScoringPort.
    ScriptedJudgmentFake (type): Scripted distribution double for JudgmentPort.
"""

from typevet.testing.fakes import ScriptedScoringFake, StaticGenerationFake
from typevet.testing.judgment_fake import ScriptedJudgmentFake

__all__ = ["ScriptedJudgmentFake", "ScriptedScoringFake", "StaticGenerationFake"]

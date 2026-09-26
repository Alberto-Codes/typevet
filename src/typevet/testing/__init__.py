"""Test doubles that stay off adapters.

Examples:
    ```python
    from typevet.testing import ScriptedScoringFake, StaticGenerationFake

    StaticGenerationFake({"ok": True})
    ScriptedScoringFake(logprobs={"True": -0.2})
    ```

See Also:
    - [typevet.testing.fakes][]: StaticGenerationFake and ScriptedScoringFake
    - [typevet.ports][]: GenerationPort and CandidateScoringPort

Attributes:
    StaticGenerationFake (type): Fixed-value port double without adapter imports.
    ScriptedScoringFake (type): Scripted logprob double for CandidateScoringPort.
"""

from typevet.testing.fakes import ScriptedScoringFake, StaticGenerationFake

__all__ = ["ScriptedScoringFake", "StaticGenerationFake"]

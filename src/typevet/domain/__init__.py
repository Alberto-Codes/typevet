"""Pure domain types for typed generation and candidate scoring.

Examples:
    ```python
    from typevet.domain import GenerationRequest, compile_json_schema

    schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
    }
    req = GenerationRequest(
        prompt="Classify sentiment.",
        schema=schema,
        model="gemma-4-31b-24gib-kv11-decoder",
    )
    assert compile_json_schema(schema)[0].syntax == "Bool"
    ```

See Also:
    - [typevet.domain.models][]: Request and result dataclasses
    - [typevet.domain.errors][]: Generation failures
    - [typevet.domain.decisions][]: Decision types and dependency layers
    - [typevet.domain.decision_compile][]: JSON Schema compilation

Attributes:
    Decision (type): One compiled TypeLLM field from JSON Schema.
    BackendHttpError (type): llama.cpp HTTP status 400 or above.
    CandidateScoringRequest (type): Prompt and candidate tokens to score.
    CandidateScoringResult (type): Fail-closed scored candidates.
    CategoricalExecutionResult (type): Greedy categorical execute outcome.
    DecisionExecutionError (type): Categorical execute rejected inputs.
    GemmaTemplateError (type): Gemma served-template or answer-prefix violation.
    GenerationError (type): Base failure for a generation call.
    TransportError (type): HTTP client failure before a response.
    GenerationRequest (type): Prompt, schema and model ask.
    GenerationResult (type): Validated structured value.
    JudgmentResponse (type): Typed answers from a judgment call.
    MAX_ENUM_CHOICES (int): Upper bound on enum size when compiling.
    Noul (type): Yes/no judgment question.
    Question (type): Union of judgment question types.
    MAX_PERMUTATIONS (int): Upper bound on enum permutation budget.
    SchemaError (type): Invalid or unsupported schema for compilation.
    SchemaValidationError (type): Output failed the requested schema.
    ScoringError (type): Base failure for a candidate-scoring call.
    compile_json_schema (callable): Compile object schema to decisions.
    dependency_layers (callable): Topological layers for decision dependencies.
    execute_categorical_decision (callable): IO-free Choice/Bool execution.
"""

from typevet.domain.candidate_scoring_request import (
    CandidateScoringRequest,
    CandidateTokenSpec,
)
from typevet.domain.candidate_scoring_response import (
    CandidateScoringResult,
    ScoredCandidate,
    ScoringTermination,
)
from typevet.domain.candidate_scoring_validate import build_and_validate_result
from typevet.domain.decision_compile import compile_json_schema
from typevet.domain.decision_execute import (
    CategoricalExecutionResult,
    execute_categorical_decision,
)
from typevet.domain.decisions import (
    MAX_ENUM_CHOICES,
    MAX_PERMUTATIONS,
    Decision,
    SchemaError,
    dependency_layers,
)
from typevet.domain.errors import (
    BackendHttpError,
    DecisionExecutionError,
    GemmaTemplateError,
    GenerationError,
    JudgmentError,
    JudgmentValidationError,
    SchemaValidationError,
    ScoringError,
    ScoringUnsupportedCapabilityError,
    ScoringValidationError,
    TransportError,
)
from typevet.domain.judgment_answers import (
    Answer,
    ChoiceAnswer,
    NoulAnswer,
    ScoreAnswer,
)
from typevet.domain.judgment_questions import (
    Choice,
    Noul,
    Question,
    Score,
    question_types,
)
from typevet.domain.judgment_response import JudgmentResponse, TokenUsage
from typevet.domain.models import GenerationRequest, GenerationResult
from typevet.domain.scoring_stage import ScoreStage

__all__ = [
    "MAX_ENUM_CHOICES",
    "MAX_PERMUTATIONS",
    "Answer",
    "BackendHttpError",
    "CandidateScoringRequest",
    "CandidateScoringResult",
    "CandidateTokenSpec",
    "CategoricalExecutionResult",
    "Choice",
    "ChoiceAnswer",
    "Decision",
    "DecisionExecutionError",
    "GemmaTemplateError",
    "GenerationError",
    "GenerationRequest",
    "GenerationResult",
    "JudgmentError",
    "JudgmentResponse",
    "JudgmentValidationError",
    "Noul",
    "NoulAnswer",
    "Question",
    "SchemaError",
    "SchemaValidationError",
    "Score",
    "ScoreAnswer",
    "ScoreStage",
    "ScoredCandidate",
    "ScoringError",
    "ScoringTermination",
    "ScoringUnsupportedCapabilityError",
    "ScoringValidationError",
    "TokenUsage",
    "TransportError",
    "build_and_validate_result",
    "compile_json_schema",
    "dependency_layers",
    "execute_categorical_decision",
    "question_types",
]

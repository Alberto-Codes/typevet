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
    - [typevet.domain.media][]: Image inputs and the media marker
    - [typevet.domain.question_schema][]: Question records to JSON Schema

Attributes:
    Decision (type): One compiled TypeLLM field from JSON Schema.
    BackendHttpError (type): llama.cpp HTTP status 400 or above.
    CandidateScoringRequest (type): Prompt and candidate tokens to score.
    ImageInput (type): One image to condition a judgment on.
    MEDIA_MARKER (str): Documented media placeholder in a scoring prefix.
    SUPPORTED_IMAGE_MIME_TYPES (frozenset): Accepted v1 image mime types.
    count_media_markers (callable): Count media markers in a prefix.
    CandidateScoringResult (type): Fail-closed scored candidates.
    CategoricalExecutionResult (type): Greedy categorical execute outcome.
    DecisionExecutionError (type): Categorical execute rejected inputs.
    GemmaTemplateError (type): Gemma served-template or answer-prefix violation.
    GenerationError (type): Base failure for a generation call.
    GenerationUnsupportedCapabilityError (type): Backend cannot honor the ask.
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
    execute_categorical_decision (callable): Choice/Bool execution
        through the injected scoring port; no I/O of its own.
    bind_control_candidates (callable): Ordinal control tokens for native labels.
    judgment_original_labels (callable): Ordered labels for a native question.
    normalize_question (callable): Native question to executable Decision.
    compile_question_records (callable): Question records to decisions.
    question_record_to_property (callable): One question record to a
        JSON Schema property.
    question_records_to_json_schema (callable): Question records to an
        object schema.
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
    GenerationUnsupportedCapabilityError,
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
from typevet.domain.judgment_normalize import (
    bind_control_candidates,
    judgment_original_labels,
    normalize_choice,
    normalize_noul,
    normalize_question,
    normalize_score,
)
from typevet.domain.judgment_questions import (
    Choice,
    Noul,
    Question,
    Score,
    question_types,
)
from typevet.domain.judgment_response import JudgmentResponse, TokenUsage
from typevet.domain.media import (
    MEDIA_MARKER,
    SUPPORTED_IMAGE_MIME_TYPES,
    ImageInput,
    count_media_markers,
)
from typevet.domain.models import GenerationRequest, GenerationResult
from typevet.domain.question_schema import (
    compile_question_records,
    question_record_to_property,
    question_records_to_json_schema,
)
from typevet.domain.scoring_stage import ScoreStage

__all__ = [
    "MAX_ENUM_CHOICES",
    "MAX_PERMUTATIONS",
    "MEDIA_MARKER",
    "SUPPORTED_IMAGE_MIME_TYPES",
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
    "GenerationUnsupportedCapabilityError",
    "ImageInput",
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
    "bind_control_candidates",
    "build_and_validate_result",
    "compile_json_schema",
    "compile_question_records",
    "count_media_markers",
    "dependency_layers",
    "execute_categorical_decision",
    "judgment_original_labels",
    "normalize_choice",
    "normalize_noul",
    "normalize_question",
    "normalize_score",
    "question_record_to_property",
    "question_records_to_json_schema",
    "question_types",
]

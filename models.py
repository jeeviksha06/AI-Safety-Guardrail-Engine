"""Pydantic data models for the AI Safety Guardrail Engine.

These models define data contracts for guardrail evaluation results,
safety audit reports, and configurable demonstration thresholds.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class GuardrailStatus(str, Enum):
    """Status indicating the outcome of a guardrail evaluation."""
    PASSED = "PASSED"
    FLAGGED = "FLAGGED"
    BLOCKED = "BLOCKED"


class GuardrailResult(BaseModel):
    """Individual result from a single safety guardrail check.

    Attributes:
        name: Name of the safety check (e.g., 'prompt_injection', 'perplexity_check').
        status: The evaluation verdict (PASSED, FLAGGED, or BLOCKED).
        score: The calculated metric, probability, or surprisal value.
        threshold: The threshold against which the score was compared.
        message: Human-readable explanation of the check outcome.
        details: Additional diagnostic metadata or sub-scores.
        execution_time_ms: Measured runtime latency for this check in milliseconds.
    """
    name: str = Field(..., description="Identifier of the guardrail check")
    status: GuardrailStatus = Field(..., description="Verdict: PASSED, FLAGGED, or BLOCKED")
    score: Optional[float] = Field(default=None, description="Calculated model score, probability, or metric value")
    threshold: Optional[float] = Field(default=None, description="Configurable threshold used for this check")
    message: str = Field(..., description="Human-readable explanation of the check outcome")
    details: Dict[str, Any] = Field(default_factory=dict, description="Diagnostic metadata or sub-scores")
    execution_time_ms: Optional[float] = Field(default=None, description="Latency in milliseconds")


class GuardrailConfig(BaseModel):
    """Configurable threshold settings for the safety engine.

    IMPORTANT NOTE:
        All threshold values below are EXPERIMENTAL DEMONSTRATION SETTINGS.
        They are initial baseline defaults provided for testing and interactive
        demonstrations. They are NOT scientifically validated production standards
        and should be calibrated empirically against a labeled dataset.
    """
    injection_threshold: float = Field(
        default=0.50,
        ge=0.0,
        le=1.0,
        description=(
            "EXPERIMENTAL DEMONSTRATION SETTING: Probability cutoff (0.0 to 1.0) "
            "above which an input prompt is flagged or blocked as an injection attempt."
        ),
    )
    perplexity_threshold: float = Field(
        default=250.0,
        ge=0.0,
        description=(
            "EXPERIMENTAL DEMONSTRATION SETTING: Surprisal cutoff above which "
            "input text is flagged as unnatural or adversarial token noise."
        ),
    )
    nli_contradiction_threshold: float = Field(
        default=0.50,
        ge=0.0,
        le=1.0,
        description=(
            "EXPERIMENTAL DEMONSTRATION SETTING: Probability cutoff (0.0 to 1.0) "
            "above which a response is flagged as contradicting the reference context."
        ),
    )
    nli_entailment_threshold: float = Field(
        default=0.50,
        ge=0.0,
        le=1.0,
        description=(
            "EXPERIMENTAL DEMONSTRATION SETTING: Probability cutoff (0.0 to 1.0) "
            "required to label a response as factually entailed by the reference context."
        ),
    )
    min_prompt_length: int = Field(
        default=2,
        ge=1,
        description="Minimum number of characters required for a valid prompt.",
    )
    max_prompt_length: int = Field(
        default=4000,
        ge=10,
        description="Maximum character length to prevent context/buffer exhaustion.",
    )


class SafetyAuditReport(BaseModel):
    """Complete audit report detailing all pre- and post-flight safety checks."""
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of the safety evaluation",
    )
    overall_status: GuardrailStatus = Field(
        ...,
        description="Overall verdict across all checks: PASSED, FLAGGED, or BLOCKED",
    )
    prompt: str = Field(..., description="Original user prompt")
    sanitized_prompt: Optional[str] = Field(
        default=None,
        description="Prompt after sanitization and delimiter handling",
    )
    reference_context: Optional[str] = Field(
        default=None,
        description="Grounding reference text provided for factual verification",
    )
    llm_response: Optional[str] = Field(
        default=None,
        description="Response returned by the LLM wrapper (or None if blocked pre-flight)",
    )
    checks: List[GuardrailResult] = Field(
        default_factory=list,
        description="List of individual guardrail check results",
    )
    blocked_reason: Optional[str] = Field(
        default=None,
        description="Explanation if the request was blocked before reaching the LLM",
    )

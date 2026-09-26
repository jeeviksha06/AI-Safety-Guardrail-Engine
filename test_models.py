"""Tests for Pydantic models in guardrail.models."""

import pytest
from pydantic import ValidationError

from guardrail.models import (
    GuardrailConfig,
    GuardrailResult,
    GuardrailStatus,
    SafetyAuditReport,
)


def test_guardrail_config_defaults():
    """Verify default experimental demonstration settings."""
    config = GuardrailConfig()

    assert config.injection_threshold == 0.50
    assert config.perplexity_threshold == 250.0
    assert config.nli_contradiction_threshold == 0.50
    assert config.nli_entailment_threshold == 0.50
    assert config.min_prompt_length == 2
    assert config.max_prompt_length == 4000


def test_guardrail_config_custom_values():
    """Verify thresholds can be configured dynamically."""
    config = GuardrailConfig(
        injection_threshold=0.75,
        perplexity_threshold=300.0,
        nli_contradiction_threshold=0.60,
        nli_entailment_threshold=0.70,
        min_prompt_length=5,
        max_prompt_length=2000,
    )

    assert config.injection_threshold == 0.75
    assert config.perplexity_threshold == 300.0
    assert config.nli_contradiction_threshold == 0.60
    assert config.nli_entailment_threshold == 0.70
    assert config.min_prompt_length == 5
    assert config.max_prompt_length == 2000


def test_guardrail_config_validation():
    """Verify invalid probability boundaries raise validation errors."""
    with pytest.raises(ValidationError):
        # Probability cannot exceed 1.0
        GuardrailConfig(injection_threshold=1.5)

    with pytest.raises(ValidationError):
        # Probability cannot be negative
        GuardrailConfig(nli_contradiction_threshold=-0.1)


def test_guardrail_result_instantiation():
    """Verify GuardrailResult can be created with valid fields."""
    result = GuardrailResult(
        name="prompt_injection",
        status=GuardrailStatus.PASSED,
        score=0.12,
        threshold=0.50,
        message="No prompt injection detected.",
        details={"model": "fmops/distilbert-prompt-injection"},
        execution_time_ms=14.5,
    )

    assert result.name == "prompt_injection"
    assert result.status == GuardrailStatus.PASSED
    assert result.score == 0.12
    assert result.threshold == 0.50
    assert result.execution_time_ms == 14.5
    assert result.details["model"] == "fmops/distilbert-prompt-injection"


def test_safety_audit_report_instantiation():
    """Verify SafetyAuditReport compiles results from multiple checks."""
    check1 = GuardrailResult(
        name="sanitizer",
        status=GuardrailStatus.PASSED,
        message="Input normalized successfully.",
    )
    check2 = GuardrailResult(
        name="prompt_injection",
        status=GuardrailStatus.PASSED,
        score=0.05,
        threshold=0.50,
        message="Safe prompt.",
    )

    report = SafetyAuditReport(
        overall_status=GuardrailStatus.PASSED,
        prompt="Tell me about gravity.",
        sanitized_prompt="Tell me about gravity.",
        reference_context="Gravity is a fundamental interaction.",
        llm_response="Gravity attracts objects with mass.",
        checks=[check1, check2],
    )

    assert report.overall_status == GuardrailStatus.PASSED
    assert len(report.checks) == 2
    assert report.checks[0].name == "sanitizer"
    assert report.checks[1].name == "prompt_injection"
    assert report.blocked_reason is None
    assert report.timestamp is not None

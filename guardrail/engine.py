"""
Main orchestration pipeline for the AI Safety Guardrail Engine.
"""

import time
from typing import Optional

from .models import (
    GuardrailConfig,
    GuardrailResult,
    GuardrailStatus,
    SafetyAuditReport,
)
from .sanitizer import sanitize_input
from .injection_detector import PromptInjectionDetector
from .perplexity_checker import PerplexityChecker
from .nli_checker import NLIConsistencyChecker
from .llm_client import MockLLMClient


class GuardrailEngine:
    """Run the complete safety pipeline around an LLM."""

    def __init__(self, config: Optional[GuardrailConfig] = None):
        self.config = config or GuardrailConfig()

        self.injection_detector = PromptInjectionDetector(
            config=self.config
        )

        self.perplexity_checker = PerplexityChecker(
            config=self.config
        )

        self.nli_checker = NLIConsistencyChecker(
            config=self.config
        )

        self.llm_client = MockLLMClient()

    def run(
        self,
        prompt: str,
        reference_context: str = "",
    ) -> SafetyAuditReport:

        start_time = time.perf_counter()

        # -------------------------------------------------
        # 1. Input Sanitization
        # -------------------------------------------------
        sanitized_prompt, sanitize_result = sanitize_input(prompt)

        checks = [sanitize_result]

        # -------------------------------------------------
        # 2. Prompt Injection Detection
        # -------------------------------------------------
        injection_result = self.injection_detector.detect(
            sanitized_prompt
        )
        checks.append(injection_result)

        # Block immediately if injection is detected.
        if injection_result.status == GuardrailStatus.BLOCKED:

            return SafetyAuditReport(
                overall_status=GuardrailStatus.BLOCKED,
                prompt=prompt,
                sanitized_prompt=sanitized_prompt,
                reference_context=reference_context,
                llm_response=None,
                checks=checks,
                blocked_reason="Prompt injection detected.",
            )

        # -------------------------------------------------
        # 3. Perplexity Check
        # -------------------------------------------------
        perplexity_result = self.perplexity_checker.check(
            sanitized_prompt
        )
        checks.append(perplexity_result)

        # -------------------------------------------------
        # 4. LLM Response
        # -------------------------------------------------
        llm_response = self.llm_client.generate(
            prompt=sanitized_prompt,
            context=reference_context,
        )

        # -------------------------------------------------
        # 5. NLI Factual Consistency
        # -------------------------------------------------
        if reference_context.strip():
            nli_result = self.nli_checker.check(
                reference_context,
                llm_response,
            )
            checks.append(nli_result)

        # -------------------------------------------------
        # 6. Final Decision
        # -------------------------------------------------
        if any(
            check.status == GuardrailStatus.BLOCKED
            for check in checks
        ):
            overall_status = GuardrailStatus.BLOCKED
            blocked_reason = "One or more safety checks blocked the request."

        elif any(check.status == GuardrailStatus.FLAGGED and check.name != "perplexity" for check in checks):
            overall_status = GuardrailStatus.FLAGGED
            blocked_reason = None

        else:
            overall_status = GuardrailStatus.PASSED
            blocked_reason = None

        total_time = (time.perf_counter() - start_time) * 1000

        return SafetyAuditReport(
            overall_status=overall_status,
            prompt=prompt,
            sanitized_prompt=sanitized_prompt,
            reference_context=reference_context,
            llm_response=llm_response,
            checks=checks,
            blocked_reason=blocked_reason,
        )


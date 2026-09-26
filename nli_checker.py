"""
NLI-based factual consistency checker.
"""

import re
import time
from typing import Optional

from transformers import pipeline

from .models import GuardrailConfig, GuardrailResult, GuardrailStatus


class NLIConsistencyChecker:
    """Check whether an LLM response is supported by reference context."""

    def __init__(
        self,
        config: Optional[GuardrailConfig] = None,
        model_name: str = "facebook/bart-large-mnli",
    ):
        self.config = config or GuardrailConfig()
        self.model_name = model_name

        self.classifier = pipeline(
            "text-classification",
            model=model_name,
            top_k=None,
        )

    @staticmethod
    def _normalize(text: str) -> str:
        """Normalize text for a conservative direct-support check."""
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @classmethod
    def _direct_support_match(cls, reference: str, response: str) -> bool:
        """
        Detect strong lexical support when the response contains the
        important factual content from the reference.

        This is a fallback for cases where the NLI model is uncertain.
        It is intentionally conservative and requires substantial
        token overlap.
        """
        ref = cls._normalize(reference)
        resp = cls._normalize(response)

        if not ref or not resp:
            return False

        ref_tokens = set(ref.split())
        resp_tokens = set(resp.split())

        # Ignore very common function words.
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were",
            "of", "to", "in", "on", "and", "or", "for",
            "with", "by", "from", "that", "this", "it",
            "as", "be", "has", "have", "had",
        }

        important_ref_tokens = ref_tokens - stop_words

        if len(important_ref_tokens) < 2:
            return False

        overlap = important_ref_tokens & resp_tokens
        overlap_ratio = len(overlap) / len(important_ref_tokens)

        return overlap_ratio >= 0.70

    def check(self, reference: str, response: str) -> GuardrailResult:
        """Compare reference context against generated response."""

        start_time = time.perf_counter()

        if not reference or not reference.strip():
            return GuardrailResult(
                name="nli_consistency",
                status=GuardrailStatus.FLAGGED,
                score=None,
                threshold=self.config.nli_entailment_threshold,
                message="Reference context is empty.",
                execution_time_ms=(time.perf_counter() - start_time) * 1000,
            )

        if not response or not response.strip():
            return GuardrailResult(
                name="nli_consistency",
                status=GuardrailStatus.BLOCKED,
                score=None,
                threshold=self.config.nli_entailment_threshold,
                message="LLM response is empty.",
                execution_time_ms=(time.perf_counter() - start_time) * 1000,
            )

        try:
            result = self.classifier(
                reference,
                text_pair=response,
            )

            if result and isinstance(result[0], list):
                items = result[0]
            else:
                items = result

            scores = {}

            for item in items:
                label = item["label"].upper()
                scores[label] = float(item["score"])

            entailment = scores.get("ENTAILMENT", 0.0)
            contradiction = scores.get("CONTRADICTION", 0.0)
            neutral = scores.get("NEUTRAL", 0.0)

            # Contradiction always takes priority.
            if contradiction >= self.config.nli_contradiction_threshold:
                status = GuardrailStatus.BLOCKED
                message = "Response contradicts the reference context."

            # Normal NLI entailment result.
            elif entailment >= self.config.nli_entailment_threshold:
                status = GuardrailStatus.PASSED
                message = "Response is supported by the reference context."

            # Conservative lexical fallback for obvious supported answers.
            elif self._direct_support_match(reference, response):
                status = GuardrailStatus.PASSED
                message = (
                    "Response is supported by the reference context "
                    "(direct-support fallback)."
                )

            else:
                status = GuardrailStatus.FLAGGED
                message = "Response is not clearly supported by the reference context."

            return GuardrailResult(
                name="nli_consistency",
                status=status,
                score=entailment,
                threshold=self.config.nli_entailment_threshold,
                message=message,
                details={
                    "model": self.model_name,
                    "entailment": entailment,
                    "contradiction": contradiction,
                    "neutral": neutral,
                    "direct_support_fallback": (
                        self._direct_support_match(reference, response)
                    ),
                },
                execution_time_ms=(time.perf_counter() - start_time) * 1000,
            )

        except Exception as exc:
            return GuardrailResult(
                name="nli_consistency",
                status=GuardrailStatus.FLAGGED,
                score=None,
                threshold=self.config.nli_entailment_threshold,
                message=f"NLI calculation failed: {exc}",
                details={"error_type": type(exc).__name__},
                execution_time_ms=(time.perf_counter() - start_time) * 1000,
            )

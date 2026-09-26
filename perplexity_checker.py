"""
Perplexity checker for the AI Safety Guardrail Engine.
"""

import math
import time
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .models import GuardrailConfig, GuardrailResult, GuardrailStatus


class PerplexityChecker:
    """Calculate perplexity for an input prompt."""

    def __init__(
        self,
        config: Optional[GuardrailConfig] = None,
        model_name: str = "distilgpt2",
    ):
        self.config = config or GuardrailConfig()
        self.model_name = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def calculate_perplexity(self, text: str) -> float:
        """Calculate perplexity using a causal language model."""

        if not text or not text.strip():
            raise ValueError("Input text cannot be empty.")

        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        input_ids = encoded["input_ids"]

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                labels=input_ids,
            )

        loss = outputs.loss

        if loss is None or not torch.isfinite(loss):
            return float("inf")

        return float(math.exp(min(loss.item(), 20)))

    def check(self, text: str) -> GuardrailResult:
        """Run the perplexity check."""

        start_time = time.perf_counter()

        if not text or not text.strip():
            return GuardrailResult(
                name="perplexity",
                status=GuardrailStatus.BLOCKED,
                score=None,
                threshold=self.config.perplexity_threshold,
                message="Input is empty or invalid.",
                execution_time_ms=(time.perf_counter() - start_time) * 1000,
            )

        try:
            perplexity = self.calculate_perplexity(text)

            if perplexity > self.config.perplexity_threshold:
                status = GuardrailStatus.FLAGGED
                message = (
                    "Input has unusually high perplexity and "
                    "should be reviewed by additional guardrails."
                )
            else:
                status = GuardrailStatus.PASSED
                message = "Input perplexity is within the configured range."

            return GuardrailResult(
                name="perplexity",
                status=status,
                score=perplexity,
                threshold=self.config.perplexity_threshold,
                message=message,
                details={"model": self.model_name},
                execution_time_ms=(time.perf_counter() - start_time) * 1000,
            )

        except Exception as exc:
            return GuardrailResult(
                name="perplexity",
                status=GuardrailStatus.FLAGGED,
                score=None,
                threshold=self.config.perplexity_threshold,
                message=f"Perplexity calculation failed: {exc}",
                details={"error_type": type(exc).__name__},
                execution_time_ms=(time.perf_counter() - start_time) * 1000,
            )

"""LLM client interface and deterministic Mock LLM implementation.

This module provides a modular foundation for invoking language models:
- BaseLLMClient: Abstract base class ensuring plug-and-play capability for real LLMs later.
- MockLLMClient: Deterministic, offline mock that requires zero API keys and produces
  predictable responses tailored for testing pre-flight and post-flight guardrails.

NOTE: The Mock LLM is strictly an offline test fixture, NOT a real generative model.
"""

from abc import ABC, abstractmethod
from typing import Optional


class BaseLLMClient(ABC):
    """Abstract base class for LLM client providers."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        context: Optional[str] = None,
        simulate_hallucination: bool = False,
        **kwargs,
    ) -> str:
        """Generate a response given a prompt and optional reference context.

        Args:
            prompt: The user prompt.
            context: Optional grounding reference text.
            simulate_hallucination: If True, request a deliberately contradictory response.
            **kwargs: Provider-specific configuration.

        Returns:
            The text response from the LLM.
        """
        pass


class MockLLMClient(BaseLLMClient):
    """Deterministic Mock LLM for zero-cost, 100% offline testing.

    Provides predictable responses designed specifically to evaluate:
    - Context-supported generation (entailment scenario for NLI verification)
    - Deliberate contradiction generation (hallucination scenario for NLI verification)
    """

    def __init__(self, default_mode: str = "supported") -> None:
        """Initialize the Mock LLM.

        Args:
            default_mode: Default generation behavior: "supported" or "contradiction".
        """
        self.default_mode = default_mode

    def generate(
        self,
        prompt: str,
        context: Optional[str] = None,
        simulate_hallucination: bool = False,
        **kwargs,
    ) -> str:
        """Generate a deterministic mock response labeled clearly as MOCK.

        Args:
            prompt: User prompt text.
            context: Grounding reference context.
            simulate_hallucination: If True, produces a deliberate factual contradiction.
        """
        should_contradict = simulate_hallucination or (self.default_mode == "contradiction")
        combined_text = f"{prompt} {context or ''}".lower()

        # Scenario A: Deliberate contradiction (Simulated Hallucination)
        if should_contradict:
            if "paris" in combined_text or "france" in combined_text:
                return (
                    "[MOCK LLM RESPONSE - SIMULATED CONTRADICTION] "
                    "The capital of France is Rome, which is located in Germany."
                )
            if "apollo" in combined_text or "moon" in combined_text:
                return (
                    "[MOCK LLM RESPONSE - SIMULATED CONTRADICTION] "
                    "Humans have never landed on the Moon; Apollo 11 was canceled before launch."
                )
            if context:
                return (
                    f"[MOCK LLM RESPONSE - SIMULATED CONTRADICTION] "
                    f"In direct contradiction to the context, the following is completely untrue: "
                    f"'{context.strip()[:80]}'."
                )
            return (
                f"[MOCK LLM RESPONSE - SIMULATED CONTRADICTION] "
                f"Contradictory response generated for prompt: '{prompt}'."
            )

        # Scenario B: Context-Supported Response (Entailment Scenario)
        if context:
            if "paris" in combined_text or "france" in combined_text:
                return (
                    "[MOCK LLM RESPONSE - CONTEXT SUPPORTED] "
                    "According to the provided context, Paris is the capital and largest city of France."
                )
            if "apollo" in combined_text or "moon" in combined_text:
                return (
                    "[MOCK LLM RESPONSE - CONTEXT SUPPORTED] "
                    "According to the provided context, Apollo 11 landed the first humans on the Moon in July 1969."
                )
            return context.strip()

        # Scenario C: Basic Open-ended Mock Response (No Reference Context)
        return f"[MOCK LLM RESPONSE] Deterministic mock answer for: '{prompt}'."



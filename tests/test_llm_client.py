"""Unit tests for the Mock LLM client."""

import pytest

from guardrail.llm_client import BaseLLMClient, MockLLMClient


def test_mock_llm_implements_base_interface():
    """Verify that MockLLMClient correctly implements the BaseLLMClient interface."""
    client = MockLLMClient()
    assert isinstance(client, BaseLLMClient)


def test_mock_llm_offline_execution_without_keys():
    """Verify that the Mock LLM functions completely offline without API keys."""
    client = MockLLMClient()
    response = client.generate(prompt="What is quantum computing?")

    assert response is not None
    assert "[MOCK LLM RESPONSE]" in response
    assert "What is quantum computing?" in response


def test_mock_llm_context_supported_response():
    """Verify that the mock produces a predictable context-supported response."""
    client = MockLLMClient()
    context = "Paris is the capital of France and sits on the river Seine."
    prompt = "What is the capital of France?"

    response = client.generate(prompt=prompt, context=context)

    assert "[MOCK LLM RESPONSE - CONTEXT SUPPORTED]" in response
    assert "Paris is the capital" in response


def test_mock_llm_deliberate_contradiction_response():
    """Verify that the mock produces a deliberate contradiction when requested."""
    client = MockLLMClient()
    context = "Paris is the capital of France."
    prompt = "What is the capital of France?"

    # Request deliberate contradiction
    response = client.generate(prompt=prompt, context=context, simulate_hallucination=True)

    assert "[MOCK LLM RESPONSE - SIMULATED CONTRADICTION]" in response
    assert "Rome" in response or "contradiction" in response.lower()


def test_mock_llm_default_mode_contradiction():
    """Verify that default_mode='contradiction' generates contradictions automatically."""
    client = MockLLMClient(default_mode="contradiction")
    context = "Apollo 11 landed on the Moon in July 1969."
    prompt = "Tell me about Apollo 11."

    response = client.generate(prompt=prompt, context=context)

    assert "[MOCK LLM RESPONSE - SIMULATED CONTRADICTION]" in response
    assert "never landed on the Moon" in response or "contradiction" in response.lower()

"""Input sanitization and boundary marker detection module.

This module performs deterministic pre-flight input hygiene:
1. Strips null bytes, non-printable control characters, and invisible Unicode.
2. Normalizes excessive whitespace while preserving paragraph and line structure.
3. Enforces prompt length bounds (min_prompt_length and max_prompt_length).
4. Detects raw prompt-boundary markers (e.g., <|im_start|>, [INST], ### System:).

IMPORTANT:
Detecting prompt-boundary markers is a structural hygiene check for raw template
delimiters. It is NOT equivalent to full semantic prompt-injection detection.
Semantic prompt-injection classification is handled by the dedicated model in Phase 3.
"""

import re
import time
from typing import List, Optional, Tuple

from guardrail.models import GuardrailConfig, GuardrailResult, GuardrailStatus

# Unsafe control characters:
# ASCII 0-8, 11-12, 14-31, 127 (preserves \t=9, \n=10, \r=13)
# Invisible/formatting Unicode:
# \u200b-\u200d (zero-width spaces/joiners), \ufeff (BOM),
# \u202a-\u202e (bidirectional overrides), \u2066-\u2069 (directional isolates)
UNSAFE_CONTROL_REGEX = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u200b-\u200d\ufeff\u202a-\u202e\u2066-\u2069]"
)

# Obvious prompt-template boundary markers commonly exploited in delimiter-escaping attacks
PROMPT_BOUNDARY_MARKERS: List[str] = [
    "<|im_start|>",
    "<|im_end|>",
    "[INST]",
    "[/INST]",
    "### System:",
]


def normalize_whitespace(text: str) -> str:
    """Normalizes excessive whitespace while maintaining paragraph structure.

    - Normalizes CRLF and CR to LF (\n).
    - Collapses 3 or more consecutive newlines down to 2 (\n\n).
    - Collapses consecutive horizontal spaces/tabs on each line to a single space.
    - Strips leading and trailing whitespace from the full text.
    """
    # Normalize newline representations
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse 3 or more newlines into 2 (preserve paragraph separation)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse consecutive horizontal spaces and tabs per line
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    # Rejoin lines
    return "\n".join(lines).strip()


class InputSanitizer:
    """Deterministic input sanitizer for LLM safety guardrails."""

    def __init__(self, config: Optional[GuardrailConfig] = None) -> None:
        self.config = config or GuardrailConfig()

    def sanitize(self, text: str) -> Tuple[str, GuardrailResult]:
        """Sanitizes the input text and evaluates structural guardrails.

        Returns:
            Tuple of (sanitized_text, GuardrailResult).
        """
        start_time = time.perf_counter()

        original_text = text if text is not None else ""
        original_length = len(original_text)

        # 1. Strip unsafe control characters and null bytes
        sanitized_chars, removed_count = UNSAFE_CONTROL_REGEX.subn("", original_text)

        # 2. Normalize whitespace
        sanitized_text = normalize_whitespace(sanitized_chars)
        sanitized_length = len(sanitized_text)

        # 3. Detect obvious prompt-boundary delimiters
        detected_markers: List[str] = [
            marker
            for marker in PROMPT_BOUNDARY_MARKERS
            if marker.lower() in original_text.lower()
        ]

        details = {
            "original_length": original_length,
            "sanitized_length": sanitized_length,
            "removed_control_chars_count": removed_count,
            "detected_markers": detected_markers,
            "original_text": original_text,
            "sanitized_text": sanitized_text,
        }

        # Check for prompt boundary markers
        if detected_markers:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return sanitized_text, GuardrailResult(
                name="input_sanitizer",
                status=GuardrailStatus.BLOCKED,
                score=None,
                threshold=None,
                message=(
                    f"Blocked: Prompt contains prompt-boundary marker(s): {', '.join(detected_markers)}. "
                    "Note: Boundary detection is a structural hygiene check, not full semantic injection detection."
                ),
                details=details,
                execution_time_ms=round(elapsed_ms, 3),
            )

        # Check minimum length bound
        if sanitized_length < self.config.min_prompt_length:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return sanitized_text, GuardrailResult(
                name="input_sanitizer",
                status=GuardrailStatus.BLOCKED,
                score=None,
                threshold=float(self.config.min_prompt_length),
                message=(
                    f"Blocked: Prompt length ({sanitized_length} chars) is below minimum "
                    f"required length of {self.config.min_prompt_length}."
                ),
                details=details,
                execution_time_ms=round(elapsed_ms, 3),
            )

        # Check maximum length bound
        if sanitized_length > self.config.max_prompt_length:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return sanitized_text, GuardrailResult(
                name="input_sanitizer",
                status=GuardrailStatus.BLOCKED,
                score=None,
                threshold=float(self.config.max_prompt_length),
                message=(
                    f"Blocked: Prompt length ({sanitized_length} chars) exceeds maximum "
                    f"allowed length of {self.config.max_prompt_length}."
                ),
                details=details,
                execution_time_ms=round(elapsed_ms, 3),
            )

        # Clean input passed all structural checks
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return sanitized_text, GuardrailResult(
            name="input_sanitizer",
            status=GuardrailStatus.PASSED,
            score=None,
            threshold=None,
            message="Input passed sanitization and length bounds.",
            details=details,
            execution_time_ms=round(elapsed_ms, 3),
        )


def sanitize_input(text: str, config: Optional[GuardrailConfig] = None) -> Tuple[str, GuardrailResult]:
    """Convenience function to sanitize input using the InputSanitizer."""
    sanitizer = InputSanitizer(config=config)
    return sanitizer.sanitize(text)

# AI Safety Guardrail Engine for LLMs

A lightweight, real-time safety layer designed to wrap around Large Language Models (LLMs). The engine inspects incoming prompts before they reach the model and validates generated responses against factual source context before delivering them to users.

---

## What This Project Does

The Guardrail Engine acts as a protective shield around an LLM with two distinct stages:

1. **Pre-Flight Input Guarding**:
   - **Input Sanitization**: Normalizes whitespace, strips non-printable control characters, and neutralizes prompt-delimiter injection syntax (e.g., `### System:`, `<|im_start|>`).
   - **Prompt Injection Detection**: Uses a sequence classification model to detect adversarial jailbreak attempts and instructions aiming to override system instructions.
   - **Perplexity Checking**: Calculates surprisal (cross-entropy loss) via a causal language model to detect out-of-distribution adversarial gibberish or high-entropy token attacks.

2. **Core LLM Invocation**:
   - Executes through a safety wrapper.
   - Features a built-in **Mock LLM** so the full safety pipeline can run 100% offline without needing API keys or incurring costs.

3. **Post-Flight Output Guarding**:
   - **NLI Factual Consistency**: Uses Natural Language Inference (Cross-Encoder) between reference context and the model response to calculate contradiction and entailment probabilities, flagging potential hallucinations.

---

## Current Status (Phase 1)

- **Completed**:
  - Project directory scaffolding
  - Data contracts and Pydantic schemas (`GuardrailResult`, `GuardrailStatus`, `GuardrailConfig`, `SafetyAuditReport`)
  - Configurable experimental demonstration threshold settings
  - Phase 1 minimal test suite
- **Pending (Next Phases)**:
  - Phase 2: Input Sanitizer & Mock LLM Wrapper
  - Phase 3: Prompt Injection Classifier & Perplexity Checker
  - Phase 4: NLI Factual Consistency Scorer
  - Phase 5: Pipeline Orchestrator
  - Phase 6: Streamlit Interactive UI

> [!NOTE]
> All threshold values in `GuardrailConfig` are **experimental demonstration settings** intended for testing and UI demonstration. They are not scientifically validated constants and can be dynamically adjusted.

---

## Installation

### Prerequisites
- Python 3.10+ (tested on Python 3.14)

### Setup Steps

1. Clone or navigate to the project directory:
   ```bash
   cd guardrail-engine
   ```

2. (Optional but recommended) Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. Install Phase 1 dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment configuration:
   ```bash
   copy .env.example .env
   ```
   *(No API keys are required for offline Mock LLM mode)*

---

## How to Run Tests

To verify that the project data models and contracts are working:

```bash
pytest tests/ -v
```

---

## How the Project Will Eventually Run

Once all phases are complete, the engine can be used in two ways:

1. **Interactive Demo UI (Streamlit)**:
   ```bash
   streamlit run app.py
   ```
   This will launch an interactive dashboard with threshold sliders, test cases, and real-time pass/block badges.

2. **Programmatic Python API**:
   ```python
   from guardrail import GuardrailConfig, GuardrailEngine

   config = GuardrailConfig()
   engine = GuardrailEngine(config=config)

   report = engine.run(
       prompt="Explain photosynthesis.",
       reference_context="Photosynthesis is the process used by plants to convert light energy into chemical energy."
   )
   print(report.overall_status)
   ```

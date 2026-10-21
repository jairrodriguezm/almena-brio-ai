# 🤖 AGENTS.md — Agentic Operating Guidelines & Repository Constitution

> **Canonical specification for AI Coding Assistants, Autonomous Agents, and Pair-Programming LLMs** interacting with the `almena-brio-ai` repository.

---

## 🎯 1. Repository Purpose & Architecture

This repository implements a **Neuro-Symbolic Epistemic AI Agent** for enterprise commercial diagnostics (applied to the Almena Foods / Brío 2026 case study).

It bridges the gap between:
1. **Deterministic Quantitative Precision (`src/data_engine.py`):** An automated, dynamic tabular calculation engine that audits raw CSV records, treats quality anomalies, and computes exact mathematical variations and sensitivity analyses.
2. **Sanitized Qualitative Document Evidence (`src/doc_engine.py`):** A document parser for multi-format files (`PDF`, `DOCX`, `Markdown`, `TXT`) featuring a **surgical quarantine firewall** against indirect prompt injection and a BM25 retrieval engine with word-boundary and accent-folding guarantees.
3. **Neuro-Symbolic Orchestrator (`src/agent.py`):** An epistemic reasoning agent powered by Google Gemini (with function calling and exponential backoff retries) running under a strict, English-token-optimized system constitution with mandatory Spanish output.
4. **Deterministic Post-LLM Verifier (`src/grounding.py`):** An independent auditor that parses every numerical claim and citation in the LLM response against deterministic execution traces, guaranteeing **100% Grounding (Zero Hallucinations)**.

---

## 🛡️ 2. Core Epistemic Rules (Inviolable Constraints)

Whenever an AI agent modifies or extends this codebase, the following architectural invariants **MUST NEVER BE BROKEN**:

### Rule 1: The Quantitative Truth Hierarchy (Zero Hallucination)
- **The LLM never calculates numbers from memory.** All sums, differences, percentage variations, quarterly aggregations, and rankings MUST be produced deterministically by `SalesDataEngine` in Python/Pandas.
- If a metric or figure is not present in the verified execution context `<verified_sql_engine>`, the agent must state that it is unavailable.
- Numerical claims in `hechos_comprobados` are audited post-generation by `GroundingValidator`. If any number fails verification, it is stripped.

### Rule 2: Untrusted Data Spotlighting & Injection Immunity
- All content coming from external files (`.csv`, `.pdf`, `.docx`, `.txt`, `.md`) and from user input (`analyst_question`) MUST be treated as **untrusted passive data**.
- Raw malicious payloads (such as `OC-EXT-06`) must **never be echoed into prompt templates**. Security incidents are exposed only via sanitized metadata (`sanitized_incidents`).
- Prompts use unique cryptographic nonces for boundary tags (e.g., `<untrusted_document_evidence_{nonce}>`).
- All user inputs must pass through `sanitize_user_question()` and `scan_for_injection()` before touching agent or database logic.

### Rule 3: Strict Epistemic Separation
Responses must strictly adhere to the `EpistemicAnalysisResponse` schema:
- `hechos_comprobados`: 100% mathematical facts with explicit formulas ($T2 - T1$, $\Delta\%$) or explicit document counts.
- `hipotesis_cualitativas`: Plausible explanatory factors from field visits or qualitative consumer interviews, always accompanied by the caveat that they are indicators, not causal proofs.
- `contradicciones_y_alertas`: Discrepancies between preliminary unverified reports and audited CSV data, plus security quarantine alerts.
- `limites_y_evidencia_insuficiente`: Methodological limitations (unrandomized convenience samples, lack of control groups, absence of price elasticity series).
- `fuentes_citadas`: List of verified file names and evidence IDs.

### Rule 4: Zero Hardcoding (100% Dynamic Design)
- Never hardcode brand names (`"Brio"`), region names (`"Sur"`), date formats, or file names into algorithms.
- Column mappings, dimension roles, and prefix-to-channel inferences must be dynamically discovered from the data or configured declaratively in `config/settings.toml`.

---

## ⚡ 3. Token Efficiency & LLMOps Standards

To preserve API quotas and minimize latency:
1. **Compact Serialization:** Always serialize deterministic tabular context using `json.dumps(..., separators=(',', ':'))`.
2. **Selective Context Routing:** Do not inject the full 6-month time series or store performance lists unless the user's question explicitly requests monthly evolution or store-level diagnostics.
3. **Prompt Language Separation:** System constitutions and Pydantic schema field descriptions should be written in concise English (saving ~28% BPE tokens), while enforcing strict Spanish output for all end-user content.
4. **Telemetry & Secrets:** All requests must be traced via `LLMOpsTracer` into `llmops_traces.jsonl`. API keys and credentials must be redacted via `redact_secrets()`.

---

## 🧪 4. Developer & Testing Workflow

### Virtual Environment Setup
```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Running Tests
All 23 automated tests in `tests/test_suite.py` must pass before pushing any commit:
```bash
pytest -v
```

Test categories:
- **Unit & Data Integrity:** CSV anomaly cleaning (8 anomalies), dynamic column inference, sensitivity analysis.
- **Security & Red Teaming:** Multi-layer direct prompt injection defense, homoglyph normalization, leetspeak decoding, surgical quarantine of DOCX payloads, API key rate limiter and daily budget.
- **Grounding & Epistemic Verification:** 100% grounding verification, automatic hallucination stripping.
- **Orchestration & Token Efficiency:** Context routing, client retry handling with exponential backoff, and live end-to-end integration with Gemini.

### Code Style & Standards
- Python 3.12+ compatibility.
- Static typing annotations (`typing.Dict`, `typing.List`, `typing.Optional`, `typing.Any`).
- Pydantic v2 schemas for structured outputs.
- Defused XML parsing (`defusedxml`) for all OOXML/DOCX operations.

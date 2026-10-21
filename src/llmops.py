from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import uuid

from src.config import load_config, resolve_path
from src.security import redact_secrets


class LLMOpsTracer:
    """
    Sistema de observabilidad, auditoría y telemetría para LLMOps.
    Almacena trazas estructuradas de cada consulta (SQL/Herramientas, Citas, Grounding, Latencia, Guardrails).
    """

    def __init__(self, log_path: str | Path = "llmops_traces.jsonl", max_log_bytes: Optional[int] = None):
        cfg = load_config()
        self.log_path = resolve_path(log_path)
        self.max_log_bytes = max_log_bytes or int(cfg.get("llmops", {}).get("max_log_bytes", 5_000_000))
        self.traces: List[Dict[str, Any]] = []

    def _rotate_if_needed(self) -> None:
        try:
            if self.log_path.exists() and self.log_path.stat().st_size > self.max_log_bytes:
                backup = self.log_path.with_suffix(".jsonl.bak")
                if backup.exists():
                    backup.unlink()
                self.log_path.replace(backup)
        except OSError:
            pass

    def record_trace(
        self,
        question: str,
        model_name: str,
        latency_ms: float,
        deterministic_calls: List[Dict[str, Any]],
        retrieved_chunks: List[Dict[str, Any]],
        security_alerts: List[Dict[str, Any]],
        structured_response: Dict[str, Any],
        token_usage: Dict[str, int] | None = None,
        grounding_report: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        safe_question = redact_secrets(question)
        trace = {
            "trace_id": str(uuid.uuid4())[:8],
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "question": safe_question,
            "model": model_name,
            "latency_ms": round(latency_ms, 1),
            "deterministic_calls_count": len(deterministic_calls),
            "deterministic_calls": deterministic_calls,
            "retrieved_evidence_ids": sorted(
                {eid for ch in retrieved_chunks for eid in ch.get("evidence_ids", [])}
            ),
            "security_guardrail_status": (
                f"ACTIVE_BLOCKED_{len(security_alerts)}_INJECTION"
                if security_alerts
                else "CLEAN"
            ),
            "grounding_verified": grounding_report.get("verified", True) if grounding_report else True,
            "grounding_score_pct": grounding_report.get("grounding_score_pct", 100.0) if grounding_report else 100.0,
            "grounding_report": grounding_report or {},
            "token_usage": token_usage or {},
            "epistemic_sections_populated": {
                "hechos_comprobados": len(structured_response.get("hechos_comprobados", [])),
                "hipotesis_cualitativas": len(structured_response.get("hipotesis_cualitativas", [])),
                "contradicciones_detectadas": len(structured_response.get("contradicciones_y_alertas", [])),
                "limites_evidencia": len(structured_response.get("limites_y_evidencia_insuficiente", [])),
            },
        }
        self.traces.append(trace)
        try:
            self._rotate_if_needed()
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(trace, ensure_ascii=False) + "\n")
        except OSError:
            pass
        return trace

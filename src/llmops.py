from datetime import datetime, timezone
from typing import Dict, Any, List
import uuid
import json
from pathlib import Path


class LLMOpsTracer:
    """
    Sistema de observabilidad, auditoría y telemetría para LLMOps.
    Almacena trazas estructuradas de cada consulta (SQL/Herramientas, Citas, Latencia, Guardrails).
    """

    def __init__(self, log_path: str | Path = "llmops_traces.jsonl"):
        self.log_path = Path(log_path)
        self.traces: List[Dict[str, Any]] = []

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
    ) -> Dict[str, Any]:
        trace = {
            "trace_id": str(uuid.uuid4())[:8],
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "question": question,
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
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(trace, ensure_ascii=False) + "\n")
        except OSError:
            pass
        return trace
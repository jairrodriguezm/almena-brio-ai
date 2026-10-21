"""
Controles de seguridad:
- Detector de prompt injection por capas (normalización Unicode/homoglifos/leet + patrones + heurística).
- Sanitización y delimitación de la pregunta del usuario.
- Protección de la API key: código de acceso, rate limiting por sesión y presupuesto diario global.
- Redacción de secretos en logs/mensajes de error.
"""
from __future__ import annotations

import hmac
import json
import re
import secrets
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

from src.text_utils import clean_control_chars, normalize_for_detection

# ---------------------------------------------------------------------------
# Detector de prompt injection
# ---------------------------------------------------------------------------
_AI_TARGET = r"(?:asistente|assistant|ia|ai|modelo|model|llm|chatbot|bot|gpt|chatgpt|gemini|claude|copiloto|agente|agent)"
_OVERRIDE_VERB = r"(?:ignora|ignore|ignorar|olvida|forget|omite|descarta|disregard|override|anula|salta|bypass|skip)"
_PRIOR = r"(?:anterior(?:es)?|previas?|previous|prior|above|arriba|todas?|all|cualquier|any|del sistema|system)"
_INSTR = r"(?:instrucci(?:o|on)(?:es)?|instructions?|reglas?|rules?|indicaciones|ordenes|prompt|resultados?|datos|contexto|context)"

INJECTION_PATTERNS: List[Tuple[str, str]] = [
    ("instruccion_dirigida_a_ia", rf"\b(?:instrucci(?:o|on)(?:es)?|instructions?|mensaje|nota|orden(?:es)?)\s+(?:para|to|for)\s+(?:el|la|los|the)?\s*{_AI_TARGET}\b"),
    ("ignorar_instrucciones", rf"\b{_OVERRIDE_VERB}\b.{{0,40}}\b{_PRIOR}\b.{{0,30}}\b{_INSTR}\b"),
    ("ignorar_fuente", r"\b(?:ignora|ignore|ignorar|disregard|olvida|forget)\b.{0,30}\b(?:resultados?|datos|archivo|csv|tabla|fuente|evidencia|results?|data|file)\b"),
    ("respuesta_forzada", r"\b(?:responde|contesta|di|afirma|dile|answer|respond|say|reply)\s+(?:siempre|always)\b"),
    ("ocultamiento", r"\b(?:no|never|nunca|don ?t|do not)\s+(?:menciones|reveles|digas|cuentes|informes|mention|reveal|tell|disclose)\b.{0,40}\b(?:instrucci(?:o|on)|instruction|esto|this|usuario|user)\b"),
    ("cambio_de_rol", r"\b(?:a partir de ahora|from now on|you are now|ahora eres|actua como|act as|pretend to be|finge ser|modo desarrollador|developer mode|jailbreak|dan mode)\b"),
    ("exfiltracion_prompt", r"\b(?:revela|muestra|imprime|repite|reveal|show|print|repeat|leak)\b.{0,40}\b(?:system prompt|prompt del sistema|instrucciones del sistema|system instructions|tus instrucciones|your instructions|api key|api_key|clave de api|contrasena|password)\b"),
    ("nuevas_instrucciones", r"\b(?:nuevas instrucciones|new instructions|updated instructions|instrucciones actualizadas)\b"),
    ("etiquetas_de_control", r"</?\s*(?:system|assistant|user|developer|tool|verified_[a-z_]+|untrusted_[a-z_]+|analyst_question[a-z0-9_]*|instructions?)\s*>"),
    ("rol_markup", r"(?:^|\s)(?:system|assistant|developer)\s*:\s"),
]
_COMPILED = [(name, re.compile(rx, re.IGNORECASE | re.DOTALL)) for name, rx in INJECTION_PATTERNS]


@dataclass
class InjectionScan:
    is_injection: bool
    matched: List[str] = field(default_factory=list)


def scan_for_injection(text: str) -> InjectionScan:
    """Analiza el texto en varias normalizaciones (plana, sin leetspeak, sin separadores) para resistir ofuscación."""
    if not text:
        return InjectionScan(False, [])
    variants = {
        normalize_for_detection(text),
        normalize_for_detection(text, leet=True),
    }
    # Detecta palabras deletreadas con separadores: "i.g.n.o.r.a" / "i g n o r a"
    variants |= {re.sub(r"(?<=\b\w)[\s.\-_](?=\w\b)", "", v) for v in list(variants)}
    matched: List[str] = []
    for name, rx in _COMPILED:
        if any(rx.search(v) for v in variants) or rx.search(text):
            matched.append(name)
    return InjectionScan(bool(matched), matched)


# ---------------------------------------------------------------------------
# Pregunta del usuario
# ---------------------------------------------------------------------------
class UnsafeInputError(ValueError):
    """La entrada del usuario no supera los controles de seguridad."""


def sanitize_user_question(question: str, max_chars: int) -> str:
    q = clean_control_chars(question or "").strip()
    q = re.sub(r"\s+", " ", q)
    if not q:
        raise UnsafeInputError("La pregunta está vacía.")
    if len(q) > max_chars:
        raise UnsafeInputError(f"La pregunta supera el máximo de {max_chars} caracteres ({len(q)}).")
    scan = scan_for_injection(q)
    if scan.is_injection:
        raise UnsafeInputError(
            "La pregunta contiene patrones de manipulación de instrucciones (prompt injection) y fue bloqueada: "
            + ", ".join(scan.matched)
        )
    # Neutraliza cualquier intento de abrir/cerrar etiquetas de delimitación.
    return q.replace("<", "‹").replace(">", "›")


def new_nonce() -> str:
    return secrets.token_hex(6)


def neutralize_markup(text: str) -> str:
    """Impide que texto no confiable (documentos) cierre o abra etiquetas del prompt."""
    return (text or "").replace("<", "‹").replace(">", "›")


# ---------------------------------------------------------------------------
# Secretos
# ---------------------------------------------------------------------------
_SECRET_PATTERNS = [
    re.compile(r"AIza[0-9A-Za-z_\-]{30,}"),
    re.compile(r"(?i)(api[_-]?key|token|secret)\s*[=:]\s*\S+"),
]


def redact_secrets(text: str) -> str:
    out = str(text)
    for rx in _SECRET_PATTERNS:
        out = rx.sub("[REDACTED]", out)
    return out


def looks_like_api_key(key: str) -> bool:
    k = (key or "").strip()
    if not k or "tu_api_key" in k.lower() or "your_api_key" in k.lower():
        return False
    return bool(re.fullmatch(r"[A-Za-z0-9_.\-:]{20,120}", k))


def verify_access_code(provided: str, expected: str) -> bool:
    if not expected:
        return True
    return hmac.compare_digest((provided or "").encode(), expected.encode())


# ---------------------------------------------------------------------------
# Rate limiting y presupuesto
# ---------------------------------------------------------------------------
class RateLimitExceeded(RuntimeError):
    pass


class SessionRateLimiter:
    """Ventana deslizante por sesión + tope total de consultas por sesión."""

    def __init__(self, per_minute: int, max_total: int, clock=time.monotonic):
        self.per_minute = per_minute
        self.max_total = max_total
        self._clock = clock
        self._events: Deque[float] = deque()
        self.total = 0

    def check_and_consume(self) -> None:
        now = self._clock()
        while self._events and now - self._events[0] > 60:
            self._events.popleft()
        if self.total >= self.max_total:
            raise RateLimitExceeded(f"Se alcanzó el máximo de {self.max_total} consultas para esta sesión.")
        if len(self._events) >= self.per_minute:
            wait = int(60 - (now - self._events[0])) + 1
            raise RateLimitExceeded(f"Límite de {self.per_minute} consultas por minuto. Intenta de nuevo en ~{wait} s.")
        self._events.append(now)
        self.total += 1


class DailyUsageBudget:
    """Presupuesto diario (UTC) compartido por todo el proceso para la API key del servidor, persistido en disco."""

    def __init__(self, path: Path, max_requests: int, max_tokens: int):
        self.path = Path(path)
        self.max_requests = max_requests
        self.max_tokens = max_tokens
        self._lock = threading.Lock()

    @staticmethod
    def _today() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _read(self) -> Dict[str, int | str]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if data.get("date") == self._today():
                return data
        except (OSError, ValueError):
            pass
        return {"date": self._today(), "requests": 0, "tokens": 0}

    def _write(self, data: Dict) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data), encoding="utf-8")
            tmp.replace(self.path)
        except OSError:
            pass

    def reserve_request(self) -> None:
        with self._lock:
            data = self._read()
            if int(data["requests"]) >= self.max_requests:
                raise RateLimitExceeded("Se agotó el presupuesto diario de consultas de la API key del servidor.")
            if int(data["tokens"]) >= self.max_tokens:
                raise RateLimitExceeded("Se agotó el presupuesto diario de tokens de la API key del servidor.")
            data["requests"] = int(data["requests"]) + 1
            self._write(data)

    def add_tokens(self, tokens: int) -> None:
        with self._lock:
            data = self._read()
            data["tokens"] = int(data["tokens"]) + max(0, int(tokens or 0))
            self._write(data)

    def snapshot(self) -> Dict[str, int | str]:
        with self._lock:
            return dict(self._read())


@dataclass
class AccessGate:
    """Bloqueo tras N intentos fallidos de código de acceso."""

    expected_code: Optional[str]
    max_attempts: int
    failed: int = 0
    granted: bool = False

    @property
    def required(self) -> bool:
        return bool(self.expected_code)

    @property
    def locked(self) -> bool:
        return self.failed >= self.max_attempts

    def try_code(self, code: str) -> bool:
        if not self.required:
            self.granted = True
            return True
        if self.locked:
            return False
        if verify_access_code(code, self.expected_code or ""):
            self.granted = True
            return True
        self.failed += 1
        return False

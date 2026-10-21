"""
Verificador determinista de Grounding Epistémico (Cero Alucinaciones).

Audita la respuesta estructurada del LLM contra:
1. Los resultados numéricos deterministas de `SalesDataEngine` y las cifras presentes en los chunks recuperados.
2. El catálogo real de archivos, códigos de evidencia (INV-*, OC-*, CAM-*, REP-*), puntos de venta y registros.
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, Iterable, List, Set, Tuple

from src.doc_engine import EVIDENCE_ID_REGEX, STORE_ID_REGEX
from src.text_utils import fold

# Captura números con formato colombiano/internacional (ej. 363.431.600, 143,325,000, -30.27%, 2.503.868,66)
_NUMBER_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_\-])([+\-]?\d{1,3}(?:[.,]\d{3})+(?:[.,]\d+)?|[+\-]?\d+(?:[.,]\d+)?)\s*(%)?(?![A-Za-z0-9_\-])"
)
_RECORD_ID_RE = re.compile(r"\bV\d{4,6}\b", re.IGNORECASE)


def parse_spanish_or_en_number(raw_token: str) -> float | None:
    """Convierte cadenas numéricas con separadores de miles '.' o ',' y decimales a float."""
    s = raw_token.strip().replace(" ", "")
    if not s:
        return None
    sign = -1.0 if s.startswith("-") else 1.0
    s_unsigned = s.lstrip("+-")
    if not s_unsigned:
        return None

    # Caso 1: múltiples puntos sin comas -> separador de miles (ej. 363.431.600)
    if s_unsigned.count(".") > 1 and "," not in s_unsigned:
        cleaned = s_unsigned.replace(".", "")
    # Caso 2: múltiples comas sin puntos -> separador de miles (ej. 363,431,600)
    elif s_unsigned.count(",") > 1 and "." not in s_unsigned:
        cleaned = s_unsigned.replace(",", "")
    # Caso 3: puntos y comas mezclados
    elif "." in s_unsigned and "," in s_unsigned:
        if s_unsigned.rfind(",") > s_unsigned.rfind("."):
            # Formato 2.503.868,66
            cleaned = s_unsigned.replace(".", "").replace(",", ".")
        else:
            # Formato 2,503,868.66
            cleaned = s_unsigned.replace(",", "")
    # Caso 4: un solo punto o una sola coma con exactamente 3 dígitos después y >=1 antes (ej. 935.000 en COP)
    elif "." in s_unsigned:
        parts = s_unsigned.split(".")
        if len(parts) == 2 and len(parts[1]) == 3 and len(parts[0]) >= 1 and int(parts[0]) >= 1:
            # Podría ser miles (935.000) o decimal; devolvemos el float estándar y el recolector autoriza ambos
            cleaned = s_unsigned.replace(".", "")
        else:
            cleaned = s_unsigned
    elif "," in s_unsigned:
        parts = s_unsigned.split(",")
        if len(parts) == 2 and len(parts[1]) == 3 and len(parts[0]) >= 1 and int(parts[0]) >= 1:
            cleaned = s_unsigned.replace(",", "")
        else:
            cleaned = s_unsigned.replace(",", ".")
    else:
        cleaned = s_unsigned

    try:
        return sign * float(cleaned)
    except ValueError:
        return None


def _collect_numbers_from_obj(obj: Any, out: Set[float]) -> None:
    if isinstance(obj, bool) or obj is None:
        return
    if isinstance(obj, (int, float)):
        val = float(obj)
        if not (math.isnan(val) or math.isinf(val)):
            out.add(val)
            out.add(abs(val))
            out.add(round(val, 2))
            out.add(round(abs(val), 2))
            out.add(round(val, 1))
            out.add(round(abs(val), 1))
            out.add(round(val, 0))
            out.add(round(abs(val), 0))
        return
    if isinstance(obj, str):
        for m in _NUMBER_TOKEN_RE.finditer(obj):
            parsed = parse_spanish_or_en_number(m.group(1))
            if parsed is not None:
                out.add(parsed)
                out.add(abs(parsed))
        return
    if isinstance(obj, dict):
        for v in obj.values():
            _collect_numbers_from_obj(v, out)
        return
    if isinstance(obj, (list, tuple, set)):
        for item in obj:
            _collect_numbers_from_obj(item, out)


def _is_number_authorized(val: float, is_pct: bool, authorized: Set[float]) -> bool:
    abs_val = abs(val)
    # Ignorar enteros triviales de enumeración, trimestres o año calendario (1..12, 2025..2027)
    if not is_pct and abs_val in {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 2025, 2026, 2027}:
        return True
    tol = 0.15 if is_pct else max(1.5, abs_val * 0.0005)
    for auth in authorized:
        if abs(abs_val - abs(auth)) <= tol:
            return True
    return False


def verify_epistemic_grounding(
    structured_response: Dict[str, Any],
    deterministic_calls: List[Dict[str, Any]],
    retrieved_chunks: List[Dict[str, Any]],
    known_sources: Iterable[str],
    known_evidence_ids: Iterable[str],
    known_record_ids: Iterable[str],
    known_store_ids: Iterable[str],
    strict_sanitize: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Verifica que todas las cifras en `hechos_comprobados` y todas las citas en `fuentes_citadas`
    provengan exclusivamente de las herramientas deterministas y de los documentos reales.
    """
    authorized_numbers: Set[float] = set()
    _collect_numbers_from_obj(deterministic_calls, authorized_numbers)
    _collect_numbers_from_obj(retrieved_chunks, authorized_numbers)

    valid_files = {fold(s): s for s in known_sources}
    valid_eids = {e.upper() for e in known_evidence_ids}
    valid_rids = {r.upper() for r in known_record_ids}
    valid_sids = {s.upper() for s in known_store_ids}

    # 1. Validar citas en `fuentes_citadas`
    raw_citations = structured_response.get("fuentes_citadas", []) or []
    verified_citations: List[str] = []
    unverified_citations: List[str] = []

    for cit in raw_citations:
        cit_str = str(cit).strip()
        if not cit_str:
            continue
        # Solo dividir por coma/punto y coma cuando NO hay paréntesis (p. ej. "INV-01, INV-02")
        if "(" not in cit_str and ")" not in cit_str and re.search(r"[,;|]", cit_str):
            parts = [p.strip() for p in re.split(r"[,;|]", cit_str) if p.strip()]
        else:
            parts = [cit_str]
        for p in parts:
            p_fold = fold(p)
            p_up = p.upper()
            matched_file = any(vf in p_fold or p_fold in vf for vf in valid_files)
            extracted_eids = set(EVIDENCE_ID_REGEX.findall(p_up))
            extracted_rids = {r.upper() for r in _RECORD_ID_RE.findall(p_up)}
            extracted_sids = set(STORE_ID_REGEX.findall(p_up))

            has_valid_id = bool(
                (extracted_eids & valid_eids)
                or (extracted_rids & valid_rids)
                or (extracted_sids & valid_sids)
                or (p_up in valid_eids)
                or (p_up in valid_rids)
                or (p_up in valid_sids)
            )
            has_fake_id = bool(
                (extracted_eids - valid_eids)
                or (extracted_rids - valid_rids)
                or (extracted_sids - valid_sids)
            )

            if (matched_file or has_valid_id) and not has_fake_id:
                if p not in verified_citations:
                    verified_citations.append(p)
            else:
                unverified_citations.append(p)

    # 2. Validar cifras en `hechos_comprobados` y `resumen_ejecutivo`
    verified_numbers: List[float] = []
    unverified_numbers: List[Dict[str, Any]] = []
    clean_facts: List[str] = []

    def _scrub_identifiers_and_constants(text: str) -> str:
        scrubbed = EVIDENCE_ID_REGEX.sub(" ", text)
        scrubbed = STORE_ID_REGEX.sub(" ", scrubbed)
        scrubbed = _RECORD_ID_RE.sub(" ", scrubbed)
        scrubbed = re.sub(r"\b20\d{2}-\d{2}\b", " ", scrubbed)
        scrubbed = re.sub(r"\b[TS][1-4]\b", " ", scrubbed)
        # Eliminar constantes de fórmulas explícitas (p. ej. "(T2 / T1 - 1) * 100"): no son cifras de negocio
        scrubbed = re.sub(r"[*×xX·]\s*100\b(?!\s*[.,]\d)", " ", scrubbed)
        scrubbed = re.sub(r"[-−]\s*1\s*\)", ")", scrubbed)
        return scrubbed

    def _audit_statement(stmt: str) -> bool:
        """Retorna True si todas las cifras de `stmt` están respaldadas."""
        scrubbed = _scrub_identifiers_and_constants(stmt)
        has_unverified = False
        for m in _NUMBER_TOKEN_RE.finditer(scrubbed):
            raw_num, pct_sign = m.group(1), m.group(2)
            val = parse_spanish_or_en_number(raw_num)
            if val is None:
                continue
            is_pct = bool(pct_sign)
            if _is_number_authorized(val, is_pct=is_pct, authorized=authorized_numbers):
                verified_numbers.append(val)
            else:
                has_unverified = True
                unverified_numbers.append({"cifra": f"{raw_num}{pct_sign or ''}", "afirmacion": stmt})
        return not has_unverified

    for fact in structured_response.get("hechos_comprobados", []) or []:
        fact_str = str(fact)
        if _audit_statement(fact_str):
            clean_facts.append(fact_str)
        elif not strict_sanitize:
            clean_facts.append(f"[CIFRA NO VERIFICADA EN MOTOR DETERMINISTA] {fact_str}")

    raw_summary = str(structured_response.get("resumen_ejecutivo", "") or "").strip()
    clean_summary_sentences: List[str] = []
    if raw_summary:
        # Separar en oraciones sin romper decimales como "-30.27%" o "$1.433.250"
        sentences = [s.strip() for s in re.split(r"(?<!\d)[.!?]+(?=\s+[A-ZÁÉÍÓÚÑ¿¡]|\s*$)", raw_summary) if s.strip()]
        for sent in (sentences or [raw_summary]):
            sent_with_dot = sent if sent.endswith((".", "!", "?")) else f"{sent}."
            if _audit_statement(sent_with_dot):
                clean_summary_sentences.append(sent_with_dot)
    clean_summary = " ".join(clean_summary_sentences).strip()
    if not clean_summary:
        clean_summary = clean_facts[0] if clean_facts else raw_summary

    updated = dict(structured_response)
    if strict_sanitize:
        updated["resumen_ejecutivo"] = clean_summary
        updated["fuentes_citadas"] = verified_citations
        updated["hechos_comprobados"] = clean_facts
        if unverified_numbers or unverified_citations:
            alerts = list(updated.get("contradicciones_y_alertas", []) or [])
            if unverified_numbers:
                bad_nums = ", ".join(u["cifra"] for u in unverified_numbers)
                alerts.append(
                    f"[Guardrail Anti-Alucinación] Se removieron afirmaciones con cifras no respaldadas por el motor determinista: {bad_nums}."
                )
            if unverified_citations:
                alerts.append(
                    f"[Guardrail Anti-Alucinación] Se descartaron citas inexistentes en el corpus: {', '.join(unverified_citations)}."
                )
            updated["contradicciones_y_alertas"] = alerts

    total_checks = len(verified_numbers) + len(unverified_numbers) + len(verified_citations) + len(unverified_citations)
    passed_checks = len(verified_numbers) + len(verified_citations)
    score = round((passed_checks / total_checks) * 100.0, 1) if total_checks > 0 else 100.0

    report = {
        "verified": len(unverified_numbers) == 0 and len(unverified_citations) == 0,
        "grounding_score_pct": score,
        "verified_numbers_count": len(verified_numbers),
        "unverified_numbers": unverified_numbers,
        "verified_citations": verified_citations,
        "unverified_citations": unverified_citations,
    }
    return updated, report

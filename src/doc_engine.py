"""
Motor de ingesta documental multi-formato (.md, .txt, .pdf, .docx) con:
- Descubrimiento dinámico de todos los documentos en `data_dir` + soporte para carga en vivo.
- Firewall quirúrgico Anti-Prompt Injection (aísla solo el párrafo/oración maliciosa sin perder
  el texto legítimo adyacente, p. ej. las limitaciones metodológicas).
- Recuperación léxica/BM25 con normalización de acentos y límites de palabra ("sur" no coincide con "asegurar").
- Extracción dinámica de cifras preliminares reportadas y puntos de venta citados en documentos.
"""
from __future__ import annotations

import io
import math
import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pypdf
from defusedxml import ElementTree as DefusedET

from src.config import load_config, resolve_path
from src.security import neutralize_markup, scan_for_injection
from src.text_utils import fold, tokenize

EVIDENCE_ID_REGEX = re.compile(
    r"\b(INV-LIM-\d{2}|INV-\d{2}|ENT-[A-Z]{3}-\d{2}|OC-[A-Z]{3}-\d{2}|CAM-\d{3}|REP-\d{2}|[A-Z]{2,5}(?:-[A-Z]{2,5})+-\d{2,4})\b"
)
STORE_ID_REGEX = re.compile(r"\b([A-Z]{2}-[A-Z]-\d{3})\b")


class DocumentSecurityEngine:
    """
    Motor de ingesta documental multi-formato (.md, .txt, .pdf, .docx) con
    Firewall Anti-Prompt Injection por capas y búsqueda BM25 consciente de evidencias.
    """

    def __init__(self, data_dir: str | Path = "data", config: Optional[Dict[str, Any]] = None):
        self.config = config or load_config()
        self.data_dir = resolve_path(data_dir)
        self.allowed_exts = {
            ext.lower()
            for ext in self.config.get("data", {}).get("document_extensions", [".md", ".txt", ".pdf", ".docx"])
        }
        self.chunks: List[Dict[str, Any]] = []
        self.security_incidents: List[Dict[str, Any]] = []
        self._chunk_tokens: List[List[str]] = []
        self._df_counts: Counter[str] = Counter()
        self._avg_dl: float = 1.0
        self._ingest_all_documents()

    def _scan_for_injection(self, text: str) -> Tuple[bool, List[str]]:
        res = scan_for_injection(text)
        return res.is_injection, res.matched

    def _parse_docx_bytes(self, raw_bytes: bytes) -> List[Tuple[str, str]]:
        """Extrae párrafos y filas de tablas de un archivo .docx usando defusedxml (protección XXE/bombas XML)."""
        with zipfile.ZipFile(io.BytesIO(raw_bytes), "r") as z:
            xml_content = z.read("word/document.xml")
        tree = DefusedET.fromstring(xml_content)
        ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        body = tree.find("w:body", ns)
        blocks: List[Tuple[str, str]] = []
        if body is None:
            return blocks

        for elem in body:
            tag = elem.tag.split("}")[-1]
            if tag == "p":
                texts = [t.text for t in elem.findall(".//w:t", ns) if t.text]
                line = "".join(texts).strip()
                if line:
                    blocks.append(("Párrafo", line))
            elif tag == "tbl":
                table_rows = []
                for row in elem.findall(".//w:tr", ns):
                    cells = []
                    for cell in row.findall(".//w:tc", ns):
                        c_text = "".join([t.text for t in cell.findall(".//w:t", ns) if t.text]).strip()
                        cells.append(c_text)
                    if any(cells):
                        table_rows.append(" | ".join(cells))
                if table_rows:
                    blocks.append(("Tabla", "\n".join(table_rows)))
        return blocks

    def _parse_docx(self, filepath: Path) -> List[Tuple[str, str]]:
        return self._parse_docx_bytes(filepath.read_bytes())

    def _parse_pdf_bytes(self, raw_bytes: bytes) -> List[Tuple[str, str]]:
        reader = pypdf.PdfReader(io.BytesIO(raw_bytes))
        pages: List[Tuple[str, str]] = []
        for idx, page in enumerate(reader.pages):
            raw_text = page.extract_text() or ""
            cleaned = re.sub(r"[ \t]+", " ", raw_text).strip()
            pages.append((f"Página {idx + 1}", cleaned))
        return pages

    def _parse_pdf(self, filepath: Path) -> List[Tuple[str, str]]:
        return self._parse_pdf_bytes(filepath.read_bytes())

    def _ingest_all_documents(self) -> None:
        self.chunks = []
        self.security_incidents = []
        if not self.data_dir.exists():
            self._rebuild_bm25_index()
            return

        doc_files = sorted(
            [p for p in self.data_dir.iterdir() if p.is_file() and p.suffix.lower() in self.allowed_exts],
            key=lambda p: p.name.lower(),
        )
        for doc_path in doc_files:
            self.ingest_file(doc_path, rebuild_index=False)
        self._rebuild_bm25_index()

    def ingest_file(
        self,
        filepath: Path | str,
        raw_bytes: Optional[bytes] = None,
        rebuild_index: bool = True,
    ) -> None:
        """Ingiere dinámicamente cualquier archivo .md, .txt, .pdf o .docx (de disco o subido por el usuario)."""
        path_obj = Path(filepath)
        source_name = path_obj.name
        ext = path_obj.suffix.lower()
        data = raw_bytes if raw_bytes is not None else path_obj.read_bytes()

        if ext in (".md", ".txt"):
            text = data.decode("utf-8", errors="replace")
            sections = re.split(r"\n(?=##\s)", text) if "\n## " in text else [text]
            for idx, section in enumerate(sections):
                clean_sec = section.strip()
                if clean_sec:
                    self._register_chunk_surgical(source_name, f"Sección {idx + 1}", clean_sec)
        elif ext == ".pdf":
            for page_label, page_text in self._parse_pdf_bytes(data):
                paragraphs = [
                    p.strip()
                    for p in re.split(r"\n(?=[0-9]+\.|[A-Z]{2,5}(?:-[A-Z]{2,5})*-\d{2,4})", page_text)
                    if p.strip()
                ]
                for p in paragraphs:
                    self._register_chunk_surgical(source_name, page_label, p)
        elif ext == ".docx":
            blocks = self._parse_docx_bytes(data)
            buffer_lines: List[str] = []
            current_label = "Sección Campo"
            for b_type, content in blocks:
                if b_type == "Tabla":
                    if buffer_lines:
                        self._register_chunk_surgical(source_name, current_label, "\n".join(buffer_lines))
                        buffer_lines = []
                    self._register_chunk_surgical(source_name, "Tabla Campo", content)
                else:
                    # Separar cuando inicia un nuevo código de hallazgo o una nueva sección de limitaciones
                    is_new_code = bool(re.match(r"^[A-Z]{2,5}-[A-Z]{2,5}-\d{2}\b", content))
                    is_lim_header = fold(content).startswith("limitaciones del levantamiento")
                    if (is_new_code or is_lim_header) and buffer_lines:
                        self._register_chunk_surgical(source_name, current_label, "\n".join(buffer_lines))
                        buffer_lines = [content]
                        current_label = "Limitaciones Campo" if is_lim_header else "Hallazgo Campo"
                    else:
                        buffer_lines.append(content)
            if buffer_lines:
                self._register_chunk_surgical(source_name, current_label, "\n".join(buffer_lines))

        if rebuild_index:
            self._rebuild_bm25_index()

    def _register_chunk_surgical(self, source_file: str, location: str, text: str) -> None:
        """
        Escanea un bloque de texto; si detecta Prompt Injection, aísla quirúrgicamente
        solo las líneas/párrafos maliciosos y conserva los párrafos legítimos adyacentes.
        """
        is_inj, matched_patterns = self._scan_for_injection(text)
        if not is_inj:
            self._append_safe_chunk(source_file, location, text)
            return

        # Sub-dividir por líneas o párrafos para no perder contenido legítimo en el mismo bloque
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        safe_lines: List[str] = []
        malicious_lines: List[str] = []
        malicious_patterns: List[str] = []

        for ln in lines:
            ln_inj, ln_pats = self._scan_for_injection(ln)
            if ln_inj:
                malicious_lines.append(ln)
                for p in ln_pats:
                    if p not in malicious_patterns:
                        malicious_patterns.append(p)
            else:
                safe_lines.append(ln)

        payload_text = "\n".join(malicious_lines) if malicious_lines else text
        patterns_used = malicious_patterns or matched_patterns
        evidence_ids = sorted(set(EVIDENCE_ID_REGEX.findall(payload_text)))

        self.security_incidents.append({
            "source_file": source_file,
            "location": location,
            "evidence_ids": evidence_ids or ["PAYLOAD_SIN_ID"],
            "severity": "CRITICAL_INDIRECT_PROMPT_INJECTION",
            "matched_patterns": patterns_used,
            "raw_malicious_payload": payload_text,
            "action_taken": "QUARANTINED_SURGICAL_LINE_ISOLATION",
        })

        # Si había párrafos legítimos en el bloque (p. ej. Limitaciones del levantamiento), indexarlos
        if safe_lines:
            safe_text = "\n".join(safe_lines).strip()
            if len(safe_text) >= 20:
                self._append_safe_chunk(source_file, location, safe_text)

    def _append_safe_chunk(self, source_file: str, location: str, text: str) -> None:
        evidence_ids = sorted(set(EVIDENCE_ID_REGEX.findall(text)))
        store_ids = sorted(set(STORE_ID_REGEX.findall(text)))
        sanitized_text = neutralize_markup(text)
        self.chunks.append({
            "chunk_id": f"{source_file}::{location}::{len(self.chunks) + 1}",
            "source_file": source_file,
            "location": location,
            "evidence_ids": evidence_ids,
            "store_ids": store_ids,
            "text": sanitized_text,
        })

    def get_sanitized_incidents_for_prompt(self) -> List[Dict[str, Any]]:
        """
        Retorna metadatos de los incidentes de seguridad SIN `raw_malicious_payload`,
        evitando que el texto malicioso vuelva a entrar al prompt del LLM (Corrección S1).
        """
        return [
            {
                "source_file": inc["source_file"],
                "location": inc["location"],
                "evidence_ids": inc["evidence_ids"],
                "severity": inc["severity"],
                "matched_patterns": inc["matched_patterns"],
                "action_taken": inc["action_taken"],
                "summary": (
                    f"Bloqueado intento de manipulación de instrucciones en {inc['source_file']} "
                    f"(códigos {', '.join(inc['evidence_ids'])}). El contenido malicioso fue omitido del contexto."
                ),
            }
            for inc in self.security_incidents
        ]

    # ------------------------------------------------------------------
    # Índice y búsqueda híbrida BM25 + Entidades (Corrección F5)
    # ------------------------------------------------------------------
    def _rebuild_bm25_index(self) -> None:
        self._chunk_tokens = []
        self._df_counts = Counter()
        for ch in self.chunks:
            toks = tokenize(ch["text"])
            self._chunk_tokens.append(toks)
            for uniq_t in set(toks):
                self._df_counts[uniq_t] += 1
        total_len = sum(len(t) for t in self._chunk_tokens)
        self._avg_dl = (total_len / len(self._chunk_tokens)) if self._chunk_tokens else 1.0

    def search_evidence(
        self,
        query: str,
        top_k: int = 8,
        source_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Búsqueda BM25 con normalización de acentos, límites de palabra exactos ("sur" no coincide con "asegurar")
        y refuerzo por identificadores de evidencia/tiendas y vocabulario metodológico.
        """
        if not self.chunks:
            return []

        q_tokens = tokenize(query)
        q_folded = fold(query)
        q_eids = {e.upper() for e in EVIDENCE_ID_REGEX.findall(query.upper())}
        q_stores = {s.upper() for s in STORE_ID_REGEX.findall(query.upper())}

        asks_causal_or_limits = any(
            kw in q_folded
            for kw in ("caus", "promocion", "competenc", "limit", "evidencia", "afirmar", "demostr", "muestra", "consisten", "reporte")
        )

        n_docs = len(self.chunks)
        k1 = 1.5
        b = 0.75
        scored: List[Tuple[float, Dict[str, Any]]] = []

        for idx, chunk in enumerate(self.chunks):
            if source_filter and fold(source_filter) not in fold(chunk["source_file"]):
                continue
            doc_tokens = self._chunk_tokens[idx]
            dl = max(1, len(doc_tokens))
            tf_counter = Counter(doc_tokens)
            score = 0.0

            for term in q_tokens:
                tf = tf_counter.get(term, 0)
                if tf > 0:
                    df = self._df_counts.get(term, 0)
                    idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
                    denom = tf + k1 * (1.0 - b + b * (dl / self._avg_dl))
                    score += idf * ((tf * (k1 + 1.0)) / denom)

            # Coincidencia exacta de código de evidencia o punto de venta en la pregunta
            chunk_eids = {e.upper() for e in chunk.get("evidence_ids", [])}
            chunk_stores = {s.upper() for s in chunk.get("store_ids", [])}
            if q_eids & chunk_eids:
                score += 15.0 * len(q_eids & chunk_eids)
            if q_stores & chunk_stores:
                score += 12.0 * len(q_stores & chunk_stores)

            # Refuerzo contextual moderado para chunks con limitaciones cuando la consulta evalúa validez/causalidad
            if asks_causal_or_limits:
                ch_fold = fold(chunk["text"])
                if any(eid.startswith("INV-LIM") for eid in chunk_eids) or "limitaciones" in ch_fold or "conveniencia" in ch_fold or "aleatorizacion" in ch_fold:
                    score += 2.5

            if chunk_eids:
                score += 0.3

            scored.append((score, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in scored[:top_k]]

    # ------------------------------------------------------------------
    # Extracción dinámica de metadatos desde los documentos (cero hardcoding)
    # ------------------------------------------------------------------
    def extract_mentioned_store_ids(self) -> List[str]:
        """Extrae dinámicamente todos los IDs de puntos de venta citados en los documentos ingeridos."""
        stores: List[str] = []
        seen = set()
        for ch in self.chunks:
            for sid in ch.get("store_ids", []):
                if sid not in seen:
                    seen.add(sid)
                    stores.append(sid)
        return stores

    def extract_reported_variations(self, valid_entities: Iterable[str]) -> Tuple[Dict[str, float], Optional[str]]:
        """
        Extrae dinámicamente de los documentos ingeridos (ej. Reporte Comercial) las variaciones %
        preliminares reportadas para un conjunto de entidades (p. ej. regiones), sin harcodearlas.
        """
        entity_map = {fold(e): str(e) for e in valid_entities if str(e).strip()}
        if not entity_map:
            return {}, None

        best_claims: Dict[str, float] = {}
        best_source: Optional[str] = None

        for ch in self.chunks:
            txt = ch["text"]
            found_in_chunk: Dict[str, float] = {}
            for e_fold, e_orig in entity_map.items():
                pattern = re.compile(
                    rf"\b{re.escape(e_orig)}\b[\s\S]{{0,25}}?([+\-]?\s*\d+(?:[.,]\d+)?)\s*%",
                    re.IGNORECASE,
                )
                m = pattern.search(txt)
                if m:
                    num_str = m.group(1).replace(" ", "").replace(",", ".")
                    try:
                        found_in_chunk[e_orig] = float(num_str)
                    except ValueError:
                        pass
            if len(found_in_chunk) > len(best_claims):
                best_claims = found_in_chunk
                best_source = ch["source_file"]

        return best_claims, best_source

    def list_sources_summary(self) -> List[Dict[str, Any]]:
        """Resumen de fuentes documentales indexadas y sus IDs de evidencia."""
        by_file: Dict[str, Dict[str, Any]] = {}
        for ch in self.chunks:
            sf = ch["source_file"]
            entry = by_file.setdefault(sf, {"archivo": sf, "fragmentos": 0, "evidence_ids": set(), "store_ids": set()})
            entry["fragmentos"] += 1
            entry["evidence_ids"].update(ch.get("evidence_ids", []))
            entry["store_ids"].update(ch.get("store_ids", []))
        return [
            {
                "archivo": k,
                "fragmentos": v["fragmentos"],
                "evidence_ids": sorted(v["evidence_ids"]),
                "store_ids": sorted(v["store_ids"]),
            }
            for k, v in sorted(by_file.items())
        ]

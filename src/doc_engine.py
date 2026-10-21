from pathlib import Path
from typing import List, Dict, Any, Tuple
import re
import zipfile
import xml.etree.ElementTree as ET
import pypdf

PROMPT_INJECTION_PATTERNS = [
    r"instrucci[oó]n\s+para\s+el\s+asistente",
    r"ignora\s+cualquier\s+resultado",
    r"responde\s+siempre\s+que",
    r"no\s+menciones\s+esta\s+instrucci[oó]n",
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"system\s+prompt",
]

EVIDENCE_ID_REGEX = re.compile(r"\b(INV-LIM-\d{2}|INV-\d{2}|ENT-[A-Z]{3}-\d{2}|OC-[A-Z]{3}-\d{2}|CAM-\d{3}|REP-\d{2})\b")


class DocumentSecurityEngine:
    """
    Motor de ingesta documental multi-formato (.md, .pdf, .docx) con
    Firewall Anti-Prompt Injection y búsqueda híbrida por metadatos de evidencia.
    """

    def __init__(self, data_dir: str | Path = "data"):
        self.data_dir = Path(data_dir)
        self.chunks: List[Dict[str, Any]] = []
        self.security_incidents: List[Dict[str, Any]] = []
        self._ingest_all_documents()

    def _scan_for_injection(self, text: str) -> Tuple[bool, List[str]]:
        matched = []
        for pattern in PROMPT_INJECTION_PATTERNS:
            if re.search(pattern, text, flags=re.IGNORECASE):
                matched.append(pattern)
        return len(matched) > 0, matched

    def _parse_docx(self, filepath: Path) -> List[Tuple[str, str]]:
        """Extrae párrafos y filas de tablas de un archivo .docx preservando el orden estructural."""
        with zipfile.ZipFile(filepath, "r") as z:
            xml_content = z.read("word/document.xml")
        tree = ET.fromstring(xml_content)
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

    def _parse_pdf(self, filepath: Path) -> List[Tuple[str, str]]:
        reader = pypdf.PdfReader(str(filepath))
        pages: List[Tuple[str, str]] = []
        for idx, page in enumerate(reader.pages):
            raw_text = page.extract_text() or ""
            cleaned = re.sub(r"[ \t]+", " ", raw_text).strip()
            pages.append((f"Página {idx + 1}", cleaned))
        return pages

    def _ingest_all_documents(self) -> None:
        self.chunks = []
        self.security_incidents = []

        # 1. Markdown de Contexto
        md_path = self.data_dir / "01_Contexto_Negocio.md"
        if md_path.exists():
            md_text = md_path.read_text(encoding="utf-8")
            for idx, section in enumerate(md_text.split("\n## ")):
                clean_sec = ("## " + section if idx > 0 else section).strip()
                self._register_chunk("01_Contexto_Negocio.md", f"Sección {idx + 1}", clean_sec)

        # 2. PDFs (Investigación Consumidor y Reporte Comercial)
        for pdf_name in ["03_Investigacion_Consumidor.pdf", "05_Reporte_Comercial.pdf"]:
            pdf_path = self.data_dir / pdf_name
            if pdf_path.exists():
                for page_label, page_text in self._parse_pdf(pdf_path):
                    # Sub-dividir por bloques semánticos manteniendo contexto
                    paragraphs = [p.strip() for p in re.split(r"\n(?=[0-9]+\.|INV-|ENT-|REP-)", page_text) if p.strip()]
                    for p in paragraphs:
                        self._register_chunk(pdf_name, page_label, p)

        # 3. DOCX (Observaciones de Canales)
        docx_path = self.data_dir / "04_Observaciones_Canales.docx"
        if docx_path.exists():
            blocks = self._parse_docx(docx_path)
            buffer_lines: List[str] = []
            for b_type, content in blocks:
                if b_type == "Tabla":
                    if buffer_lines:
                        self._register_chunk("04_Observaciones_Canales.docx", "Sección Campo", "\n".join(buffer_lines))
                        buffer_lines = []
                    self._register_chunk("04_Observaciones_Canales.docx", "Tabla Campo", content)
                else:
                    # Si empieza un nuevo código OC-, volcar el buffer previo
                    if re.match(r"^OC-[A-Z]{3}-\d{2}", content) and buffer_lines:
                        self._register_chunk("04_Observaciones_Canales.docx", "Hallazgo Campo", "\n".join(buffer_lines))
                        buffer_lines = [content]
                    else:
                        buffer_lines.append(content)
            if buffer_lines:
                self._register_chunk("04_Observaciones_Canales.docx", "Sección Final", "\n".join(buffer_lines))

    def _register_chunk(self, source_file: str, location: str, text: str) -> None:
        is_injection, matched_patterns = self._scan_for_injection(text)
        evidence_ids = sorted(set(EVIDENCE_ID_REGEX.findall(text)))

        if is_injection:
            incident = {
                "source_file": source_file,
                "location": location,
                "evidence_ids": evidence_ids or ["OC-EXT-06"],
                "severity": "CRITICAL_INDIRECT_PROMPT_INJECTION",
                "matched_patterns": matched_patterns,
                "raw_malicious_payload": text,
                "action_taken": "QUARANTINED_AND_DEFANGED",
            }
            self.security_incidents.append(incident)
            # No se agrega como instrucción ni evidencia confiable al índice de recuperación
            return

        self.chunks.append({
            "chunk_id": f"{source_file}::{location}::{len(self.chunks)+1}",
            "source_file": source_file,
            "location": location,
            "evidence_ids": evidence_ids,
            "text": text,
        })

    def search_evidence(self, query: str, top_k: int = 8) -> List[Dict[str, Any]]:
        """
        Búsqueda léxica + semántica de palabras clave e identificadores de evidencia (INV-*, OC-*, REP-*).
        """
        query_terms = [w.lower() for w in re.findall(r"\w+", query) if len(w) > 2]
        scored = []
        for chunk in self.chunks:
            text_lower = chunk["text"].lower()
            score = 0.0
            for term in query_terms:
                if term in text_lower:
                    score += 2.0
            # Boost por códigos de evidencia clave
            if chunk["evidence_ids"]:
                score += 1.5
            for eid in chunk["evidence_ids"]:
                if eid.lower() in query.lower():
                    score += 10.0
            scored.append((score, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        # Si la consulta es general, retornar los chunks más ricos en evidencia
        results = [item[1] for item in scored[:top_k]]
        return results
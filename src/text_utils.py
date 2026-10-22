"""Utilidades de normalización de texto compartidas (búsqueda, firewall, validación)."""
from __future__ import annotations

import re
import unicodedata
from typing import Any, List

ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060-\u2064\ufeff\u00ad]")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# Homoglifos frecuentes (cirílico/griego) usados para evadir filtros.
HOMOGLYPHS = str.maketrans({
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x", "і": "i", "ј": "j",
    "А": "A", "Е": "E", "О": "O", "Р": "P", "С": "C", "Т": "T", "Х": "X", "І": "I", "К": "K", "М": "M", "Н": "H", "В": "B",
    "α": "a", "ο": "o", "ε": "e", "ι": "i", "ρ": "p", "ν": "v",
})
LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})

STOPWORDS = set(
    """
    a al algo algun alguna algunas alguno algunos ante antes aquel aquella como con contra cual cuales cuando de del desde
    donde dos el ella ellas ellos en entre era es esa esas ese eso esos esta estas este esto estos fue fueron ha han hay la
    las le les lo los mas me mi mis muy nada ni no nos o otra otras otro otros para pero por porque que quien se sea segun
    ser si sin sobre son su sus tal tambien te tiene tienen todo todos tu un una unas uno unos y ya
    podria podrian podemos puede pueden
    the of and or to in on for is are was were be by with from that this what which how why
    """.split()
)


def strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def clean_control_chars(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = ZERO_WIDTH_RE.sub("", text)
    return CONTROL_RE.sub(" ", text)


def fold(text: str) -> str:
    """Clave de comparación: sin acentos, minúsculas, espacios colapsados."""
    return re.sub(r"\s+", " ", strip_accents(clean_control_chars(str(text))).casefold()).strip()


def normalize_for_detection(text: str, leet: bool = False) -> str:
    t = clean_control_chars(text).translate(HOMOGLYPHS)
    t = strip_accents(t).casefold()
    if leet:
        t = t.translate(LEET)
    t = re.sub(r"[\"'“”‘’«»`*_~]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _light_stem(token: str) -> str:
    for suf in ("ciones", "cion", "mente", "es", "s"):
        if len(token) > len(suf) + 3 and token.endswith(suf):
            return token[: -len(suf)]
    return token


def tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", fold(text))
    return [_light_stem(t) for t in tokens if t not in STOPWORDS and (len(t) > 2 or any(ch.isdigit() for ch in t))]


def escape_latex_dollars(text: str) -> str:
    """
    Escapa cualquier símbolo '$' que no esté ya escapado para evitar
    que KaTeX/Streamlit lo interprete como fórmula matemática.
    """
    if not isinstance(text, str):
        return text
    # Reemplaza '$' por '\$' únicamente si no viene precedido de '\'
    return re.sub(r'(?<!\\)\$', r'\$', text)


def clean_epistemic_payload(data: Any) -> Any:
    """
    Recorre recursivamente diccionarios, listas o strings para desinfectar
    colisiones con sintaxis LaTeX.
    """
    if isinstance(data, str):
        return escape_latex_dollars(data)
    elif isinstance(data, list):
        return [clean_epistemic_payload(item) for item in data]
    elif isinstance(data, dict):
        return {k: clean_epistemic_payload(v) for k, v in data.items()}
    return data

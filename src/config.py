"""Carga de configuración centralizada (config/settings.toml + variables de entorno)."""
from __future__ import annotations

import copy
import os
import tomllib
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "settings.toml"


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def resolve_path(path: str | Path) -> Path:
    """Resuelve rutas relativas contra la raíz del proyecto (no contra el cwd)."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_config(path: Optional[str | Path] = None, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg_path = Path(path) if path else Path(os.getenv("BRIO_CONFIG", DEFAULT_CONFIG_PATH))
    with open(cfg_path, "rb") as f:
        cfg = tomllib.load(f)
    if os.getenv("GEMINI_MODEL"):
        cfg.setdefault("llm", {})["model"] = os.environ["GEMINI_MODEL"]
    return _deep_merge(cfg, overrides or {})

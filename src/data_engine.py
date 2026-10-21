"""
Motor determinista de análisis comercial y auditoría de calidad de datos.

Diseño 100% guiado por datos (cero valores de negocio harcodeados):
- Detecta dinámicamente dimensiones, métricas, meses, trimestres y mapeos operativos
  directamente desde cualquier CSV o DataFrame cargado.
- Registra en `audit_log` cada anomalía tratada y conserva en `quarantine_df` los registros
  excluidos para ejecutar análisis de sensibilidad dinámicos.
- Expone herramientas deterministas para Function Calling del agente y para el Dashboard.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.config import load_config, resolve_path
from src.text_utils import fold

PERIOD_RE = re.compile(r"^(\d{4})-(0[1-9]|1[0-2])$")
DERIVED_PRICE = "precio_unitario_promedio"
DERIVED_COUNT = "registros"

ESTADO_CORREGIDO = "CORREGIDO_INCLUIDO"
ESTADO_EXCLUIDO = "EXCLUIDO_CUARENTENA"
ESTADO_ALERTA = "CONSERVADO_CON_ALERTA"


class DataQueryError(ValueError):
    """Error de validación en parámetros de consulta sobre el motor determinista."""


def _r(val: Any, nd: int = 2) -> Optional[float]:
    if val is None:
        return None
    try:
        if pd.isna(val) or np.isinf(val):
            return None
    except TypeError:
        return None
    return round(float(val), nd)


class SalesDataEngine:
    """
    Motor determinista de análisis comercial y auditoría automática de calidad de datos.
    Funciona con `02_Ventas_2026.csv` o con cualquier otro CSV/DataFrame que siga el esquema configurable.
    """

    def __init__(
        self,
        csv_path: str | Path | pd.DataFrame = "data/02_Ventas_2026.csv",
        config: Optional[Dict[str, Any]] = None,
        source_name: Optional[str] = None,
    ):
        self.config = config or load_config()
        sch = self.config["schema"]
        self.id_col: str = sch["id_column"]
        self.period_col: str = sch["period_column"]
        self.sales_col: str = sch["sales_column"]
        self.units_col: Optional[str] = sch.get("units_column")
        self.currency: str = sch.get("currency", "COP")
        self.max_listed_values: int = int(sch.get("max_listed_values", 30))

        if isinstance(csv_path, pd.DataFrame):
            self.csv_path = Path(source_name or "uploaded_dataset.csv")
            self._initial_df: Optional[pd.DataFrame] = csv_path.copy()
        else:
            self.csv_path = resolve_path(csv_path)
            self._initial_df = None

        self.df_raw: pd.DataFrame = pd.DataFrame()
        self.df_clean: pd.DataFrame = pd.DataFrame()
        self.quarantine_df: pd.DataFrame = pd.DataFrame()
        self.audit_log: List[Dict[str, Any]] = []
        self.dimensions: List[str] = []
        self.metrics: List[str] = []
        self.consistency_maps: Dict[str, Dict[str, str]] = {}
        self._load_and_audit()

    # ------------------------------------------------------------------
    # Ingesta y auditoría automática sin reglas harcodeadas
    # ------------------------------------------------------------------
    def _detect_schema(self, raw: pd.DataFrame) -> None:
        required = [self.id_col, self.period_col, self.sales_col]
        missing = [c for c in required if c not in raw.columns]
        if missing:
            raise ValueError(f"El dataset no contiene las columnas requeridas {missing}. Columnas: {list(raw.columns)}")

        metrics: List[str] = []
        dims: List[str] = []
        for col in raw.columns:
            if col in (self.id_col, self.period_col) or col.startswith("_"):
                continue
            if col in (self.sales_col, self.units_col):
                metrics.append(col)
                continue
            non_null = raw[col].dropna()
            num_ratio = pd.to_numeric(non_null, errors="coerce").notna().mean() if len(non_null) else 0.0
            if num_ratio >= 0.9:
                metrics.append(col)
            else:
                dims.append(col)
        self.metrics = metrics
        self.dimensions = dims

    @property
    def low_card_dimensions(self) -> List[str]:
        src = self.df_clean if not self.df_clean.empty else self.df_raw
        if src.empty:
            return list(self.dimensions)
        return [d for d in self.dimensions if src[d].nunique(dropna=True) <= self.max_listed_values]

    def _upsert_audit_entry(
        self,
        row: pd.Series,
        columna: str,
        tipo_anomalia: str,
        valor_original: str,
        accion_aplicada: str,
        estado: str,
    ) -> None:
        reg_id = str(row.get(self.id_col, ""))
        fila_csv = int(row.get("_fila_csv", 0))
        linea_archivo = int(row.get("_linea_archivo", fila_csv + 2))
        marca_val = str(row.get("marca", "")).strip().title() if "marca" in row else ""
        region_val = str(row.get("region", "")).strip().title() if "region" in row else ""
        mes_val = str(row.get(self.period_col, "")).strip()

        # Si el mismo registro y columna ya tuvo una corrección de formato, consolidamos en un único hallazgo
        for existing in self.audit_log:
            if (
                existing["registro_id"] == reg_id
                and existing.get("columna") == columna
                and existing["estado"] == ESTADO_CORREGIDO
                and estado == ESTADO_CORREGIDO
            ):
                existing["tipo_anomalia"] = f"{existing['tipo_anomalia']} + {tipo_anomalia}"
                existing["accion_aplicada"] = f"{existing['accion_aplicada']} -> {accion_aplicada}"
                return

        self.audit_log.append({
            "registro_id": reg_id,
            "fila_csv": fila_csv,
            "linea_archivo": linea_archivo,
            "columna": columna,
            "marca": marca_val,
            "region": region_val,
            "mes": mes_val,
            "tipo_anomalia": tipo_anomalia,
            "valor_original": valor_original,
            "accion_aplicada": accion_aplicada,
            "estado": estado,
        })

    def _load_and_audit(self) -> None:
        if self._initial_df is not None:
            raw = self._initial_df.astype(str).replace({"nan": np.nan, "None": np.nan})
        else:
            raw = pd.read_csv(self.csv_path, dtype=str)

        self._detect_schema(raw)
        raw["_fila_csv"] = np.arange(len(raw), dtype=int)
        raw["_linea_archivo"] = raw["_fila_csv"] + 2
        if self.units_col and self.units_col in raw.columns:
            raw["unidades_raw"] = raw[self.units_col]
            raw["unidades"] = pd.to_numeric(raw[self.units_col], errors="coerce")
        raw["venta_raw"] = raw[self.sales_col]
        raw["venta_neta_cop_num"] = pd.to_numeric(raw[self.sales_col], errors="coerce")
        self.df_raw = raw.copy()

        df = raw.copy()
        self.audit_log = []
        self.consistency_maps = {}

        # 1. Normalización dinámica de espacios y capitalización hacia la forma canónica mayoritaria
        text_cols = [self.id_col, self.period_col, *self.dimensions]
        for col in text_cols:
            if col not in df.columns:
                continue
            raw_series = df[col].fillna("").astype(str)
            stripped = raw_series.str.strip()
            if col in self.dimensions and col != "punto_venta_id":
                # Inferir forma canónica por mayoría (insensible a mayúsculas/acentos)
                keys = stripped.map(fold)
                counts = (
                    pd.DataFrame({"k": keys, "v": stripped.str.title()})
                    .query("k != ''")
                    .groupby(["k", "v"])
                    .size()
                    .reset_index(name="n")
                    .sort_values(["k", "n"], ascending=[True, False])
                    .drop_duplicates("k")
                    .set_index("k")["v"]
                )
                clean_series = keys.map(counts).fillna(stripped.str.title())
            else:
                clean_series = stripped

            diff_mask = df[col].notna() & (raw_series != clean_series)
            for idx in df[diff_mask].index:
                row = df.loc[idx].copy()
                row[col] = clean_series.loc[idx]
                self._upsert_audit_entry(
                    row=row,
                    columna=col,
                    tipo_anomalia=f"Formato/ Texto en '{col}'",
                    valor_original=repr(raw_series.loc[idx]),
                    accion_aplicada=f"Normalizado a '{clean_series.loc[idx]}'",
                    estado=ESTADO_CORREGIDO,
                )
            df[col] = clean_series

        # 2. Reglas de consistencia inferidas estadísticamente de los propios datos
        min_share = float(self.config.get("analysis", {}).get("consistency_min_share", 0.95))
        for rule in self.config["schema"].get("consistency_rules", []):
            col = rule["column"]
            src_col = rule["source_column"]
            pattern = rule["pattern"]
            if col not in df.columns or src_col not in df.columns:
                continue
            codes = df[src_col].astype(str).str.extract(pattern, expand=False)
            pair_counts = (
                pd.DataFrame({"code": codes, "val": df[col]})
                .dropna()
                .groupby(["code", "val"])
                .size()
            )
            inferred_map: Dict[str, str] = {}
            if not pair_counts.empty:
                for code, grp in pair_counts.groupby(level=0):
                    top_val = grp.droplevel(0).idxmax()
                    top_count = int(grp.max())
                    total_code = int(grp.sum())
                    if top_count >= 2 and (top_count / total_code) >= min_share:
                        inferred_map[str(code)] = str(top_val)
            self.consistency_maps[f"{col}<-{src_col}"] = inferred_map

            expected = codes.map(inferred_map)
            mismatch_mask = expected.notna() & (df[col] != expected)
            for idx in df[mismatch_mask].index:
                row = df.loc[idx]
                self._upsert_audit_entry(
                    row=row,
                    columna=col,
                    tipo_anomalia=f"Inconsistencia {col.title()} vs {src_col}",
                    valor_original=f"{col}='{row[col]}' en '{row[src_col]}'",
                    accion_aplicada=f"Alineado {col} a '{expected.loc[idx]}' según prefijo operativo inferido ({src_col})",
                    estado=ESTADO_CORREGIDO,
                )
                df.loc[idx, col] = expected.loc[idx]

        quarantined_chunks: List[pd.DataFrame] = []

        def _quarantine(mask: pd.Series, motivo: str) -> None:
            nonlocal df
            if mask.any():
                q = df[mask].copy()
                q["_motivo_cuarentena"] = motivo
                quarantined_chunks.append(q)
                df = df[~mask].copy()

        # 3. Deduplicación por llave primaria registro_id (distinguiendo duplicado exacto vs conflicto de contenido)
        compare_cols = [self.period_col, *self.dimensions, "venta_raw"] + (["unidades_raw"] if "unidades_raw" in df.columns else [])
        dup_ids = df[df.duplicated(subset=[self.id_col], keep=False)][self.id_col].unique()
        exact_dup_mask = pd.Series(False, index=df.index)
        conflict_dup_mask = pd.Series(False, index=df.index)
        for rid in dup_ids:
            grp = df[df[self.id_col] == rid]
            if len(grp[compare_cols].drop_duplicates()) == 1:
                for idx in grp.index[1:]:
                    exact_dup_mask.loc[idx] = True
                    row = df.loc[idx]
                    self._upsert_audit_entry(
                        row=row,
                        columna=self.id_col,
                        tipo_anomalia="Registro ID Duplicado Exacto",
                        valor_original=f"{row[self.id_col]} (${row['venta_raw']} {self.currency})",
                        accion_aplicada="Eliminada segunda aparición para evitar doble conteo",
                        estado=ESTADO_EXCLUIDO,
                    )
            else:
                for idx in grp.index:
                    conflict_dup_mask.loc[idx] = True
                    row = df.loc[idx]
                    self._upsert_audit_entry(
                        row=row,
                        columna=self.id_col,
                        tipo_anomalia="Registro ID con Conflicto de Valores",
                        valor_original=f"{row[self.id_col]} (${row['venta_raw']} {self.currency})",
                        accion_aplicada="Aislado en cuarentena por conflicto de llave primaria con datos distintos",
                        estado=ESTADO_EXCLUIDO,
                    )
        _quarantine(exact_dup_mask, "duplicado_exacto")
        _quarantine(conflict_dup_mask, "conflicto_id")

        # 4. Validación dinámica de periodo calendario AAAA-MM (mes 01..12)
        invalid_month_mask = ~df[self.period_col].fillna("").astype(str).str.match(PERIOD_RE)
        for idx in df[invalid_month_mask].index:
            row = df.loc[idx]
            self._upsert_audit_entry(
                row=row,
                columna=self.period_col,
                tipo_anomalia="Mes Inválido Fuera de Rango Calendario",
                valor_original=f"{self.period_col}='{row[self.period_col]}' (${row['venta_raw']} {self.currency})",
                accion_aplicada="Excluido del análisis por mes calendario inexistente",
                estado=ESTADO_EXCLUIDO,
            )
        _quarantine(invalid_month_mask, "periodo_invalido")

        # 5. Valores no numéricos / faltantes en venta_neta_cop
        null_venta_mask = df["venta_neta_cop_num"].isna()
        for idx in df[null_venta_mask].index:
            row = df.loc[idx]
            u_raw = row.get("unidades_raw", "N/A")
            self._upsert_audit_entry(
                row=row,
                columna=self.sales_col,
                tipo_anomalia=f"Venta Neta Faltante ('{row['venta_raw']}')",
                valor_original=f"{self.sales_col}='{row['venta_raw']}' (unidades={u_raw})",
                accion_aplicada=f"Excluido de suma {self.currency} sin imputar artificialmente (incluido en sensibilidad)",
                estado=ESTADO_EXCLUIDO,
            )
        _quarantine(null_venta_mask, "venta_faltante")

        # 6. Valores negativos sin política de devoluciones
        neg_mask = df["venta_neta_cop_num"] < 0
        if "unidades" in df.columns:
            neg_mask = neg_mask | (df["unidades"].notna() & (df["unidades"] < 0))
        for idx in df[neg_mask].index:
            row = df.loc[idx]
            self._upsert_audit_entry(
                row=row,
                columna=self.sales_col,
                tipo_anomalia="Valores Negativos sin Política de Devolución",
                valor_original=f"unidades={row.get('unidades_raw')}, venta={row['venta_raw']}",
                accion_aplicada="Aislado en cuarentena por ausencia de política de negativos",
                estado=ESTADO_EXCLUIDO,
            )
        _quarantine(neg_mask, "valores_negativos")

        # 7. Unidades faltantes o <= 0 con venta válida
        if "unidades" in df.columns:
            null_units_mask = df["unidades"].isna() | (df["unidades"] <= 0)
            for idx in df[null_units_mask].index:
                row = df.loc[idx]
                self._upsert_audit_entry(
                    row=row,
                    columna=self.units_col or "unidades",
                    tipo_anomalia=f"Unidades Nulas (NaN) con Venta {self.currency} Válida",
                    valor_original=f"unidades={row.get('unidades_raw')}, venta=${row['venta_raw']} {self.currency}",
                    accion_aplicada=f"Conservado para análisis en {self.currency}; excluido en conteo de unidades",
                    estado=ESTADO_ALERTA,
                )

        df = self._enrich_periods(df)
        self.df_clean = df.reset_index(drop=True)
        self.quarantine_df = (
            pd.concat(quarantined_chunks, ignore_index=True)
            if quarantined_chunks
            else pd.DataFrame(columns=list(df.columns) + ["_motivo_cuarentena"])
        )

    def _enrich_periods(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        extracted = out[self.period_col].astype(str).str.extract(PERIOD_RE)
        year = extracted[0]
        month_num = pd.to_numeric(extracted[1], errors="coerce")
        q_num = ((month_num - 1) // 3 + 1).astype("Int64")
        s_num = ((month_num - 1) // 6 + 1).astype("Int64")
        out["anio"] = year
        out["trimestre"] = "T" + q_num.astype(str)
        out["anio_trimestre"] = year + "-T" + q_num.astype(str)
        out["semestre"] = year + "-S" + s_num.astype(str)
        return out

    # ------------------------------------------------------------------
    # Descubrimiento dinámico de periodos, dimensiones y métricas
    # ------------------------------------------------------------------
    @property
    def available_periods(self) -> Dict[str, List[str]]:
        if self.df_clean.empty:
            return {"meses": [], "trimestres": [], "anio_trimestres": [], "semestres": [], "anios": []}
        return {
            "meses": sorted(self.df_clean[self.period_col].dropna().unique().tolist()),
            "trimestres": sorted(self.df_clean["trimestre"].dropna().unique().tolist()),
            "anio_trimestres": sorted(self.df_clean["anio_trimestre"].dropna().unique().tolist()),
            "semestres": sorted(self.df_clean["semestre"].dropna().unique().tolist()),
            "anios": sorted(self.df_clean["anio"].dropna().unique().tolist()),
        }

    def default_comparison_quarters(self) -> Tuple[str, str]:
        """Retorna dinámicamente los dos trimestres principales disponibles en el dataset."""
        qs = self.available_periods["trimestres"]
        if len(qs) >= 2:
            return qs[0], qs[1]
        if len(qs) == 1:
            return qs[0], qs[0]
        return "T1", "T2"

    def resolve_period_months(self, period_label: str, df_ref: Optional[pd.DataFrame] = None) -> Tuple[str, List[str]]:
        """Resuelve dinámicamente etiquetas como 'T1', 'T2', '2026-T1', '2026-04', '2026-S1' o '2026'."""
        ref = df_ref if df_ref is not None else self.df_clean
        if not period_label or not str(period_label).strip():
            raise DataQueryError("El periodo no puede estar vacío.")
        raw_lbl = str(period_label).strip().upper().replace(" ", "")
        raw_lbl = raw_lbl.replace("Q", "T")

        if re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", raw_lbl):
            months = sorted(ref.loc[ref[self.period_col] == raw_lbl, self.period_col].unique().tolist())
            label = raw_lbl
        elif re.fullmatch(r"T[1-4]", raw_lbl):
            months = sorted(ref.loc[ref["trimestre"] == raw_lbl, self.period_col].unique().tolist())
            label = raw_lbl
        elif re.fullmatch(r"\d{4}-T[1-4]", raw_lbl):
            months = sorted(ref.loc[ref["anio_trimestre"] == raw_lbl, self.period_col].unique().tolist())
            label = raw_lbl
        elif re.fullmatch(r"\d{4}-S[1-2]", raw_lbl):
            months = sorted(ref.loc[ref["semestre"] == raw_lbl, self.period_col].unique().tolist())
            label = raw_lbl
        elif re.fullmatch(r"S[1-2]", raw_lbl):
            months = sorted(ref.loc[ref["semestre"].str.endswith(raw_lbl), self.period_col].unique().tolist())
            label = raw_lbl
        elif re.fullmatch(r"\d{4}", raw_lbl):
            months = sorted(ref.loc[ref["anio"] == raw_lbl, self.period_col].unique().tolist())
            label = raw_lbl
        else:
            raise DataQueryError(
                f"Periodo '{period_label}' no válido. Disponibles: {self.available_periods}"
            )

        if not months:
            raise DataQueryError(
                f"El periodo '{period_label}' no tiene registros en el dataset. Disponibles: {self.available_periods}"
            )
        return label, months

    def _normalize_metric(self, metric: Optional[str]) -> str:
        if not metric:
            return "venta_neta_cop_num"
        m = fold(metric).replace(" ", "_")
        if m in ("venta_neta_cop_num", "venta_neta_cop", "venta", "ventas", "cop", "valor"):
            return "venta_neta_cop_num"
        if m in ("unidades", "unidad", "units", "volumen"):
            return "unidades"
        if m in (DERIVED_PRICE, "precio", "precio_unitario", "ticket_promedio"):
            return DERIVED_PRICE
        if m in (DERIVED_COUNT, "conteo", "transacciones", "filas"):
            return DERIVED_COUNT
        raise DataQueryError(
            f"Métrica '{metric}' no soportada. Use: 'venta_neta_cop', 'unidades', '{DERIVED_PRICE}' o '{DERIVED_COUNT}'."
        )

    def _validate_dimensions(self, cols: Iterable[str]) -> List[str]:
        clean_cols: List[str] = []
        valid_map = {fold(c): c for c in self.dimensions + [self.period_col, "trimestre"]}
        for c in cols:
            key = fold(c)
            if key not in valid_map:
                raise DataQueryError(
                    f"Columna '{c}' no existe. Dimensiones válidas: {self.dimensions + [self.period_col, 'trimestre']}"
                )
            clean_cols.append(valid_map[key])
        return clean_cols

    def _apply_filters(
        self,
        df: pd.DataFrame,
        filters: Optional[Dict[str, Any]],
        use_clean_data: bool = True,
    ) -> pd.DataFrame:
        if not filters:
            return df
        out = df
        valid_map = {fold(c): c for c in self.dimensions + [self.period_col, "trimestre", self.id_col]}
        for raw_col, raw_val in filters.items():
            if raw_val is None or raw_val == "":
                continue
            col_key = fold(raw_col)
            if col_key not in valid_map:
                raise DataQueryError(f"Columna de filtro '{raw_col}' inválida. Disponibles: {list(valid_map.values())}")
            col = valid_map[col_key]
            vals = raw_val if isinstance(raw_val, (list, tuple, set)) else [raw_val]
            vals = [str(v) for v in vals if v is not None and str(v).strip() != "" and fold(str(v)) != "todas"]
            if not vals:
                continue
            if not use_clean_data:
                out = out[out[col].isin(vals)]
            else:
                folded_targets = {fold(v) for v in vals}
                known_folded = {fold(v) for v in self.df_clean[col].dropna().unique()}
                if folded_targets and not (folded_targets & known_folded):
                    sample = sorted(self.df_clean[col].dropna().unique().tolist())[:15]
                    raise DataQueryError(f"Valor(es) {vals} no encontrados en '{col}'. Valores disponibles: {sample}")
                out = out[out[col].astype(str).map(fold).isin(folded_targets)]
        return out

    def _compute_group_metric(self, df: pd.DataFrame, group_by: List[str], metric_col: str) -> pd.Series:
        if metric_col == DERIVED_COUNT:
            if group_by:
                return df.groupby(group_by, dropna=False).size().astype(float)
            return pd.Series({"_total": float(len(df))})
        if metric_col == DERIVED_PRICE:
            valid = df[df["unidades"].notna() & (df["unidades"] > 0)]
            if group_by:
                g = valid.groupby(group_by, dropna=False)
                return g["venta_neta_cop_num"].sum() / g["unidades"].sum()
            tot_u = float(valid["unidades"].sum())
            return pd.Series({"_total": float(valid["venta_neta_cop_num"].sum()) / tot_u if tot_u > 0 else np.nan})
        if group_by:
            return df.groupby(group_by, dropna=False)[metric_col].sum(min_count=1).fillna(0.0)
        return pd.Series({"_total": float(df[metric_col].sum())})

    def _prepare_raw_for_comparison(self) -> pd.DataFrame:
        raw = self.df_raw.copy()
        raw = raw[raw[self.period_col].fillna("").astype(str).str.match(PERIOD_RE)].copy()
        return self._enrich_periods(raw)

    # ------------------------------------------------------------------
    # Consultas deterministas universales (cualquier periodo, marca o dimensión)
    # ------------------------------------------------------------------
    def compare_periods(
        self,
        base_period: Optional[str] = None,
        comparison_period: Optional[str] = None,
        group_by: Optional[Sequence[str]] = None,
        filters: Optional[Dict[str, Any]] = None,
        metric: str = "venta_neta_cop",
        use_clean_data: bool = True,
    ) -> Dict[str, Any]:
        """Compara cualquier par de periodos (ej. T1 vs T2, 2026-01 vs 2026-06) para cualquier filtro y agrupación."""
        q1_def, q2_def = self.default_comparison_quarters()
        base_p = base_period or q1_def
        comp_p = comparison_period or q2_def
        metric_col = self._normalize_metric(metric)
        group_cols = self._validate_dimensions(group_by or [self.config["analysis"].get("default_dimension", "region")])

        source_df = self.df_clean.copy() if use_clean_data else self._prepare_raw_for_comparison()
        source_df = self._apply_filters(source_df, filters, use_clean_data=use_clean_data)

        base_lbl, base_months = self.resolve_period_months(base_p, source_df if not source_df.empty else self.df_clean)
        comp_lbl, comp_months = self.resolve_period_months(comp_p, source_df if not source_df.empty else self.df_clean)

        df_base = source_df[source_df[self.period_col].isin(base_months)]
        df_comp = source_df[source_df[self.period_col].isin(comp_months)]

        s_base = self._compute_group_metric(df_base, group_cols, metric_col)
        s_comp = self._compute_group_metric(df_comp, group_cols, metric_col)
        tot_base = float(self._compute_group_metric(df_base, [], metric_col)["_total"])
        tot_comp = float(self._compute_group_metric(df_comp, [], metric_col)["_total"])
        tot_abs = tot_comp - tot_base
        tot_pct = ((tot_comp / tot_base) - 1.0) * 100.0 if tot_base > 0 else 0.0

        combined = pd.concat([s_base.rename("base"), s_comp.rename("comp")], axis=1)
        if metric_col != DERIVED_PRICE:
            combined = combined.fillna(0.0)
        combined["variacion_absoluta"] = combined["comp"] - combined["base"]
        combined["variacion_pct"] = np.where(
            combined["base"] > 0,
            ((combined["comp"] / combined["base"]) - 1.0) * 100.0,
            np.nan,
        )
        combined = combined.sort_values(by="variacion_pct", ascending=True, na_position="last").reset_index()

        # Detectar grupos fantasma cuando se desactiva la limpieza (ej. 'Sur ' con espacio al final)
        folded_counts: Dict[Tuple[str, ...], int] = {}
        if not use_clean_data:
            for _, r in combined.iterrows():
                k = tuple(fold(str(r[c])) for c in group_cols)
                folded_counts[k] = folded_counts.get(k, 0) + 1

        filas: List[Dict[str, Any]] = []
        for rank_idx, (_, row) in enumerate(combined.iterrows(), start=1):
            vb = float(row["base"]) if pd.notna(row["base"]) else 0.0
            vc = float(row["comp"]) if pd.notna(row["comp"]) else 0.0
            v_abs = float(row["variacion_absoluta"]) if pd.notna(row["variacion_absoluta"]) else 0.0
            v_pct = _r(row["variacion_pct"])
            share_of_change = _r((v_abs / tot_abs) * 100.0) if (metric_col != DERIVED_PRICE and tot_abs != 0) else None
            entry: Dict[str, Any] = {
                **{k: row[k] for k in group_cols},
                base_lbl: _r(vb),
                comp_lbl: _r(vc),
                "T1": _r(vb),
                "T2": _r(vc),
                "variacion_absoluta": _r(v_abs),
                "variacion_pct": v_pct,
                "participacion_cambio_total_pct": share_of_change,
                "ranking": rank_idx,
            }
            if not use_clean_data:
                k = tuple(fold(str(row[c])) for c in group_cols)
                if folded_counts.get(k, 0) > 1:
                    entry["alerta_calidad"] = "Categoría fragmentada por formato/espacios sin limpiar en datos crudos"
            filas.append(entry)

        metric_public = (
            "venta_neta_cop"
            if metric_col == "venta_neta_cop_num"
            else metric_col
        )
        return {
            "periodo_base": base_lbl,
            "periodo_comparacion": comp_lbl,
            "filtros": filters or {},
            "agrupacion": group_cols,
            "metrica": metric_public,
            "datos_limpios": use_clean_data,
            "total_T1": _r(tot_base) or 0.0,
            "total_T2": _r(tot_comp) or 0.0,
            "total_base": _r(tot_base) or 0.0,
            "total_comparacion": _r(tot_comp) or 0.0,
            "total_variacion_absoluta": _r(tot_abs) or 0.0,
            "total_variacion_pct": _r(tot_pct) or 0.0,
            "filas": filas,
        }

    def compare_quarterly_sales(
        self,
        marca: str = "Brio",
        group_by: Optional[List[str]] = None,
        region_filter: Optional[str] = None,
        canal_filter: Optional[str] = None,
        presentacion_filter: Optional[str] = None,
        metric: str = "venta_neta_cop_num",
        use_clean_data: bool = True,
        base_period: Optional[str] = None,
        comparison_period: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Wrapper compatible y dinámico para comparación trimestral filtrable por marca/región/canal/presentación."""
        filters: Dict[str, Any] = {}
        if marca and fold(marca) != "todas" and "marca" in self.dimensions:
            filters["marca"] = marca
        if region_filter and "region" in self.dimensions:
            filters["region"] = region_filter
        if canal_filter and "canal" in self.dimensions:
            filters["canal"] = canal_filter
        if presentacion_filter and "presentacion" in self.dimensions:
            filters["presentacion"] = presentacion_filter

        res = self.compare_periods(
            base_period=base_period,
            comparison_period=comparison_period,
            group_by=group_by or ["region"],
            filters=filters,
            metric=metric,
            use_clean_data=use_clean_data,
        )
        res["marca"] = marca
        return res

    def query_time_series(
        self,
        group_by: Optional[Sequence[str]] = None,
        filters: Optional[Dict[str, Any]] = None,
        metric: str = "venta_neta_cop",
        granularity: str = "mes",
    ) -> Dict[str, Any]:
        """Serie temporal mensual o trimestral para cualquier filtro y agrupación."""
        metric_col = self._normalize_metric(metric)
        time_col = self.period_col if fold(granularity) in ("mes", "mensual", "month") else "trimestre"
        group_cols = self._validate_dimensions(group_by or [])
        df = self._apply_filters(self.df_clean, filters, use_clean_data=True)
        keys = group_cols + [time_col]
        s = self._compute_group_metric(df, keys, metric_col)
        rows: List[Dict[str, Any]] = []
        for idx_key, val in s.items():
            key_tuple = idx_key if isinstance(idx_key, tuple) else (idx_key,)
            row = {col: k for col, k in zip(keys, key_tuple)}
            row["valor"] = _r(val)
            rows.append(row)
        rows.sort(key=lambda r: tuple(str(r[k]) for k in keys))
        return {
            "granularidad": time_col,
            "metrica": "venta_neta_cop" if metric_col == "venta_neta_cop_num" else metric_col,
            "filtros": filters or {},
            "agrupacion": group_cols,
            "filas": rows,
        }

    def get_store_performance(
        self,
        punto_venta_ids: Optional[List[str]] = None,
        marca: str = "Brio",
        metric: str = "venta_neta_cop",
        top_n: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Consulta el desempeño exacto T1 vs T2 de puntos de venta específicos o del top N de tiendas,
        con protección contra división por cero o trimestres faltantes.
        """
        if "punto_venta_id" not in self.df_clean.columns:
            return []
        metric_col = self._normalize_metric(metric)
        q1, q2 = self.default_comparison_quarters()
        sub = self.df_clean.copy()
        if marca and fold(marca) != "todas" and "marca" in sub.columns:
            sub = sub[sub["marca"].astype(str).map(fold) == fold(marca)]
        if punto_venta_ids:
            wanted = {str(x).strip().upper() for x in punto_venta_ids if str(x).strip()}
            sub = sub[sub["punto_venta_id"].astype(str).str.upper().isin(wanted)]
        if sub.empty:
            return []

        group_cols = [c for c in ["punto_venta_id", "region", "canal", "presentacion"] if c in sub.columns]
        piv = (
            sub.groupby(group_cols + ["trimestre"], dropna=False)[
                "venta_neta_cop_num" if metric_col not in ("unidades",) else "unidades"
            ]
            .sum()
            .unstack(fill_value=0.0)
            .reset_index()
        )
        for col in [q1, q2]:
            if col not in piv.columns:
                piv[col] = 0.0

        piv["T1"] = piv[q1]
        piv["T2"] = piv[q2]
        piv["variacion_abs"] = piv["T2"] - piv["T1"]
        piv["variacion_pct"] = np.where(
            piv["T1"] > 0,
            ((piv["T2"] / piv["T1"]) - 1.0) * 100.0,
            np.nan,
        )
        piv = piv.sort_values(by="variacion_pct", ascending=True, na_position="last")
        if not punto_venta_ids:
            piv = piv.head(top_n)

        records: List[Dict[str, Any]] = []
        for _, r in piv.iterrows():
            item = {c: r[c] for c in group_cols}
            item["T1"] = _r(r["T1"]) or 0.0
            item["T2"] = _r(r["T2"]) or 0.0
            item["variacion_abs"] = _r(r["variacion_abs"]) or 0.0
            item["variacion_pct"] = _r(r["variacion_pct"])
            records.append(item)
        return records

    # ------------------------------------------------------------------
    # Análisis de sensibilidad 100% dinámico (sin registros ni unidades harcodeadas)
    # ------------------------------------------------------------------
    def get_sensitivity_analysis_all(self, group_dim: str = "region") -> List[Dict[str, Any]]:
        """
        Detecta automáticamente en `quarantine_df` todos los registros excluidos por venta faltante
        que sí tienen unidades > 0, estima su venta al precio unitario mediano de su mismo segmento
        y calcula si cambia la conclusión de caída por `group_dim`.
        """
        if self.quarantine_df.empty or "unidades" not in self.quarantine_df.columns:
            return []
        if group_dim not in self.df_clean.columns:
            group_dim = self.dimensions[0] if self.dimensions else self.period_col

        missing_sales_rows = self.quarantine_df[
            (self.quarantine_df["_motivo_cuarentena"] == "venta_faltante")
            & (self.quarantine_df["unidades"].notna())
            & (self.quarantine_df["unidades"] > 0)
        ]
        if missing_sales_rows.empty:
            return []

        q1, q2 = self.default_comparison_quarters()
        _, q1_months = self.resolve_period_months(q1)
        _, q2_months = self.resolve_period_months(q2)
        seg_cols = [c for c in ["marca", "region", "canal", "presentacion"] if c in self.df_clean.columns]
        valid_priced = self.df_clean[self.df_clean["unidades"].notna() & (self.df_clean["unidades"] > 0)]

        analyses: List[Dict[str, Any]] = []
        for _, rec in missing_sales_rows.iterrows():
            units_val = float(rec["unidades"])
            # Buscar precio unitario mediano en el segmento más específico disponible
            active_seg = list(seg_cols)
            median_price = np.nan
            while active_seg:
                mask = np.logical_and.reduce([valid_priced[c] == rec[c] for c in active_seg])
                sub = valid_priced[mask]
                if len(sub) >= 3:
                    median_price = float((sub["venta_neta_cop_num"] / sub["unidades"]).median())
                    break
                active_seg = active_seg[:-1]
            if np.isnan(median_price):
                median_price = float((valid_priced["venta_neta_cop_num"] / valid_priced["unidades"]).median())

            estimated_sale = round(units_val * median_price, 2)
            brand_val = str(rec.get("marca", "todas"))
            grp_val = str(rec.get(group_dim, ""))
            comp = self.compare_quarterly_sales(marca=brand_val, group_by=[group_dim])
            target_row = next((r for r in comp["filas"] if fold(str(r[group_dim])) == fold(grp_val)), None)
            if not target_row:
                continue

            t1_val = float(target_row["T1"])
            t2_val = float(target_row["T2"])
            rec_month = str(rec[self.period_col])
            t1_imp = t1_val + (estimated_sale if rec_month in q1_months else 0.0)
            t2_imp = t2_val + (estimated_sale if rec_month in q2_months else 0.0)
            pct_imp = ((t2_imp / t1_imp) - 1.0) * 100.0 if t1_imp > 0 else 0.0

            other_pcts = [
                float(r["variacion_pct"])
                for r in comp["filas"]
                if fold(str(r[group_dim])) != fold(grp_val) and r["variacion_pct"] is not None
            ]
            old_rank = int(target_row.get("ranking", 1))
            new_rank = 1 + sum(1 for p in other_pcts if p < pct_imp)

            analyses.append({
                "registro_id": str(rec[self.id_col]),
                "marca": brand_val,
                group_dim: grp_val,
                "mes": rec_month,
                "punto_venta_id": str(rec.get("punto_venta_id", "")),
                "unidades_registradas": int(units_val) if units_val.is_integer() else round(units_val, 2),
                "segmento_imputacion": {c: rec[c] for c in active_seg},
                "precio_unitario_mediano_cop": round(median_price, 2),
                "venta_imputada_estimada_cop": estimated_sale,
                "var_pct_sin_imputar": target_row["variacion_pct"],
                "var_pct_con_imputacion": round(pct_imp, 2),
                "ranking_sin_imputar": old_rank,
                "ranking_con_imputacion": new_rank,
                "cambia_ranking": new_rank != old_rank,
                "conclusion": (
                    f"Incluso imputando {rec[self.id_col]} (~${estimated_sale:,.0f} {self.currency}), "
                    f"la variación de {grp_val} pasaría de {target_row['variacion_pct']:.2f}% a {pct_imp:.2f}%, "
                    f"{'manteniendo su posición #' + str(new_rank) if new_rank == old_rank else 'cambiando a posición #' + str(new_rank)}."
                ),
            })
        return analyses

    def get_sensitivity_analysis_v03997(self) -> Dict[str, Any]:
        """Retorna el análisis de sensibilidad dinámico del primer registro con venta faltante (ej. V03997)."""
        all_sens = self.get_sensitivity_analysis_all()
        if all_sens:
            return all_sens[0]
        return {
            "registro_id": None,
            "unidades_registradas": 0,
            "conclusion": "No existen registros excluidos por venta faltante con unidades positivas en el dataset actual.",
        }

    def known_values(self, column: str) -> List[str]:
        """Retorna la lista ordenada de valores únicos limpios para una columna."""
        if column not in self.df_clean.columns:
            return []
        return sorted(self.df_clean[column].dropna().astype(str).unique().tolist())

    def describe_dataset(self) -> Dict[str, Any]:
        """Metadatos dinámicos del dataset para anclar al agente ante cualquier CSV."""
        dim_values: Dict[str, Any] = {}
        for d in self.dimensions:
            uniq = sorted(self.df_clean[d].dropna().astype(str).unique().tolist())
            if len(uniq) <= self.max_listed_values:
                dim_values[d] = uniq
            else:
                dim_values[d] = {"total_unicos": len(uniq), "ejemplos": uniq[:10]}
        return {
            "archivo": self.csv_path.name,
            "registros_crudos": int(len(self.df_raw)),
            "registros_limpios": int(len(self.df_clean)),
            "registros_en_cuarentena": int(len(self.quarantine_df)),
            "anomalias_detectadas": int(len(self.audit_log)),
            "dimensiones": dim_values,
            "metricas_disponibles": ["venta_neta_cop", "unidades", DERIVED_PRICE, DERIVED_COUNT],
            "periodos_disponibles": self.available_periods,
            "mapeos_consistencia_inferidos": self.consistency_maps,
        }

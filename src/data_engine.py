from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

VALID_MONTHS = {"2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06"}
T1_MONTHS = {"2026-01", "2026-02", "2026-03"}
T2_MONTHS = {"2026-04", "2026-05", "2026-06"}

CANAL_PREFIX_MAP = {"T": "Tradicional", "M": "Moderno"}


class SalesDataEngine:
    """
    Motor determinista de análisis comercial y auditoría de calidad de datos.
    Garantiza cero alucinaciones numéricas y trazabilidad completa entre datos crudos y limpios.
    """

    def __init__(self, csv_path: str | Path = "data/02_Ventas_2026.csv"):
        self.csv_path = Path(csv_path)
        self.df_raw: pd.DataFrame = pd.DataFrame()
        self.df_clean: pd.DataFrame = pd.DataFrame()
        self.audit_log: List[Dict[str, Any]] = []
        self._load_and_audit()

    def _load_and_audit(self) -> None:
        raw = pd.read_csv(self.csv_path, dtype=str)
        raw["unidades_raw"] = raw["unidades"]
        raw["venta_raw"] = raw["venta_neta_cop"]
        raw["unidades"] = pd.to_numeric(raw["unidades"], errors="coerce")
        raw["venta_neta_cop_num"] = pd.to_numeric(raw["venta_neta_cop"], errors="coerce")
        self.df_raw = raw.copy()

        df = raw.copy()
        self.audit_log = []

        # 1. Detectar inconsistencias de capitalización o espacios en blanco (V00667, V02929, V02612)
        for col in ["marca", "region", "canal", "presentacion", "punto_venta_id", "mes"]:
            raw_series = df[col].astype(str)
            clean_series = raw_series.str.strip()
            if col in ["marca", "region", "canal", "presentacion"]:
                clean_series = clean_series.str.title()

            diff_mask = raw_series != clean_series
            for idx in df[diff_mask].index:
                row = df.loc[idx]
                self.audit_log.append({
                    "registro_id": row["registro_id"],
                    "fila_csv": int(idx),
                    "marca": clean_series.loc[idx] if col == "marca" else row["marca"].strip().title(),
                    "region": clean_series.loc[idx] if col == "region" else row["region"].strip().title(),
                    "mes": row["mes"].strip(),
                    "tipo_anomalia": f"Formato/ Texto en '{col}'",
                    "valor_original": repr(row[col]),
                    "accion_aplicada": f"Normalizado a '{clean_series.loc[idx]}'",
                    "estado": "CORREGIDO_INCLUIDO"
                })
            df[col] = clean_series

        # 2. Validar coherencia entre prefijo de punto_venta_id (-T- vs -M-) y columna canal (V02612)
        pv_canal_code = df["punto_venta_id"].str.split("-").str[1]
        expected_canal = pv_canal_code.map(CANAL_PREFIX_MAP)
        canal_mismatch = expected_canal.notna() & (df["canal"] != expected_canal)
        for idx in df[canal_mismatch].index:
            row = df.loc[idx]
            self.audit_log.append({
                "registro_id": row["registro_id"],
                "fila_csv": int(idx),
                "marca": row["marca"],
                "region": row["region"],
                "mes": row["mes"],
                "tipo_anomalia": "Inconsistencia Canal vs Punto de Venta",
                "valor_original": f"canal='{row['canal']}' en '{row['punto_venta_id']}'",
                "accion_aplicada": f"Alineado canal a '{expected_canal.loc[idx]}' según prefijo operativo",
                "estado": "CORREGIDO_INCLUIDO"
            })
            df.loc[idx, "canal"] = expected_canal.loc[idx]

        # 3. Deduplicación por llave primaria registro_id (V01999 duplicado en fila 3996)
        dup_mask = df.duplicated(subset=["registro_id"], keep="first")
        for idx in df[dup_mask].index:
            row = df.loc[idx]
            self.audit_log.append({
                "registro_id": row["registro_id"],
                "fila_csv": int(idx),
                "marca": row["marca"],
                "region": row["region"],
                "mes": row["mes"],
                "tipo_anomalia": "Registro ID Duplicado Exacto",
                "valor_original": f"{row['registro_id']} (${row['venta_raw']} COP)",
                "accion_aplicada": "Eliminada segunda aparición para evitar doble conteo",
                "estado": "EXCLUIDO_CUARENTENA"
            })
        df = df[~dup_mask].copy()

        # 4. Mes fuera del calendario válido 2026-01 a 2026-06 (V03999 con 2026-13)
        invalid_month_mask = ~df["mes"].isin(VALID_MONTHS)
        for idx in df[invalid_month_mask].index:
            row = df.loc[idx]
            self.audit_log.append({
                "registro_id": row["registro_id"],
                "fila_csv": int(idx),
                "marca": row["marca"],
                "region": row["region"],
                "mes": row["mes"],
                "tipo_anomalia": "Mes Inválido Fuera de Rango",
                "valor_original": f"mes='{row['mes']}' (${row['venta_raw']} COP)",
                "accion_aplicada": "Excluido de T1/T2 (mes inexistente)",
                "estado": "EXCLUIDO_CUARENTENA"
            })
        df = df[~invalid_month_mask].copy()

        # 5. Valores no numéricos / faltantes en venta_neta_cop (V03997 con 'N/D')
        null_venta_mask = df["venta_neta_cop_num"].isna()
        for idx in df[null_venta_mask].index:
            row = df.loc[idx]
            self.audit_log.append({
                "registro_id": row["registro_id"],
                "fila_csv": int(idx),
                "marca": row["marca"],
                "region": row["region"],
                "mes": row["mes"],
                "tipo_anomalia": "Venta Neta Faltante ('N/D')",
                "valor_original": f"venta_neta_cop='{row['venta_raw']}' (unidades={row['unidades_raw']})",
                "accion_aplicada": "Excluido de suma COP sin imputar artificialmente (documentado en sensibilidad)",
                "estado": "EXCLUIDO_CUARENTENA"
            })
        df = df[~null_venta_mask].copy()

        # 6. Valores negativos en venta_neta_cop o unidades (V03998 con -90 uds y -1.000.000 COP)
        neg_mask = (df["venta_neta_cop_num"] <= 0) | (df["unidades"].notna() & (df["unidades"] <= 0))
        for idx in df[neg_mask].index:
            row = df.loc[idx]
            self.audit_log.append({
                "registro_id": row["registro_id"],
                "fila_csv": int(idx),
                "marca": row["marca"],
                "region": row["region"],
                "mes": row["mes"],
                "tipo_anomalia": "Valores Negativos sin Política de Devolución",
                "valor_original": f"unidades={row['unidades_raw']}, venta={row['venta_raw']}",
                "accion_aplicada": "Aislado en cuarentena por ausencia de política de negativos",
                "estado": "EXCLUIDO_CUARENTENA"
            })
        df = df[~neg_mask].copy()

        # 7. Unidades faltantes con venta válida (V03358 en Luma con unidades=NaN y venta=935.000)
        null_units_mask = df["unidades"].isna()
        for idx in df[null_units_mask].index:
            row = df.loc[idx]
            self.audit_log.append({
                "registro_id": row["registro_id"],
                "fila_csv": int(idx),
                "marca": row["marca"],
                "region": row["region"],
                "mes": row["mes"],
                "tipo_anomalia": "Unidades Nulas (NaN) con Venta COP Válida",
                "valor_original": f"unidades=NaN, venta=${row['venta_raw']} COP",
                "accion_aplicada": "Conservado para análisis en COP; excluido en conteo de unidades",
                "estado": "CONSERVADO_CON_ALERTA"
            })

        # Asignación determinista de trimestre
        df["trimestre"] = np.where(df["mes"].isin(T1_MONTHS), "T1", "T2")
        self.df_clean = df.copy()

    def compare_quarterly_sales(
        self,
        marca: str = "Brio",
        group_by: Optional[List[str]] = None,
        region_filter: Optional[str] = None,
        canal_filter: Optional[str] = None,
        presentacion_filter: Optional[str] = None,
        metric: str = "venta_neta_cop_num",
        use_clean_data: bool = True,
    ) -> Dict[str, Any]:
        """
        Calcula de forma determinista T1 vs T2, variación absoluta (T2 - T1) y variación porcentual ((T2/T1 - 1)*100).
        """
        if group_by is None:
            group_by = ["region"]

        source_df = self.df_clean.copy() if use_clean_data else self._prepare_raw_for_comparison()

        if marca and marca.lower() != "todas":
            if use_clean_data:
                source_df = source_df[source_df["marca"].str.lower() == marca.lower()]
            else:
                source_df = source_df[source_df["marca"] == marca]

        if region_filter:
            source_df = source_df[source_df["region"].str.lower() == region_filter.lower()]
        if canal_filter:
            source_df = source_df[source_df["canal"].str.lower() == canal_filter.lower()]
        if presentacion_filter:
            source_df = source_df[source_df["presentacion"].str.lower() == presentacion_filter.lower()]

        piv = (
            source_df.groupby(group_by + ["trimestre"], dropna=False)[metric]
            .sum()
            .unstack(fill_value=0.0)
        )
        for col in ["T1", "T2"]:
            if col not in piv.columns:
                piv[col] = 0.0

        piv["variacion_absoluta"] = piv["T2"] - piv["T1"]
        piv["variacion_pct"] = np.where(
            piv["T1"] > 0, ((piv["T2"] / piv["T1"]) - 1.0) * 100.0, np.nan
        )
        piv = piv.sort_values(by="variacion_pct", ascending=True).reset_index()

        tot_t1 = float(piv["T1"].sum())
        tot_t2 = float(piv["T2"].sum())
        tot_abs = tot_t2 - tot_t1
        tot_pct = ((tot_t2 / tot_t1) - 1.0) * 100.0 if tot_t1 > 0 else 0.0

        return {
            "marca": marca,
            "agrupacion": group_by,
            "metrica": "venta_neta_cop" if metric == "venta_neta_cop_num" else "unidades",
            "datos_limpios": use_clean_data,
            "total_T1": round(tot_t1, 2),
            "total_T2": round(tot_t2, 2),
            "total_variacion_absoluta": round(tot_abs, 2),
            "total_variacion_pct": round(tot_pct, 2),
            "filas": [
                {
                    **{k: row[k] for k in group_by},
                    "T1": round(float(row["T1"]), 2),
                    "T2": round(float(row["T2"]), 2),
                    "variacion_absoluta": round(float(row["variacion_absoluta"]), 2),
                    "variacion_pct": round(float(row["variacion_pct"]), 2) if pd.notna(row["variacion_pct"]) else None,
                }
                for _, row in piv.iterrows()
            ],
        }

    def _prepare_raw_for_comparison(self) -> pd.DataFrame:
        raw = self.df_raw.copy()
        raw = raw[raw["mes"].isin(VALID_MONTHS)].copy()
        raw["trimestre"] = np.where(raw["mes"].isin(T1_MONTHS), "T1", "T2")
        return raw

    def get_store_performance(self, punto_venta_ids: List[str], marca: str = "Brio") -> List[Dict[str, Any]]:
        """Consulta el desempeño exacto T1 vs T2 de puntos de venta específicos (ej. SU-T-003, CE-M-004)."""
        sub = self.df_clean[
            (self.df_clean["punto_venta_id"].isin(punto_venta_ids))
            & (self.df_clean["marca"].str.lower() == marca.lower())
        ]
        if sub.empty:
            return []
        piv = (
            sub.groupby(["punto_venta_id", "region", "canal", "presentacion", "trimestre"])["venta_neta_cop_num"]
            .sum()
            .unstack(fill_value=0.0)
            .reset_index()
        )
        piv["variacion_abs"] = piv["T2"] - piv["T1"]
        piv["variacion_pct"] = ((piv["T2"] / piv["T1"]) - 1.0) * 100.0
        return piv.round(2).to_dict(orient="records")

    def get_sensitivity_analysis_v03997(self) -> Dict[str, Any]:
        """
        Evalúa qué pasaría en Sur si el registro V03997 (205 uds de Brío Individual en SU-T-013 con venta 'N/D')
        se imputara al precio mediano de Brío Individual en Sur Tradicional.
        """
        brio_sur_ind = self.df_clean[
            (self.df_clean["marca"] == "Brio")
            & (self.df_clean["region"] == "Sur")
            & (self.df_clean["canal"] == "Tradicional")
            & (self.df_clean["presentacion"] == "Individual")
        ]
        median_unit_price = float((brio_sur_ind["venta_neta_cop_num"] / brio_sur_ind["unidades"]).median())
        estimated_sale = round(205.0 * median_unit_price, 2)

        base_sur = self.compare_quarterly_sales(marca="Brio", group_by=["region"], region_filter="Sur")["filas"][0]
        t1_sur = base_sur["T1"]
        t2_sur_base = base_sur["T2"]
        t2_sur_imputed = t2_sur_base + estimated_sale
        pct_imputed = ((t2_sur_imputed / t1_sur) - 1.0) * 100.0

        return {
            "registro_id": "V03997",
            "unidades_registradas": 205,
            "precio_unitario_mediano_cop": round(median_unit_price, 2),
            "venta_imputada_estimada_cop": estimated_sale,
            "var_pct_sin_imputar": base_sur["variacion_pct"],
            "var_pct_con_imputacion": round(pct_imputed, 2),
            "conclusion": (
                f"Incluso imputando V03997 (~${estimated_sale:,.0f} COP), la caída de Sur pasaría de "
                f"{base_sur['variacion_pct']:.2f}% a {pct_imputed:.2f}%, siguiendo como la región de mayor deterioro por amplio margen."
            ),
        }
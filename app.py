import io
import os
import streamlit as st
import pandas as pd
import plotly.express as px

from src.data_engine import SalesDataEngine
from src.doc_engine import DocumentSecurityEngine
from src.agent import BrioEpistemicAgent
from src.llmops import LLMOpsTracer
from src.security import RateLimitExceeded, SessionRateLimiter, UnsafeInputError, redact_secrets

st.set_page_config(
    page_title="Almena Foods | Brío Epistemic AI Analyst",
    page_icon="🛡️",
    layout="wide",
)


@st.cache_resource
def init_default_engines():
    data_eng = SalesDataEngine("data/02_Ventas_2026.csv")
    doc_eng = DocumentSecurityEngine("data")
    tracer = LLMOpsTracer()
    return data_eng, doc_eng, tracer


base_data_engine, base_doc_engine, llmops_tracer = init_default_engines()

if "session_rate_limiter" not in st.session_state:
    st.session_state.session_rate_limiter = SessionRateLimiter(per_minute=6, max_total=40)

if "custom_data_engine" not in st.session_state:
    st.session_state.custom_data_engine = None

if "custom_doc_engine" not in st.session_state:
    st.session_state.custom_doc_engine = None

data_engine: SalesDataEngine = st.session_state.custom_data_engine or base_data_engine
doc_engine: DocumentSecurityEngine = st.session_state.custom_doc_engine or base_doc_engine
agent = BrioEpistemicAgent(
    data_engine=data_engine,
    doc_engine=doc_engine,
    tracer=llmops_tracer,
    rate_limiter=st.session_state.session_rate_limiter,
)

# Barra lateral de configuración, seguridad y carga dinámica de fuentes
with st.sidebar:
    st.markdown("## 🛡️ Almena AI Governance")
    st.caption(f"Copiloto Neuro-Simbólico de Análisis Comercial — Fuente activa: `{data_engine.csv_path.name}`")

    env_key = os.getenv("GEMINI_API_KEY", "")
    try:
        if not env_key and "GEMINI_API_KEY" in st.secrets:
            env_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass

    api_key_input = st.text_input(
        "🔑 Gemini API Key (opcional si ya está en Secrets/.env)",
        value="",
        type="password",
        help="Protegida con validación de formato, redacción en logs y límite de tasa por sesión.",
    )
    effective_api_key = api_key_input.strip() or env_key

    st.divider()
    st.markdown("### 🚦 Estado Dinámico del Sistema")
    st.success(
        f"📊 CSV Verificado (`{data_engine.csv_path.name}`): {len(data_engine.df_clean):,} registros válidos / "
        f"{len(data_engine.audit_log)} anomalías tratadas ({len(data_engine.quarantine_df)} en cuarentena)"
    )
    inc_ids = sorted({eid for inc in doc_engine.security_incidents for eid in inc.get("evidence_ids", [])})
    st.error(
        f"🛡️ Firewall Activo: {len(doc_engine.security_incidents)} ataque(s) de Prompt Injection neutralizado(s) "
        f"({', '.join(inc_ids) if inc_ids else 'ninguno'})"
    )
    st.info(
        f"📚 Evidencia Indexada (BM25): {len(doc_engine.chunks)} fragmentos en "
        f"{len(doc_engine.list_sources_summary())} documentos"
    )
    limiter = st.session_state.session_rate_limiter
    st.caption(
        f"🔒 Cuota de Sesión: {limiter.total}/{limiter.max_total} consultas usadas "
        f"(máx. {limiter.per_minute}/min · máx. {agent.max_question_chars} caracteres/pregunta)"
    )

    with st.expander("📂 Cargar Otros Datos o Fuentes (Dinámico)"):
        st.caption("Sube un CSV alternativo o documentos adicionales (.md, .txt, .pdf, .docx) para probar el motor sin código a mano.")
        uploaded_csv = st.file_uploader("Reemplazar CSV de ventas", type=["csv"], key="up_csv")
        if uploaded_csv is not None:
            try:
                df_up = pd.read_csv(io.BytesIO(uploaded_csv.getvalue()), dtype=str)
                st.session_state.custom_data_engine = SalesDataEngine(df_up, source_name=uploaded_csv.name)
                st.success(f"CSV `{uploaded_csv.name}` cargado y auditado.")
                st.rerun()
            except Exception as exc:
                st.error(f"Error al cargar CSV: {redact_secrets(str(exc))}")

        uploaded_docs = st.file_uploader(
            "Agregar documentos (.md, .txt, .pdf, .docx)",
            type=["md", "txt", "pdf", "docx"],
            accept_multiple_files=True,
            key="up_docs",
        )
        if uploaded_docs and st.button("Indexar documentos subidos", use_container_width=True):
            new_doc_eng = DocumentSecurityEngine("data")
            for udoc in uploaded_docs:
                new_doc_eng.ingest_file(udoc.name, raw_bytes=udoc.getvalue(), rebuild_index=False)
            new_doc_eng._rebuild_bm25_index()
            st.session_state.custom_doc_engine = new_doc_eng
            st.success(f"{len(uploaded_docs)} documento(s) escaneado(s) e indexado(s).")
            st.rerun()

        if (st.session_state.custom_data_engine is not None or st.session_state.custom_doc_engine is not None) and st.button(
            "Restaurar datos originales", use_container_width=True
        ):
            st.session_state.custom_data_engine = None
            st.session_state.custom_doc_engine = None
            st.rerun()

st.title("🧠 Almena Foods — Analista IA Epistémico")
doc_names = ", ".join(f"`{s['archivo']}`" for s in doc_engine.list_sources_summary())
st.markdown(
    f"Integra **cálculo determinista verificable** sobre `{data_engine.csv_path.name}` con **evidencia documental sanitizada** "
    f"({doc_names}), distinguiendo rigurosamente **hechos comprobados, hipótesis e información insuficiente** con **verificación automática anti-alucinación**."
)

tab_chat, tab_dashboard, tab_audit, tab_llmops = st.tabs([
    "💬 1. Consultor Epistémico (IA + Function Calling)",
    "📊 2. Tablero Dinámico vs. Reporte Comercial",
    "🧹 3. Auditoría Forense (CSV & Firewall Quirúrgico)",
    "⚙️ 4. Observabilidad LLMOps & Grounding",
])

REFERENCE_QUESTIONS = [
    "¿Cuál fue la región con mayor caída porcentual en ventas netas de Brío entre T1 y T2 de 2026? Muestra los valores y el cálculo.",
    "¿Qué hallazgos de consumidor y observaciones comerciales podrían ayudar a explicar el deterioro de Brío?",
    "¿Son consistentes las principales conclusiones del reporte comercial con los datos de ventas?",
    "¿Podemos afirmar que las promociones de la competencia causaron la caída de ventas? ¿Qué evidencia lo sustenta o limita?",
]

with tab_chat:
    st.markdown("#### ⚡ Consultas de Referencia (1 Clic) o Pregunta Libre sobre Cualquier Marca/Mes/Métrica")
    cols = st.columns(2)
    selected_preset = None
    for i, rq in enumerate(REFERENCE_QUESTIONS):
        if cols[i % 2].button(f"📌 Consulta {i+1}: {rq[:72]}...", key=f"btn_rq_{i}", use_container_width=True):
            selected_preset = rq

    user_query = st.chat_input(
        "Pregunta por cualquier marca (Brío, Luma, Nativa, Senda, Rizo), mes, región, canal, unidades o documento..."
    )
    active_query = selected_preset or user_query

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if active_query:
        with st.spinner("Validando seguridad + ejecutando motor determinista + verificando grounding cero-alucinaciones..."):
            try:
                result = agent.ask(
                    active_query,
                    api_key_override=effective_api_key,
                    chat_history=st.session_state.chat_history,
                )
                st.session_state.chat_history.insert(0, {"query": active_query, "result": result})
            except UnsafeInputError as sec_err:
                st.error(f"🛡️ Consulta bloqueada por el Firewall de Entrada: {sec_err}")
            except RateLimitExceeded as rate_err:
                st.warning(f"⏱️ Control de uso de API Key: {rate_err}")
            except ValueError as val_err:
                st.warning(f"⚠️ Configuración requerida: {redact_secrets(str(val_err))}")
            except Exception as err:
                st.error(f"Error al procesar la consulta: {redact_secrets(str(err))}")

    for item in st.session_state.chat_history:
        q = item["query"]
        res = item["result"]["response"]
        tr = item["result"]["trace"]
        gr = item["result"].get("grounding", tr.get("grounding_report", {}))

        with st.container(border=True):
            st.markdown(f"### 🧑 Analista: *{q}*")
            if gr:
                g_score = gr.get("grounding_score_pct", 100.0)
                st.caption(
                    f"🛡️ **Grounding Verificado:** `{g_score}%` · "
                    f"Cifras comprobadas: `{gr.get('verified_numbers_count', 0)}` · "
                    f"Citas verificadas: `{len(gr.get('verified_citations', []))}`"
                )
            st.info(f"**🎯 Síntesis Ejecutiva:**\n\n{res['resumen_ejecutivo']}")

            c1, c2 = st.columns(2)
            with c1:
                st.success("**✅ Hechos Comprobados (Datos Verificados)**")
                for h in res.get("hechos_comprobados", []):
                    st.markdown(f"- {h}")

                st.warning("**⚠️ Contradicciones y Alertas de Calidad/Seguridad**")
                for c in res.get("contradicciones_y_alertas", []):
                    st.markdown(f"- {c}")

            with c2:
                st.info("**💡 Hipótesis Cualitativas (Indicios de Campo/Consumidor)**")
                for hip in res.get("hipotesis_cualitativas", []):
                    st.markdown(f"- {hip}")

                st.error("**🛑 Límites Metodológicos y Evidencia Insuficiente**")
                for lim in res.get("limites_y_evidencia_insuficiente", []):
                    st.markdown(f"- {lim}")

            st.caption(f"📚 **Fuentes y Códigos de Evidencia Citados:** `{', '.join(res.get('fuentes_citadas', []))}`")

            with st.expander(
                f"🔍 Inspeccionar Trazabilidad Técnica y LLMOps (Trace ID: {tr['trace_id']} | {tr['latency_ms']} ms)"
            ):
                st.json(tr)

with tab_dashboard:
    st.markdown(f"### 📊 Contraste Dinámico Verificado: `{data_engine.csv_path.name}` vs. Reportes Documentales")

    ctrl1, ctrl2, ctrl3 = st.columns([1.2, 1.2, 2.0])
    available_brands = data_engine.known_values("marca")
    default_b_idx = available_brands.index("Brio") if "Brio" in available_brands else 0
    selected_brand = ctrl1.selectbox(
        "Marca a analizar",
        options=available_brands + ["todas"] if available_brands else ["todas"],
        index=default_b_idx,
    )
    selected_metric = ctrl2.selectbox(
        "Métrica",
        options=["venta_neta_cop", "unidades", "precio_unitario_promedio"],
        index=0,
    )
    use_clean = ctrl3.toggle(
        "Aplicar Pipeline de Limpieza de Datos (Desactivar para ver distorsión con datos crudos)",
        value=True,
    )

    comp = data_engine.compare_quarterly_sales(
        marca=selected_brand,
        group_by=["region"],
        metric=selected_metric,
        use_clean_data=use_clean,
    )
    df_reg = pd.DataFrame(comp["filas"])

    # Extracción 100% dinámica de las variaciones preliminares reportadas en los documentos
    valid_regions = data_engine.known_values("region")
    report_claims, report_source = doc_engine.extract_reported_variations(valid_regions)

    if not df_reg.empty and report_claims and selected_brand.lower() == "brio" and selected_metric == "venta_neta_cop":
        df_reg["reportado_en_pdf_pct"] = df_reg["region"].map(report_claims)
        df_reg["discrepancia_pp"] = (df_reg["variacion_pct"] - df_reg["reportado_en_pdf_pct"]).round(2)

    valid_rows = [r for r in comp["filas"] if r.get("variacion_pct") is not None]
    worst_row = valid_rows[0] if valid_rows else None
    ghost_rows = [r for r in comp["filas"] if r.get("alerta_calidad") or r.get("variacion_pct") is None]

    unit_prefix = "$" if selected_metric in ("venta_neta_cop", "precio_unitario_promedio") else ""
    unit_suffix = f" {data_engine.currency}" if selected_metric in ("venta_neta_cop", "precio_unitario_promedio") else " uds"
    decimals = ",.0f" if selected_metric != "precio_unitario_promedio" else ",.2f"

    k1, k2, k3, k4 = st.columns(4)
    k1.metric(
        f"{comp['periodo_base']} ({selected_brand})",
        f"{unit_prefix}{comp['total_T1']:{decimals}}{unit_suffix}",
    )
    k2.metric(
        f"{comp['periodo_comparacion']} ({selected_brand})",
        f"{unit_prefix}{comp['total_T2']:{decimals}}{unit_suffix}",
        f"{comp['total_variacion_pct']:.2f}%",
    )
    if worst_row:
        share_pct = worst_row.get("participacion_cambio_total_pct")
        share_label = f"Explica {abs(share_pct):.1f}% del cambio neto" if share_pct is not None else "Mayor contracción"
        k3.metric(
            "Región Mayor Caída Real (CSV)",
            f"{worst_row['region']} ({worst_row['variacion_pct']:.2f}%)",
            share_label,
            delta_color="inverse",
        )
    if report_claims and selected_brand.lower() == "brio":
        rep_worst_reg, rep_worst_val = min(report_claims.items(), key=lambda kv: kv[1])
        real_match = next((r for r in valid_rows if r["region"] == rep_worst_reg), None)
        real_pct_str = f"Real en CSV: {real_match['variacion_pct']:.2f}%" if real_match else "Sin cruce"
        k4.metric(
            f"Mayor Caída según `{report_source or 'Reporte'}`",
            f"{rep_worst_reg} ({rep_worst_val:.1f}%)",
            real_pct_str,
            delta_color="off",
        )
    else:
        k4.metric("Registros Auditados Activos", f"{len(data_engine.df_clean):,}", f"{len(data_engine.audit_log)} anomalías tratadas")

    if not use_clean and ghost_rows:
        st.warning(
            "⚠️ **Distorsión por Datos Crudos Detectada:** Al desactivar el pipeline de limpieza aparecen filas fragmentadas "
            f"(ej. `{', '.join(repr(r['region']) for r in ghost_rows)}`) y registros sin normalizar que alteran las variaciones reales."
        )

    st.dataframe(df_reg, use_container_width=True)

    if not df_reg.empty:
        plot_cols = ["variacion_pct"] + (["reportado_en_pdf_pct"] if "reportado_en_pdf_pct" in df_reg.columns else [])
        fig = px.bar(
            df_reg.dropna(subset=["variacion_pct"]).melt(
                id_vars=["region"],
                value_vars=plot_cols,
                var_name="Fuente",
                value_name="Variación %",
            ),
            x="region",
            y="Variación %",
            color="Fuente",
            barmode="group",
            text_auto=".2f",
            title=f"Variación % {comp['periodo_comparacion']} vs {comp['periodo_base']} por Región ({selected_brand} · {selected_metric})",
        )
        st.plotly_chart(fig, use_container_width=True)

    c_can, c_pres = st.columns(2)
    with c_can:
        st.markdown(f"#### Desglose por Región y Canal ({selected_brand})")
        st.dataframe(
            pd.DataFrame(
                data_engine.compare_quarterly_sales(
                    selected_brand, ["region", "canal"], metric=selected_metric, use_clean_data=use_clean
                )["filas"]
            ),
            use_container_width=True,
        )
    with c_pres:
        st.markdown(f"#### Desglose por Región y Presentación ({selected_brand})")
        st.dataframe(
            pd.DataFrame(
                data_engine.compare_quarterly_sales(
                    selected_brand, ["region", "presentacion"], metric=selected_metric, use_clean_data=use_clean
                )["filas"]
            ),
            use_container_width=True,
        )

    with st.expander("📈 Ver Evolución Mensual Detallada"):
        ts_data = data_engine.query_time_series(
            group_by=["region"],
            filters={"marca": selected_brand} if selected_brand != "todas" else {},
            metric=selected_metric,
            granularity="mes",
        )
        df_ts = pd.DataFrame(ts_data["filas"])
        if not df_ts.empty:
            fig_ts = px.line(
                df_ts,
                x="mes",
                y="valor",
                color="region",
                markers=True,
                title=f"Evolución Mensual de {selected_metric} por Región ({selected_brand})",
            )
            st.plotly_chart(fig_ts, use_container_width=True)

with tab_audit:
    st.markdown("### 🛡️ 1. Incidentes de Seguridad Neutralizados (*Indirect Prompt Injection*)")
    for inc in doc_engine.security_incidents:
        st.error(
            f"**Amenaza Bloqueada en `{inc['source_file']}` · `{inc['location']}` "
            f"(Código `{', '.join(inc['evidence_ids'])}` | Patrones: `{', '.join(inc['matched_patterns'])}`):**\n\n"
            f"**Acción:** `{inc['action_taken']}` — Aislado quirúrgicamente sin descartar los párrafos legítimos adyacentes "
            f"y omitido del prompt del LLM."
        )
        st.code(inc["raw_malicious_payload"], language="text")

    st.divider()
    st.markdown(
        f"### 🧹 2. Bitácora Dinámica de las {len(data_engine.audit_log)} Anomalías Tratadas en `{data_engine.csv_path.name}`"
    )
    st.dataframe(pd.DataFrame(data_engine.audit_log), use_container_width=True)

    st.markdown("### 🔬 3. Análisis de Sensibilidad Dinámico (Registros con Venta Faltante en Cuarentena)")
    all_sens = data_engine.get_sensitivity_analysis_all()
    st.json(all_sens if len(all_sens) != 1 else all_sens[0])

with tab_llmops:
    arch_path = os.path.join(os.path.dirname(__file__), "docs", "arquitectura_interactiva.html")
    if os.path.exists(arch_path):
        with st.expander("🗺️ Explorar Diagrama de Arquitectura Interactiva (Vista Negocio y Vista Técnica)", expanded=True):
            with open(arch_path, "r", encoding="utf-8") as f_arch:
                st.components.v1.html(f_arch.read(), height=560, scrolling=True)

    st.markdown("### ⚙️ Registro de Trazas en Tiempo Real y Verificación de Grounding (LLMOps)")
    if not llmops_tracer.traces:
        st.info("Aún no se han ejecutado consultas en esta sesión. Haz una pregunta en la Pestaña 1 para ver la telemetría.")
    else:
        st.dataframe(pd.DataFrame(llmops_tracer.traces), use_container_width=True)

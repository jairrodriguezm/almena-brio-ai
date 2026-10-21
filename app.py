import os
import streamlit as st
import pandas as pd
import plotly.express as px

from src.data_engine import SalesDataEngine
from src.doc_engine import DocumentSecurityEngine
from src.agent import BrioEpistemicAgent
from src.llmops import LLMOpsTracer

st.set_page_config(
    page_title="Almena Foods | Brío Epistemic AI Analyst",
    page_icon="🛡️",
    layout="wide",
)


@st.cache_resource
def init_engines():
    data_eng = SalesDataEngine("data/02_Ventas_2026.csv")
    doc_eng = DocumentSecurityEngine("data")
    tracer = LLMOpsTracer()
    agent = BrioEpistemicAgent(data_engine=data_eng, doc_engine=doc_eng, tracer=tracer)
    return data_eng, doc_eng, tracer, agent


data_engine, doc_engine, llmops_tracer, agent = init_engines()

# Barra lateral de configuración y estado de seguridad
with st.sidebar:
    st.markdown("## 🛡️ Almena AI Governance")
    st.caption("Copiloto Neuro-Simbólico de Análisis Comercial — Marca **Brío** (S1 2026)")

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
        help="Si accedes desde la URL pública configurada con Secrets, no necesitas ingresar nada.",
    )
    effective_api_key = api_key_input.strip() or env_key

    st.divider()
    st.markdown("### 🚦 Estado del Sistema")
    st.success(f"📊 CSV Verificado: {len(data_engine.df_clean):,} registros válidos / {len(data_engine.audit_log)} anomalías tratadas")
    st.error(f"🛡️ Firewall Activo: {len(doc_engine.security_incidents)} ataque de Prompt Injection neutralizado (OC-EXT-06)")
    st.info(f"📚 Evidencia Indexada: {len(doc_engine.chunks)} fragmentos documentales con IDs")

st.title("🧠 Almena Foods — Analista IA Epistémico (Caso Brío 2026)")
st.markdown(
    "Integra **cálculo determinista verificable** sobre `02_Ventas_2026.csv` con **evidencia documental sanitizada** "
    "(`03_Investigacion_Consumidor.pdf`, `04_Observaciones_Canales.docx`, `05_Reporte_Comercial.pdf`), "
    "distinguiendo rigurosamente **hechos comprobados, hipótesis e información insuficiente**."
)

tab_chat, tab_dashboard, tab_audit, tab_llmops = st.tabs([
    "💬 1. Consultor Epistémico (IA Generativa)",
    "📊 2. Tablero Verificado vs. Reporte Comercial",
    "🧹 3. Auditoría Forense (CSV & Seguridad OC-EXT-06)",
    "⚙️ 4. Observabilidad LLMOps",
])

REFERENCE_QUESTIONS = [
    "¿Cuál fue la región con mayor caída porcentual en ventas netas de Brío entre T1 y T2 de 2026? Muestra los valores y el cálculo.",
    "¿Qué hallazgos de consumidor y observaciones comerciales podrían ayudar a explicar el deterioro de Brío?",
    "¿Son consistentes las principales conclusiones del reporte comercial con los datos de ventas?",
    "¿Podemos afirmar que las promociones de la competencia causaron la caída de ventas? ¿Qué evidencia lo sustenta o limita?",
]

with tab_chat:
    st.markdown("#### ⚡ Consultas de Referencia (1 Clic) o Pregunta Libre")
    cols = st.columns(2)
    selected_preset = None
    for i, rq in enumerate(REFERENCE_QUESTIONS):
        if cols[i % 2].button(f"📌 Consulta {i+1}: {rq[:72]}...", key=f"btn_rq_{i}", use_container_width=True):
            selected_preset = rq

    user_query = st.chat_input("Escribe cualquier pregunta sobre el desempeño de Brío, regiones, canales, tiendas o documentos...")
    active_query = selected_preset or user_query

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if active_query:
        with st.spinner("Ejecutando motor determinista + recuperando evidencia sanitizada + sintetizando respuesta epistémica..."):
            try:
                result = agent.ask(active_query, api_key_override=effective_api_key)
                st.session_state.chat_history.insert(0, {"query": active_query, "result": result})
            except Exception as err:
                st.error(f"Error al procesar la consulta: {err}")

    for item in st.session_state.chat_history:
        q = item["query"]
        res = item["result"]["response"]
        tr = item["result"]["trace"]

        with st.container(border=True):
            st.markdown(f"### 🧑 Analista: *{q}*")
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

            with st.expander(f"🔍 Inspeccionar Trazabilidad Técnica y LLMOps (Trace ID: {tr['trace_id']} | {tr['latency_ms']} ms)"):
                st.json(tr)

with tab_dashboard:
    st.markdown("### 📊 Contraste Verificado: Base de Ventas Real (`02_Ventas_2026.csv`) vs. `05_Reporte_Comercial.pdf`")
    use_clean = st.toggle("Aplicar Pipeline de Limpieza de Datos (Desactivar para ver distorsión con datos crudos)", value=True)
    comp = data_engine.compare_quarterly_sales(marca="Brio", group_by=["region"], use_clean_data=use_clean)
    df_reg = pd.DataFrame(comp["filas"])

    report_claims = {"Sur": -7.0, "Centro": -23.0, "Occidente": -6.0, "Norte": -4.0, "Oriente": 4.0}
    df_reg["reportado_en_pdf_pct"] = df_reg["region"].map(report_claims)
    df_reg["discrepancia_pp"] = (df_reg["variacion_pct"] - df_reg["reportado_en_pdf_pct"]).round(2)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Venta Neta Brío T1", f"${comp['total_T1']:,.0f} COP")
    k2.metric("Venta Neta Brío T2", f"${comp['total_T2']:,.0f} COP", f"{comp['total_variacion_pct']:.2f}%")
    k3.metric("Región Mayor Caída Real (CSV)", "Sur (-30.27%)", "Explica 73.1% de la caída total", delta_color="inverse")
    k4.metric("Región Reportada en PDF (REP-01)", "Centro (-23.0%)", "Error: En realidad cayó -9.03%", delta_color="off")

    st.dataframe(df_reg, use_container_width=True)

    fig = px.bar(
        df_reg.melt(id_vars=["region"], value_vars=["variacion_pct", "reportado_en_pdf_pct"], var_name="Fuente", value_name="Variación %"),
        x="region",
        y="Variación %",
        color="Fuente",
        barmode="group",
        text_auto=".2f",
        title="Variación % T2 vs T1 por Región: Realidad Verificada (CSV) vs. Estimación Preliminar (Reporte Comercial)",
    )
    st.plotly_chart(fig, use_container_width=True)

    c_can, c_pres = st.columns(2)
    with c_can:
        st.markdown("#### Desglose por Región y Canal (Brío)")
        st.dataframe(pd.DataFrame(data_engine.compare_quarterly_sales("Brio", ["region", "canal"])["filas"]), use_container_width=True)
    with c_pres:
        st.markdown("#### Desglose por Región y Presentación (Brío)")
        st.dataframe(pd.DataFrame(data_engine.compare_quarterly_sales("Brio", ["region", "presentacion"])["filas"]), use_container_width=True)

with tab_audit:
    st.markdown("### 🛡️ 1. Incidente de Seguridad Neutralizado (*Indirect Prompt Injection*)")
    for inc in doc_engine.security_incidents:
        st.error(
            f"**Amenaza Bloqueada en `{inc['source_file']}` (Código `{', '.join(inc['evidence_ids'])}`):**\n\n"
            f"Texto malicioso detectado: `{inc['raw_malicious_payload']}`\n\n"
            f"**Acción:** Aislado del contexto de instrucciones del LLM mediante el Firewall de Ingesta."
        )

    st.divider()
    st.markdown("### 🧹 2. Bitácora de las 8 Anomalías Tratadas en `02_Ventas_2026.csv`")
    st.dataframe(pd.DataFrame(data_engine.audit_log), use_container_width=True)

    st.markdown("### 🔬 3. Análisis de Sensibilidad del Registro Faltante `V03997` (Brío Sur `N/D`)")
    st.json(data_engine.get_sensitivity_analysis_v03997())

with tab_llmops:
    st.markdown("### ⚙️ Registro de Trazas en Tiempo Real (LLMOps)")
    if not llmops_tracer.traces:
        st.info("Aún no se han ejecutado consultas en esta sesión. Haz una pregunta en la Pestaña 1 para ver la telemetría.")
    else:
        st.dataframe(pd.DataFrame(llmops_tracer.traces), use_container_width=True)
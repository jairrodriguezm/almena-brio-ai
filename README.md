# 🛡️ Almena Foods — Copiloto Analítico Epistémico (Caso Brío 2026)

Aplicación de Inteligencia Artificial Neuro-Simbólica para el análisis comercial de **Brío** en el primer semestre de 2026. Integra cálculo determinista sobre registros transaccionales (`02_Ventas_2026.csv`) y recuperación documental segura sobre estudios cualitativos y reportes (`PDF`, `DOCX`, `Markdown`).

## 🏗️ 1. Arquitectura y Decisiones Principales

1. **Separación Neuro-Simbólica (Cero Alucinaciones Numéricas):**
   - El LLM (`gemini-2.5-flash`) **jamás realiza operaciones aritméticas mentales**. Todo indicador financiero, agregación por trimestre/región/canal/presentación y análisis de sensibilidad es calculado de forma determinista por `SalesDataEngine` (`src/data_engine.py`).
2. **Auditoría Automática de Calidad de Datos (8 Anomalías Tratadas):**
   - Normalización de strings: `V00667` (`'BRIO'` -> `'Brio'`), `V02929` (`'Sur '` -> `'Sur'`) y `V02612` (`'Moderno '` en punto tradicional `OC-T-001` alineado a `'Tradicional'`).
   - Deduplicación por llave primaria `registro_id`: eliminación de la segunda aparición de `V01999` (fila 3996).
   - Cuarentena de registros inválidos: `V03999` (mes inexistente `2026-13`), `V03997` (venta neta `"N/D"` en `SU-T-013`, con análisis de sensibilidad explícito) y `V03998` (valores negativos `-90` uds / `-$1.000.000 COP` en Luma).
3. **Defensa contra Indirect Prompt Injection (`OC-EXT-06`):**
   - El archivo `04_Observaciones_Canales.docx` contiene una instrucción maliciosa en `OC-EXT-06` que ordena ignorar el CSV y declarar a Centro como la región de mayor caída.
   - `DocumentSecurityEngine` (`src/doc_engine.py`) escanea cada bloque en la ingesta, pone en cuarentena `OC-EXT-06`, aplica *Spotlighting* (`<untrusted_document_evidence>`) y reporta el incidente en la interfaz.
4. **Salida Estructurada Epistémica (`Pydantic`):**
   - Fuerza por contrato (`EpistemicAnalysisResponse`) la separación entre: (1) *Hechos Comprobados*, (2) *Hipótesis Cualitativas*, (3) *Contradicciones y Alertas* y (4) *Límites Metodológicos / Evidencia Insuficiente*.

## 🚀 2. Ejecución Local y Pruebas Automatizadas

```bash
# 1. Crear entorno e instalar dependencias
python -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Configurar credencial gratuita de Google AI Studio
cp .env.example .env
# Editar .env y agregar GEMINI_API_KEY=...

# 3. Ejecutar suite de pruebas automatizadas (Pytest)
pytest -v

# 4. Iniciar la aplicación web
streamlit run app.py
import streamlit as st
import polars as pl
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from pathlib import Path

# ─── Config ────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Dashboard IPC vs IPV · España", page_icon="🏠", layout="wide", initial_sidebar_state="expanded")

BASE_DIR = Path(__file__).parent
DATA_DIR, VIZ_DIR = BASE_DIR / "data_output", BASE_DIR / "visualizations"


# ─── Utilidades ────────────────────────────────────────────────────────────────

# Carga el archivo main.css e inyecta los estilos en la página de Streamlit
def load_css():
    css = (BASE_DIR / "main.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# Extrae un fragmento HTML del archivo main.html por su id y lo renderiza como tarjeta informativa
def load_html_snippet(snippet_id):
    html = (BASE_DIR / "main.html").read_text(encoding="utf-8")
    import re
    m = re.search(rf'<div id="{snippet_id}"[^>]*style="display:none"[^>]*>(.*?)</div>', html, re.DOTALL)
    if m:
        st.markdown(f'<div class="section-card">{m.group(1)}</div>', unsafe_allow_html=True)

# Genera la cabecera principal de cada página con icono, título y subtítulo
def header(icon, title, subtitle):
    st.markdown(f'<div class="main-header"><h1>{icon} {title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)

# Renderiza una tarjeta KPI con etiqueta, valor principal y variación (delta)
def kpi(label, value, delta, delta_class="positive"):
    st.markdown(f'<div class="kpi-card"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div><div class="kpi-delta {delta_class}">{delta}</div></div>', unsafe_allow_html=True)

# Carga y muestra un archivo HTML de visualización Plotly dentro de un iframe
def show_viz(filename, height=600):
    path = VIZ_DIR / filename
    if path.exists():
        st.components.v1.html(path.read_text(encoding="utf-8"), height=height, scrolling=True)
    else:
        st.warning(f"Visualización no encontrada: `{filename}`")


# ─── Carga de datos ───────────────────────────────────────────────────────────

# Limpia valores infinitos y NaN de las columnas indicadas, reemplazándolos por null
def _clean_inf(df, cols):
    for c in cols:
        if c in df.columns:
            df = df.with_columns(pl.when(pl.col(c).is_infinite() | pl.col(c).is_nan()).then(None).otherwise(pl.col(c)).alias(c))
    return df

# Convierte columnas con formato decimal español (coma) a formato numérico estándar (punto)
def _fix_decimal(df, cols):
    for c in cols:
        if c in df.columns:
            df = df.with_columns(pl.col(c).cast(pl.Utf8).str.replace(",", ".").cast(pl.Float64, strict=False).alias(c))
    return df

# Carga el CSV de comparativa anual IPC vs IPV y limpia valores infinitos
@st.cache_data
def load_comparativa():
    return _clean_inf(pl.read_csv(DATA_DIR / "Comparativa_IPC_IPV.csv"), ["Variacion_IPC", "Variacion_IPV"])

# Carga el CSV de evolución del IPC y convierte los timestamps (ms) a fechas legibles
@st.cache_data
def load_evolucion_ipc():
    df = pl.read_csv(DATA_DIR / "Evolucion_IPC.csv")
    df = df.with_columns(
        (pl.col("Fecha") * 1000).cast(pl.Datetime("us")).dt.strftime("%Y-%m").alias("Fecha_str")
    )
    return df

# Entrena un modelo de regresión lineal simple (IPC → IPV) y devuelve el modelo, R², coeficiente e intercepto
@st.cache_data
def train_model():
    df = load_comparativa().filter(pl.col("IPC_Promedio").is_not_null() & pl.col("IPV_Promedio").is_not_null())
    X, y = df["IPC_Promedio"].to_numpy().reshape(-1, 1), df["IPV_Promedio"].to_numpy()
    m = LinearRegression().fit(X, y)
    return m, r2_score(y, m.predict(X)), m.coef_[0], m.intercept_


# ─── CSS ───────────────────────────────────────────────────────────────────────
load_css()

# ─── Sidebar ───────────────────────────────────────────────────────────────────
PAGES = ["📊 Resumen General", "📈 Evolución Temporal", "🗺️ Mapa IPC por CCAA", "🔗 Correlación IPC-IPV", "🤖 Modelos Predictivos"]
with st.sidebar:
    st.markdown("### 🏠 IPC vs IPV\n**Dashboard España**")
    st.markdown("---")
    page = st.radio("Nav", PAGES, label_visibility="collapsed")
    st.markdown("---")
    html_footer = (BASE_DIR / "main.html").read_text(encoding="utf-8")
    import re
    footer_match = re.search(r'<div class="sidebar-footer".*?</div>', html_footer, re.DOTALL)
    if footer_match:
        st.markdown(footer_match.group(0), unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PÁGINA 1: RESUMEN GENERAL
# ═══════════════════════════════════════════════════════════════════════════════
if page == PAGES[0]:
    header("📊", "Resumen General", "Visión global de la relación entre el IPC y el IPV en España")
    df = load_comparativa()
    last, prev = df.row(-1, named=True), df.row(-2, named=True)
    anyo = int(last["Anyo"])
    ipc_chg = ((last["IPC_Promedio"] - prev["IPC_Promedio"]) / abs(prev["IPC_Promedio"]) * 100) if prev["IPC_Promedio"] else 0
    ipv_chg = ((last["IPV_Promedio"] - prev["IPV_Promedio"]) / abs(prev["IPV_Promedio"]) * 100) if prev["IPV_Promedio"] else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi(f"IPC Promedio {anyo}", f"{last['IPC_Promedio']:.1f}%", f"{'▲' if ipc_chg>0 else '▼'} {abs(ipc_chg):.1f}% vs {anyo-1}", "positive" if ipc_chg<0 else "negative")
    with c2: kpi(f"IPV Promedio {anyo}", f"{last['IPV_Promedio']:.1f}", f"{'▲' if ipv_chg>0 else '▼'} {abs(ipv_chg):.1f}% vs {anyo-1}", "negative" if ipv_chg>0 else "positive")
    with c3: kpi("Años de datos", str(df.height), f"{int(df['Anyo'].min())} - {int(df['Anyo'].max())}")
    with c4: kpi("Ratio IPV/IPC", f"{last['Ratio_IPV_IPC']:.1f}", "Mayor ratio = mayor divergencia")

    st.markdown("<br>", unsafe_allow_html=True)
    cl, cr = st.columns(2)
    with cl:
        st.subheader("IPC vs IPV — Evolución Anual")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df["Anyo"].to_list(), y=df["IPC_Promedio"].to_list(), name="IPC (%)", marker_color="#818cf8", opacity=0.9))
        fig.add_trace(go.Scatter(x=df["Anyo"].to_list(), y=df["IPV_Promedio"].to_list(), name="IPV", mode="lines+markers", line=dict(color="#f472b6", width=3), marker=dict(size=8), yaxis="y2"))
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(title="IPC (%)", title_font_color="#818cf8", tickfont_color="#818cf8"),
            yaxis2=dict(title="IPV", title_font_color="#f472b6", tickfont_color="#f472b6", overlaying="y", side="right"),
            legend=dict(orientation="h", y=-0.15), margin=dict(l=40, r=40, t=20, b=60), height=400)
        st.plotly_chart(fig, use_container_width=True)
    with cr:
        st.subheader("Tabla Comparativa Anual")
        st.dataframe(df.select([
            pl.col("Anyo").cast(pl.Int32).alias("Año"), pl.col("IPC_Promedio").round(2).alias("IPC (%)"),
            pl.col("IPV_Promedio").round(2).alias("IPV"), pl.col("Ratio_IPV_IPC").round(2).alias("Ratio")
        ]).to_pandas(), use_container_width=True, height=400, hide_index=True)

    load_html_snippet("tmpl-resumen-info")


# ═══════════════════════════════════════════════════════════════════════════════
# PÁGINA 2: EVOLUCIÓN TEMPORAL
# ═══════════════════════════════════════════════════════════════════════════════
elif page == PAGES[1]:
    header("📈", "Evolución Temporal", "Series temporales del IPC y el IPV a nivel nacional")
    df_ipc = load_evolucion_ipc()
    years = sorted(df_ipc["Anyo"].unique().to_list())
    yr = st.slider("Rango de años", int(min(years)), int(max(years)), (int(min(years)), int(max(years))))
    df_f = df_ipc.filter((pl.col("Anyo") >= yr[0]) & (pl.col("Anyo") <= yr[1]))

    st.subheader("Variación Anual del IPC (%)")
    fig = px.line(df_f.to_pandas(), x="Fecha_str", y="Valor", labels={"Valor": "IPC (%)", "Fecha_str": "Fecha"}, template="plotly_dark")
    fig.update_traces(line=dict(color="#818cf8", width=2.5))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=400, margin=dict(l=40, r=20, t=20, b=40))
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Visualización Avanzada — Evolución Temporal")
    show_viz("evolucion_temporal.html")


# ═══════════════════════════════════════════════════════════════════════════════
# PÁGINA 3: MAPA IPC
# ═══════════════════════════════════════════════════════════════════════════════
elif page == PAGES[2]:
    header("🗺️", "Mapa del IPC por Comunidad Autónoma", "Distribución geográfica del IPC en España")
    show_viz("mapa_ipc_ccaa.html", 700)
    load_html_snippet("tmpl-mapa-info")


# ═══════════════════════════════════════════════════════════════════════════════
# PÁGINA 4: CORRELACIÓN
# ═══════════════════════════════════════════════════════════════════════════════
elif page == PAGES[3]:
    header("🔗", "Correlación IPC — IPV", "Análisis de la relación entre la inflación general y los precios de vivienda")
    show_viz("correlacion_ipc_ipv.html")

    df = load_comparativa()
    corr = np.corrcoef(df["IPC_Promedio"].to_numpy(), df["IPV_Promedio"].to_numpy())[0, 1]
    strength = "fuerte" if abs(corr) > 0.7 else "moderada" if abs(corr) > 0.4 else "débil"
    direction = "positiva" if corr > 0 else "negativa"

    c1, c2 = st.columns(2)
    with c1: kpi("Coef. Correlación (Pearson)", f"{corr:.4f}", f"Correlación {strength}", "positive" if abs(corr) > 0.5 else "negative")
    with c2:
        st.markdown(f'''<div class="section-card"><h4>📖 Interpretación</h4>
        <p>La correlación <strong>{direction}</strong> de <strong>{corr:.4f}</strong> entre el IPC y el IPV
        indica que {"cuando el IPC sube, el IPV también tiende a subir" if corr > 0 else "no existe una relación lineal directa clara"}.
        La inflación y los precios de la vivienda {"están relacionados a nivel macro" if abs(corr) > 0.4 else "pueden estar influidos por factores independientes"}.</p>
        </div>''', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PÁGINA 5: MODELOS PREDICTIVOS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == PAGES[4]:
    header("🤖", "Modelos Predictivos", "Regresiones lineales IPC → IPV: modelos de la actividad 3.3 e interfaz de predicción interactiva")

    st.subheader("🎯 Predicción Interactiva")
    st.markdown("Introduce un valor de IPC (variación anual promedio, %) para predecir el IPV con regresión lineal (datos 2017-2025).")
    model, r2, coef, intercept = train_model()

    ci, cr = st.columns([1, 2])
    with ci:
        ipc_in = st.number_input("Valor de IPC (%)", -5.0, 20.0, 3.0, 0.1, help="Variación anual del IPC en %")
        st.button("🔮 Predecir IPV", use_container_width=True, type="primary")
    with cr:
        pred = model.predict(np.array([[ipc_in]]))[0]
        st.markdown(f'<div class="prediction-box"><div class="prediction-label">IPV Predicho para IPC = {ipc_in:.1f}%</div><div class="prediction-value">{pred:.2f}</div><div class="prediction-label">Índice de Precios de Vivienda</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    fit_label = "Buen ajuste" if r2 > 0.7 else "Ajuste moderado" if r2 > 0.4 else "Ajuste bajo"
    with c1: kpi("R² del modelo", f"{r2:.4f}", fit_label, "positive" if r2 > 0.5 else "negative")
    with c2: kpi("Coeficiente", f"{coef:.3f}", f"Por cada +1% IPC, IPV {'sube' if coef>0 else 'baja'} {abs(coef):.2f} pts")
    with c3: kpi("Intercepto", f"{intercept:.2f}", "Valor base del IPV cuando IPC = 0")

    # Gráfico de regresión
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Regresión Lineal — Datos + Predicción")
    df = load_comparativa()
    X_d, y_d = df["IPC_Promedio"].to_numpy(), df["IPV_Promedio"].to_numpy()
    X_ln = np.linspace(X_d.min()-1, max(X_d.max()+1, ipc_in+1), 100)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X_d, y=y_d, mode="markers+text", name="Históricos", marker=dict(size=12, color="#818cf8", line=dict(width=1, color="white")), text=df["Anyo"].cast(pl.Int32).to_list(), textposition="top center", textfont=dict(size=10, color="#94a3b8")))
    fig.add_trace(go.Scatter(x=X_ln, y=model.predict(X_ln.reshape(-1,1)), mode="lines", name=f"Regresión (R²={r2:.3f})", line=dict(color="#f472b6", width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=[ipc_in], y=[pred], mode="markers", name=f"Predicción (IPC={ipc_in:.1f}%)", marker=dict(size=16, color="#34d399", symbol="star", line=dict(width=2, color="white"))))
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis_title="IPC (%)", yaxis_title="IPV", height=450, margin=dict(l=40, r=40, t=20, b=60), legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig, use_container_width=True)

    # Modelos 3.3
    st.markdown("---")
    st.subheader("📊 Modelos de Regresión — Actividad 3.3")
    reg_files = {"🇪🇸 General España": "regresion_general_espana.html", "🏘️ Por CCAA": "regresion_por_ccaa.html", "📊 R² por CCAA": "regresion_r2_por_ccaa.html", "🏠 Por Tipo Vivienda": "regresion_tipo_vivienda.html"}
    for tab, fname in zip(st.tabs(list(reg_files.keys())), reg_files.values()):
        with tab:
            show_viz(fname)

    load_html_snippet("tmpl-modelos-info")

# 📊 Dashboard IPC vs IPV — España

Panel de control interactivo que analiza la relación entre el **Índice de Precios al Consumo (IPC)** y el **Índice de Precios de Vivienda (IPV)** en España, integrando datos desde 2007 hasta 2025.

## 🚀 Tecnologías

- **Streamlit** — Interfaz web interactiva
- **Polars** — Procesamiento de datos de alto rendimiento
- **Plotly** — Visualizaciones interactivas
- **Scikit-learn** — Modelos de regresión predictiva
- **NumPy** — Cálculos numéricos

## 📁 Estructura del Proyecto

```
3.4_SBD/
├── app.py                  # Aplicación principal Streamlit
├── requirements.txt        # Dependencias del proyecto
├── README.md               # Este archivo
├── data_output/            # Datos procesados
│   ├── Comparativa_IPC_IPV.csv
│   ├── Evolucion_IPC.csv
│   ├── Evolucion_IPV.csv
│   ├── Variaciones_Interanuales.csv
│   ├── ipc_clean.csv
│   └── ipv_clean.csv
└── visualizations/         # Gráficos Plotly (HTML)
    ├── regresion_general_espana.html
    ├── regresion_por_ccaa.html
    ├── regresion_r2_por_ccaa.html
    ├── regresion_tipo_vivienda.html
    ├── correlacion_ipc_ipv.html
    ├── evolucion_temporal.html
    └── mapa_ipc_ccaa.html
```

## ⚙️ Instalación y Ejecución

### 1. Crear entorno virtual

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. Ejecutar el dashboard

```bash
streamlit run app.py
```

El dashboard se abrirá automáticamente en `http://localhost:8501`.

## 📋 Secciones del Dashboard

| Sección | Descripción |
|---------|-------------|
| 📊 **Resumen General** | KPIs clave, gráfico comparativo anual IPC vs IPV, tabla interactiva |
| 📈 **Evolución Temporal** | Series temporales del IPC con filtro por años, visualización avanzada |
| 🗺️ **Mapa IPC por CCAA** | Mapa coroplético del IPC por Comunidad Autónoma |
| 🔗 **Correlación IPC-IPV** | Análisis de correlación con coeficiente de Pearson |
| 🤖 **Modelos Predictivos** | Regresión lineal interactiva (IPC→IPV) + modelos de la actividad 3.3 |

## 👤 Autores

**Jorge Castillo y Pedro Zarzuela**  
Actividad 3.4 · Sistemas de Big Data (SBD)
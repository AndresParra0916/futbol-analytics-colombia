import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import joblib
from sklearn.metrics.pairwise import cosine_similarity

# Configuración de la página
st.set_page_config(
    page_title="Futbol Analytics Colombia",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("⚽ Futbol Analytics Colombia")
st.markdown("### Sistema de Scouting Inteligente y Prevención de Lesiones")
st.markdown("---")

# ============================================
# FUNCIONES DE CARGA DE DATOS
# ============================================
@st.cache_data(ttl=3600)
def cargar_posiciones():
    try:
        df = pd.read_csv('data/tabla_posiciones.csv')
        return df
    except:
        return None

@st.cache_data(ttl=3600)
def cargar_jugadores():
    try:
        df = pd.read_csv('data/jugadores_stats.csv')
        return df
    except:
        return None

@st.cache_resource
def cargar_modelo_scouting():
    try:
        scaler = joblib.load('models/scaler_scouting.pkl')
        ref = joblib.load('models/referencia_scouting.pkl')
        return scaler, ref
    except:
        return None, None

def recomendar_similares(nombre, scaler, ref, top_n=5):
    if scaler is None or ref is None:
        return None
    if nombre not in ref['Nombre'].values:
        return None
    features = ['goles_p90', 'asistencias_p90', 'recuperaciones_p90', 
                'duelos_ganados_p90', 'pases_progresivos_p90']
    X = scaler.transform(ref[features].fillna(0).values)
    idx = ref[ref['Nombre'] == nombre].index[0]
    sim = cosine_similarity([X[idx]], X)[0]
    top_idx = np.argsort(sim)[::-1][1:top_n+1]
    return ref.iloc[top_idx][['Nombre', 'Equipo'] + features].assign(similitud=sim[top_idx].round(3))

# ============================================
# SIDEBAR - NAVEGACIÓN
# ============================================
st.sidebar.image("https://github.com/AndresParra0916/futbol-analytics-colombia/blob/main/reports/puntos_por_equipo.png?raw=True", use_column_width=True)
st.sidebar.markdown("## Navegación")
opcion = st.sidebar.radio("Ir a:", ["📊 Tabla de Posiciones", "⚽ Top Goleadores", "🔍 Scouting", "⚠️ Riesgo de Lesión"])

# ============================================
# 1. TABLA DE POSICIONES
# ============================================
if opcion == "📊 Tabla de Posiciones":
    st.header("📊 Tabla de Posiciones - Liga BetPlay")
    df_pos = cargar_posiciones()
    if df_pos is not None:
        st.dataframe(df_pos, use_container_width=True, hide_index=True)
        fig = px.bar(df_pos, x='Equipo', y='PTS', title='Puntos por Equipo',
                     color='PTS', color_continuous_scale='Viridis', text='PTS')
        fig.update_layout(xaxis_tickangle=-45, height=500)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No se encontraron datos de posiciones. Ejecuta primero el workflow.")

# ============================================
# 2. TOP GOLEADORES Y ASISTENTES
# ============================================
elif opcion == "⚽ Top Goleadores":
    st.header("⚽ Estadísticas de Jugadores")
    df_jug = cargar_jugadores()
    if df_jug is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Top 10 Goleadores")
            top_g = df_jug.nlargest(10, 'Goles')[['Nombre', 'Equipo', 'Goles', 'goles_p90']]
            st.dataframe(top_g, use_container_width=True, hide_index=True)
            fig_g = px.bar(top_g, x='Nombre', y='Goles', title='Goles', color='Goles')
            st.plotly_chart(fig_g, use_container_width=True)
        with col2:
            st.subheader("Top 10 Asistentes")
            top_a = df_jug.nlargest(10, 'Asistencias')[['Nombre', 'Equipo', 'Asistencias', 'asistencias_p90']]
            st.dataframe(top_a, use_container_width=True, hide_index=True)
            fig_a = px.bar(top_a, x='Nombre', y='Asistencias', title='Asistencias', color='Asistencias')
            st.plotly_chart(fig_a, use_container_width=True)
    else:
        st.warning("No se encontraron datos de jugadores.")

# ============================================
# 3. SCOUTING - RECOMENDACIONES
# ============================================
elif opcion == "🔍 Scouting":
    st.header("🔍 Motor de Scouting - Encontrar Jugadores Similares")
    df_jug = cargar_jugadores()
    scaler, ref = cargar_modelo_scouting()
    if df_jug is not None and scaler is not None:
        jugadores_lista = sorted(df_jug['Nombre'].unique())
        jugador_base = st.selectbox("Selecciona un jugador de referencia:", jugadores_lista)
        top_n = st.slider("Número de recomendaciones:", 3, 10, 5)
        if st.button("🔍 Recomendar similares"):
            similares = recomendar_similares(jugador_base, scaler, ref, top_n)
            if similares is not None:
                st.success(f"Jugadores similares a {jugador_base}:")
                st.dataframe(similares, use_container_width=True, hide_index=True)
            else:
                st.error("No se pudieron generar recomendaciones.")
    else:
        st.warning("Modelo de scouting no disponible. Ejecuta primero el workflow.")

# ============================================
# 4. RIESGO DE LESIÓN (SIMULADO)
# ============================================
else:
    st.header("⚠️ Predicción de Riesgo de Lesión")
    st.markdown("_Esta sección utiliza un modelo simulado. Con datos GPS reales, la precisión será profesional._")
    col1, col2 = st.columns(2)
    with col1:
        minutos = st.number_input("Minutos en la semana:", 0, 180, 90)
        acwr = st.number_input("ACWR (carga aguda/crónica):", 0.5, 2.5, 1.2, 0.05)
        sprints = st.number_input("Número de sprints:", 0, 50, 12)
    with col2:
        aceleraciones = st.number_input("Aceleraciones intensas:", 0, 100, 30)
        descanso = st.number_input("Días de descanso:", 1, 7, 3)
        fatiga = st.number_input("Índice de fatiga (0-4):", 0.0, 4.0, 1.0, 0.1)
    
    # Modelo simulado (cuando tengas datos GPS reales, reemplázalo)
    np.random.seed(42)
    proba = np.clip(0.1 + (minutos/180)*0.3 + (acwr-1)*0.2 + (sprints/50)*0.1 + (1/descanso)*0.1, 0.05, 0.95)
    riesgo_texto = "🔴 ALTO" if proba > 0.6 else "🟡 MODERADO" if proba > 0.3 else "🟢 BAJO"
    st.metric("Probabilidad de lesión en la próxima semana", f"{proba:.1%}")
    st.markdown(f"**Riesgo: {riesgo_texto}**")
    st.progress(proba)
    if proba > 0.6:
        st.error("⚠️ Alto riesgo. Considera reducir carga o aumentar descanso.")
    elif proba > 0.3:
        st.warning("📉 Riesgo moderado. Monitorear fatiga y recuperación.")
    else:
        st.success("✅ Riesgo bajo. Mantener la planificación.")

# Pie de página
st.sidebar.markdown("---")
st.sidebar.info("**Actualización automática:** Cada lunes y jueves (10 AM Colombia).")

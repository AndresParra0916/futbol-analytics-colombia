import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import joblib
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Futbol Analytics Colombia", page_icon="⚽", layout="wide")

st.title("⚽ Futbol Analytics Colombia")
st.markdown("### Sistema de Scouting Inteligente y Prevención de Lesiones")
st.markdown("---")

# ============================================
# CARGAR DATOS (usando tus archivos reales)
# ============================================

@st.cache_data(ttl=3600)
def cargar_posiciones():
    # Tu archivo de posiciones se llama 'tabla_posiciones.csv' (sin 'betplay')
    try:
        return pd.read_csv('data/tabla_posiciones.csv')
    except:
        return None

@st.cache_data(ttl=3600)
def cargar_jugadores():
    # Usar el archivo de estadísticas de la API (con los 360 registros)
    try:
        df = pd.read_csv('data/estadisticas_api_2024_prueba.csv')
        return df
    except:
        return None

@st.cache_resource
def cargar_scouting():
    try:
        scaler = joblib.load('models/scaler_scouting.pkl')
        ref = joblib.load('models/referencia_scouting.pkl')
        return scaler, ref
    except:
        return None, None

def recomendar_similares(nombre, scaler, ref, top_n=5):
    if scaler is None or ref is None:
        return None
    if nombre not in ref['player_name'].values:
        return None
    features = ['goles_p90', 'asistencias_p90', 'tiros_p90', 'pases_p90', 'entradas_p90', 'duelos_ganados_p90']
    X = scaler.transform(ref[features].fillna(0).values)
    idx = ref[ref['player_name'] == nombre].index[0]
    sim = cosine_similarity([X[idx]], X)[0]
    top_idx = np.argsort(sim)[::-1][1:top_n+1]
    return ref.iloc[top_idx][['player_name'] + features].assign(similitud=sim[top_idx].round(3))

# ============================================
# MENÚ
# ============================================
opcion = st.sidebar.radio("Menú", ["📊 Tabla de Posiciones", "⚽ Top Goleadores", "🔍 Scouting", "⚠️ Riesgo de Lesión"])

# ============================================
# 1. TABLA DE POSICIONES
# ============================================
if opcion == "📊 Tabla de Posiciones":
    st.header("Tabla de Posiciones")
    df = cargar_posiciones()
    if df is not None:
        st.dataframe(df, use_container_width=True)
    else:
        st.warning("Archivo 'data/tabla_posiciones.csv' no encontrado. Ejecuta primero el scraping de ESPN.")

# ============================================
# 2. TOP GOLEADORES (desde estadísticas de la API)
# ============================================
elif opcion == "⚽ Top Goleadores":
    st.header("Top Goleadores")
    df_jug = cargar_jugadores()
    if df_jug is not None:
        # Agrupar por jugador y sumar goles
        top_g = df_jug.groupby('player_name')['goals'].sum().nlargest(10).reset_index()
        top_g.columns = ['Jugador', 'Goles']
        st.dataframe(top_g, use_container_width=True)
        fig = px.bar(top_g, x='Jugador', y='Goles', title='Goles por Jugador')
        st.plotly_chart(fig)
    else:
        st.warning("Datos de jugadores no disponibles. Ejecuta primero 'api_stats_2024_prueba.py'")

# ============================================
# 3. SCOUTING (recomendaciones)
# ============================================
elif opcion == "🔍 Scouting":
    st.header("🔍 Motor de Scouting - Encontrar Jugadores Similares")
    df_jug = cargar_jugadores()
    scaler, ref = cargar_scouting()
    if df_jug is not None and scaler is not None:
        jugadores_lista = sorted(df_jug['player_name'].unique())
        if len(jugadores_lista) == 0:
            st.warning("No hay jugadores en los datos.")
        else:
            jugador_base = st.selectbox("Selecciona un jugador de referencia:", jugadores_lista)
            top_n = st.slider("Número de recomendaciones:", 3, 10, 5)
            if st.button("🔍 Recomendar similares"):
                similares = recomendar_similares(jugador_base, scaler, ref, top_n)
                if similares is not None:
                    st.success(f"Jugadores similares a {jugador_base}:")
                    st.dataframe(similares, use_container_width=True, hide_index=True)
                else:
                    st.error("No se pudieron generar recomendaciones. Asegúrate de que el modelo esté entrenado.")
    else:
        st.warning("Modelo de scouting no disponible. Ejecuta 'entrenar_con_api_2024.py' primero.")

# ============================================
# 4. RIESGO DE LESIÓN (simulado)
# ============================================
else:
    st.header("Predicción de Riesgo de Lesión")
    st.info("Modelo simulado. Con datos GPS reales se puede personalizar.")
    col1, col2 = st.columns(2)
    with col1:
        minutos = st.number_input("Minutos en la semana", 0, 180, 90)
        sprints = st.number_input("Sprints", 0, 50, 12)
    with col2:
        descanso = st.number_input("Días de descanso", 1, 7, 3)
        aceleraciones = st.number_input("Aceleraciones", 0, 100, 30)
    riesgo = np.clip(0.1 + (minutos/180)*0.3 + (sprints/50)*0.2 + (1/descanso)*0.1 + (aceleraciones/100)*0.1, 0, 1)
    st.metric("Probabilidad de lesión en la próxima semana", f"{riesgo:.1%}")
    if riesgo > 0.6:
        st.error("⚠️ Alto riesgo. Considera reducir carga o aumentar descanso.")
    elif riesgo > 0.3:
        st.warning("📉 Riesgo moderado. Monitorear fatiga.")
    else:
        st.success("✅ Riesgo bajo.")
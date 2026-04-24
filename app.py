import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(layout="wide")
st.title("⚽ Futbol Analytics Colombia")
st.markdown("### Sistema de Scouting Inteligente y Prevención de Lesiones")
st.markdown("---")

# Cargar modelos (usando los archivos que ya probaste que funcionan)
try:
    scaler = joblib.load("models/scaler_scouting_2025.pkl")
    ref = joblib.load("models/referencia_scouting_2025.pkl")
    features = ['goles_p90', 'asistencias_p90', 'tiros_p90', 'pases_p90', 'entradas_p90', 'duelos_ganados_p90']
    st.success("✅ Modelos cargados correctamente")
except Exception as e:
    st.error(f"Error cargando modelos: {e}")
    st.stop()

# Tabla de posiciones
try:
    df_tabla = pd.read_csv("data/tabla_posiciones.csv")
    st.header("📊 Tabla de Posiciones 2025")
    st.dataframe(df_tabla, use_container_width=True)
except:
    st.warning("Tabla de posiciones no disponible (se generará automáticamente más tarde)")

# Top goleadores
try:
    df_stats = pd.read_csv("data/estadisticas_api_2025_prueba.csv")
    goleadores = df_stats.groupby('player_name')['goals'].sum().nlargest(10).reset_index()
    goleadores.columns = ['Jugador', 'Goles']
    st.header("⚽ Top Goleadores")
    st.dataframe(goleadores, use_container_width=True)
except:
    st.warning("Datos de goleadores no disponibles")

# Scouting
st.header("🔍 Motor de Scouting")
jugadores = sorted(ref['player_name'].unique())
jugador_base = st.selectbox("Selecciona un jugador de referencia", jugadores)
top_n = st.slider("Número de recomendaciones", 3, 10, 5)

if st.button("Recomendar similares"):
    idx = ref[ref['player_name'] == jugador_base].index[0]
    X = scaler.transform(ref[features].fillna(0).values)
    sim = cosine_similarity([X[idx]], X)[0]
    top_idx = np.argsort(sim)[::-1][1:top_n+1]
    resultados = ref.iloc[top_idx][['player_name'] + features].copy()
    resultados['similitud'] = sim[top_idx].round(3)
    st.success(f"Jugadores similares a {jugador_base}:")
    st.dataframe(resultados, use_container_width=True)

# Riesgo de lesión (demo)
st.header("⚠️ Riesgo de Lesión (Demo)")
st.info("Modelo conceptual. Con datos GPS del club se puede personalizar.")
minutos = st.number_input("Minutos en la semana", 0, 180, 90)
riesgo = np.clip(0.1 + minutos/180 * 0.3, 0, 1)
st.metric("Probabilidad estimada", f"{riesgo:.1%}")
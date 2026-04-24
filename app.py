import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(layout="wide")

# ========== PWA: MANIFEST Y SERVICE WORKER ==========
st.markdown("""
    <link rel="manifest" href="/manifest.json">
    <script>
        if ('serviceWorker' in navigator) {
            window.addEventListener('load', () => {
                navigator.serviceWorker.register('/service-worker.js')
                .then(reg => console.log('Service Worker registrado', reg))
                .catch(err => console.log('Error al registrar SW', err));
            });
        }
    </script>
""", unsafe_allow_html=True)
# ===================================================

st.title("⚽ Futbol Analytics Colombia")
st.markdown("### Sistema de Scouting Inteligente y Prevención de Lesiones")
st.markdown("---")

# Cargar modelos
try:
    scaler = joblib.load("models/scaler_scouting_2025.pkl")
    ref = joblib.load("models/referencia_scouting_2025.pkl")
    features = ['goles_p90', 'asistencias_p90', 'tiros_p90', 'pases_p90', 'entradas_p90', 'duelos_ganados_p90']
    st.success("✅ Modelos cargados correctamente")
except Exception as e:
    st.error(f"Error cargando modelos: {e}")

# Tabla de posiciones
try:
    df_tabla = pd.read_csv("data/tabla_posiciones.csv")
    st.header("📊 Tabla de Posiciones 2025")
    st.dataframe(df_tabla, use_container_width=True)
except:
    st.warning("Tabla de posiciones no disponible")

# Top goleadores
try:
    df_stats = pd.read_csv("data/estadisticas_api.csv")
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

# Riesgo de lesión
st.header("⚠️ Riesgo de Lesión (Demo)")
st.info("Modelo conceptual. Con datos GPS del club se puede personalizar.")
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
    st.error("⚠️ Alto riesgo. Considera reducir carga.")
elif riesgo > 0.3:
    st.warning("📉 Riesgo moderado.")
else:
    st.success("✅ Riesgo bajo.")
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import requests

st.set_page_config(layout="wide")
st.title("⚽ Futbol Analytics Colombia")
st.markdown("### Sistema de Scouting Inteligente y Prevención de Lesiones")
st.markdown("---")

# ============================================
# 1. TABLA DE POSICIONES (desde API si no existe CSV)
# ============================================
try:
    df_tabla = pd.read_csv("data/tabla_posiciones.csv")
except:
    API_KEY = "ebb8f00138af0df132bbda386d55981c"
    headers = {"x-apisports-key": API_KEY}
    url = "https://v3.football.api-sports.io/standings?league=239&season=2025"
    r = requests.get(url, headers=headers)
    data = r.json()
    standings_arrays = data['response'][0]['league']['standings']
    tabla_general = [arr for arr in standings_arrays if len(arr)==20][0]
    equipos = []
    for t in tabla_general:
        equipos.append({
            'Equipo': t['team']['name'],
            'PJ': t['all']['played'],
            'PG': t['all']['win'],
            'PE': t['all']['draw'],
            'PP': t['all']['lose'],
            'GF': t['all']['goals']['for'],
            'GC': t['all']['goals']['against'],
            'DIF': t['goalsDiff'],
            'PTS': t['points']
        })
    df_tabla = pd.DataFrame(equipos)

st.header("📊 Tabla de Posiciones 2025")
st.dataframe(df_tabla, use_container_width=True, hide_index=True)

# ============================================
# 2. TOP GOLEADORES (desde estadísticas)
# ============================================
try:
    df_stats = pd.read_csv("data/estadisticas_api_2025_prueba.csv")
    goleadores = df_stats.groupby('player_name')['goals'].sum().nlargest(10).reset_index()
    goleadores.columns = ['Jugador', 'Goles']
    st.header("⚽ Top Goleadores")
    st.dataframe(goleadores, use_container_width=True, hide_index=True)
    fig = px.bar(goleadores, x='Jugador', y='Goles', title='Goles por Jugador')
    st.plotly_chart(fig)
except:
    st.warning("Datos de goleadores no disponibles")

# ============================================
# 3. SCOUTING (con los modelos correctos)
# ============================================
st.header("🔍 Motor de Scouting - Encontrar Jugadores Similares")

try:
    # Cargar modelos desde la carpeta models (usando los de 2025)
    scaler = joblib.load("models/scaler_scouting_2025.pkl")
    ref = joblib.load("models/referencia_scouting_2025.pkl")
    features = ['goles_p90', 'asistencias_p90', 'tiros_p90', 'pases_p90', 'entradas_p90', 'duelos_ganados_p90']
    
    jugadores_lista = sorted(ref['player_name'].unique())
    jugador_base = st.selectbox("Selecciona un jugador de referencia:", jugadores_lista)
    top_n = st.slider("Número de recomendaciones:", 3, 10, 5)
    
    if st.button("🔍 Recomendar similares"):
        idx = ref[ref['player_name'] == jugador_base].index[0]
        X = scaler.transform(ref[features].fillna(0).values)
        sim = cosine_similarity([X[idx]], X)[0]
        top_idx = np.argsort(sim)[::-1][1:top_n+1]
        resultados = ref.iloc[top_idx][['player_name'] + features].copy()
        resultados['similitud'] = sim[top_idx].round(3)
        st.success(f"Jugadores similares a {jugador_base}:")
        st.dataframe(resultados, use_container_width=True, hide_index=True)
except Exception as e:
    st.warning(f"No se pudo cargar el modelo de scouting. Error: {e}")

# ============================================
# 4. RIESGO DE LESIÓN (demo)
# ============================================
st.header("⚠️ Predicción de Riesgo de Lesión")
st.info("Modelo conceptual. Con datos GPS reales del club se puede personalizar.")
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
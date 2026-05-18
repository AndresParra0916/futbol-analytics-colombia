# app.py (versión profesional y atractiva)
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity

# Configuración de página
st.set_page_config(page_title="Futbol Analytics Colombia", page_icon="⚽", layout="wide")

# CSS personalizado para mejorar la apariencia
st.markdown("""
<style>
    .reportview-container .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stButton > button {
        background-color: #1E88E5;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        transition: 0.3s;
    }
    .stButton > button:hover {
        background-color: #0D47A1;
        color: white;
    }
    .css-1aumxhk {
        background-color: #f5f7fa;
    }
    .card {
        background-color: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1E88E5 0%, #0D47A1 100%);
        color: white;
        border-radius: 16px;
        padding: 1rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.title("⚽ **Futbol Analytics Colombia**")
st.markdown("### *Sistema de Scouting Inteligente y Prevención de Lesiones*")
st.markdown("---")

# ============================================
# 1. TABLA DE POSICIONES Y GOLEADORES (en tarjetas)
# ============================================
with st.container():
    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        try:
            df_standings = pd.read_csv("data/tabla_posiciones.csv")
            st.subheader("📊 Tabla de Posiciones 2026 - Colombia")
            st.dataframe(df_standings, use_container_width=True)
        except:
            st.warning("Tabla de posiciones no disponible")
        st.markdown('</div>', unsafe_allow_html=True)
    with col_right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        try:
            df_scorers = pd.read_csv("data/top_goleadores.csv")
            st.subheader("⚽ Top Goleadores 2026 - Colombia")
            st.dataframe(df_scorers.head(10), use_container_width=True)
            fig = px.bar(df_scorers.head(10), x='Jugador', y='Goles', title='Top 10 Goleadores',
                         color='Goles', color_continuous_scale='Blues')
            st.plotly_chart(fig, use_container_width=True)
        except:
            st.warning("Top goleadores no disponible")
        st.markdown('</div>', unsafe_allow_html=True)

# ============================================
# 2. SCOUTING AVANZADO (con diseño mejorado)
# ============================================
st.markdown("---")
st.header("🔍 **Scouting Avanzado**")
st.markdown("### Encuentra jugadores similares en cualquier liga latinoamericana")

# Cargar datos unificados
try:
    df_players = pd.read_csv("data/players_unificado.csv")
    scaler = joblib.load("models/scaler_unificado.pkl")
    ref = joblib.load("models/referencia_unificado.pkl")
    features = ['goals_p90', 'assists_p90', 'shots_p90', 'passes_p90', 'tackles_p90', 'duels_won_p90']
    st.success(f"✅ {len(df_players)} jugadores de {df_players['league_name'].nunique()} ligas cargados.")
except Exception as e:
    st.error(f"Error cargando datos de scouting: {e}")
    st.stop()

# Preparar datos para visualización
if 'player_id' not in ref.columns:
    id_map = df_players[['player_name', 'team_name', 'player_id']].drop_duplicates()
    ref = ref.merge(id_map, on=['player_name', 'team_name'], how='left')
ref = ref.fillna({'country': 'Sin país', 'team_name': 'Sin equipo', 'position': 'No especificada'})
ref['display_name'] = ref['player_name'] + ' (' + ref['team_name'] + ')'

# Filtros laterales
with st.container():
    col_f1, col_f2, col_f3, col_f4 = st.columns([1.5, 1.5, 1.5, 2])
    with col_f1:
        all_leagues = sorted(ref['league_name'].dropna().unique())
        selected_leagues = st.multiselect("🏆 Liga(s)", all_leagues, default=all_leagues[:3])
    with col_f2:
        all_countries = sorted(ref['country'].dropna().unique())
        selected_countries = st.multiselect("🌎 País(es)", all_countries, default=[])
    with col_f3:
        all_positions = sorted(ref['position'].dropna().unique())
        selected_positions = st.multiselect("📍 Posición(es)", all_positions, default=[])
    with col_f4:
        all_players = sorted(ref['display_name'].unique())
        selected_player = st.selectbox("🎯 Jugador de referencia", all_players)

    top_n = st.slider("📊 Número de recomendaciones", 3, 20, 5)

if st.button("🔍 Buscar jugadores similares", use_container_width=True):
    player_row = ref[ref['display_name'] == selected_player].iloc[0]
    player_id = player_row['player_id']
    player_name = player_row['player_name']
    player_team = player_row['team_name']

    # Filtrado
    mask = (ref['league_name'].isin(selected_leagues) if selected_leagues else True)
    if selected_countries:
        mask &= (ref['country'].isin(selected_countries))
    if selected_positions:
        mask &= (ref['position'].isin(selected_positions))
    candidates = ref[mask].copy()
    candidates = candidates[candidates['player_id'] != player_id]

    if candidates.empty:
        st.warning("No hay jugadores con esos filtros.")
    else:
        idx_ref = ref[ref['player_id'] == player_id].index[0]
        vec_ref = scaler.transform([ref.loc[idx_ref, features].fillna(0).values])[0]
        X_cand = scaler.transform(candidates[features].fillna(0).values)
        sim = cosine_similarity([vec_ref], X_cand)[0]
        candidates['similitud'] = sim
        results = candidates.sort_values('similitud', ascending=False).head(top_n)

        # Mostrar resultados en una tabla elegante
        st.subheader(f"🏅 Jugadores similares a **{player_name} ({player_team})**")
        cols_show = ['player_name', 'team_name', 'league_name', 'country', 'position', 'goals_p90', 'assists_p90', 'similitud']
        st.dataframe(results[cols_show].round(2), use_container_width=True)

        # Gráfico de barras de similitud
        fig = px.bar(results, x='player_name', y='similitud', color='league_name',
                     title=f"Similitud con {player_name}", labels={'player_name': 'Jugador', 'similitud': 'Similitud (coseno)'})
        st.plotly_chart(fig, use_container_width=True)

# ============================================
# 3. PREDICCIÓN DE LESIÓN (con todas las variables)
# ============================================
st.markdown("---")
st.header("⚠️ **Predicción de Riesgo de Lesión**")
st.markdown("### Modelo basado en carga GPS, fatiga y descanso")

try:
    injury_model = joblib.load("models/modelo_lesiones.pkl")
    injury_scaler = joblib.load("models/scaler_lesiones.pkl")
    model_loaded = True
except:
    model_loaded = False
    st.warning("Modelo de lesiones no entrenado. Ejecuta 'entrenar_modelo_lesiones.py' primero.")

if model_loaded:
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Ingrese los datos de la última semana para estimar el riesgo")

        col_g1, col_g2, col_g3 = st.columns(3)
        with col_g1:
            high_intensity_dist = st.number_input("🚀 Distancia alta intensidad (m)", min_value=0, max_value=15000, value=6000, step=100)
            sprints = st.number_input("⚡ Sprints", min_value=0, max_value=200, value=25, step=1)
        with col_g2:
            accelerations = st.number_input("📈 Aceleraciones (>3 m/s²)", min_value=0, max_value=300, value=45, step=1)
            decelerations = st.number_input("📉 Desaceleraciones (< -3 m/s²)", min_value=0, max_value=300, value=50, step=1)
        with col_g3:
            rest_days = st.number_input("😴 Días de descanso (promedio semana)", min_value=0.0, max_value=7.0, value=2.0, step=0.5)
            fatigue = st.number_input("📊 Nivel de fatiga (escala 1-10)", min_value=1.0, max_value=10.0, value=5.0, step=0.5)

        # Valores típicos de carga crónica (ajustar según histórico del jugador)
        chronic_high = 5500
        chronic_sprints = 22
        chronic_acc = 40
        chronic_dec = 45
        chronic_load = 3500

        acwr_high = high_intensity_dist / chronic_high
        acwr_sprints = sprints / chronic_sprints
        acwr_acc = accelerations / chronic_acc
        acwr_dec = decelerations / chronic_dec
        acwr_load = (high_intensity_dist + accelerations + decelerations) / chronic_load

        # Entrada para el modelo: [acwr_high, acwr_sprints, acwr_acc, acwr_dec, acwr_load, rest_days, fatigue, minutos estimados]
        minutes_est = 90 if high_intensity_dist > 3000 else 60
        X_input = np.array([[acwr_high, acwr_sprints, acwr_acc, acwr_dec, acwr_load, rest_days, fatigue, minutes_est]])
        X_scaled = injury_scaler.transform(X_input)
        proba = injury_model.predict_proba(X_scaled)[0][1]
        riesgo = proba

        # Mostrar resultado con colores
        st.markdown("---")
        col_met, col_msj = st.columns([1, 2])
        with col_met:
            st.metric("📉 Probabilidad de lesión en la próxima semana", f"{riesgo:.1%}")
        with col_msj:
            if riesgo > 0.6:
                st.error("⚠️ **ALTO RIESGO** - Reduzca carga o aumente descanso inmediatamente.")
            elif riesgo > 0.3:
                st.warning("📉 **RIESGO MODERADO** - Monitoree fatiga y ajuste cargas individualmente.")
            else:
                st.success("✅ **RIESGO BAJO** - Mantener plan de entrenamiento actual.")

        # Barra de progreso
        progress_color = "#d9534f" if riesgo > 0.6 else "#f0ad4e" if riesgo > 0.3 else "#5cb85c"
        st.progress(riesgo, text=f"Riesgo: {riesgo:.1%}")
        st.markdown(f'<div style="width:100%; background-color:#e9ecef; border-radius:8px;"><div style="width:{riesgo*100}%; background-color:{progress_color}; border-radius:8px; text-align:center; padding:2px;"> </div></div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)
else:
    st.info("Para activar esta sección, entrena el modelo con datos GPS reales o sintéticos usando `entrenar_modelo_lesiones.py`.")
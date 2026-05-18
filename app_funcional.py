import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(layout="wide")
st.title("⚽ Futbol Analytics Colombia")
st.markdown("### Sistema de Scouting Inteligente y Prevención de Lesiones")
st.markdown("---")

# ============================================
# 1. CARGAR DATOS UNIFICADOS
# ============================================
try:
    df_players = pd.read_csv("data/players_unificado.csv")
except:
    st.error("No se encuentra 'data/players_unificado.csv'. Ejecuta 'actualizar_datos.py' primero.")
    st.stop()

try:
    scaler = joblib.load("models/scaler_unificado.pkl")
    ref = joblib.load("models/referencia_unificado.pkl")
except Exception as e:
    st.error(f"Error cargando modelos: {e}")
    st.stop()

features = ['goals_p90', 'assists_p90', 'shots_p90', 'passes_p90', 'tackles_p90', 'duels_won_p90']
# Verificar que ref tiene las columnas necesarias (puede llamarse 'league_name' o 'league')
if 'league_name' not in ref.columns and 'league' in ref.columns:
    ref.rename(columns={'league': 'league_name'}, inplace=True)
if 'country' not in ref.columns:
    # Si no hay país, intentamos obtenerlo de df_players
    country_map = df_players[['league_name', 'country']].drop_duplicates()
    ref = ref.merge(country_map, on='league_name', how='left')

# Unificar con la metadata de posición y equipo (si no están)
if 'team_name' not in ref.columns:
    team_map = df_players[['player_name', 'team_name']].drop_duplicates()
    ref = ref.merge(team_map, on='player_name', how='left')
if 'position' not in ref.columns:
    pos_map = df_players[['player_name', 'position']].drop_duplicates()
    ref = ref.merge(pos_map, on='player_name', how='left')

ref = ref.fillna({'position': 'No especificada', 'team_name': 'Sin equipo', 'country': 'Sin país'})

# ============================================
# 2. TABLA DE POSICIONES (SOLO COLOMBIA)
# ============================================
try:
    df_tabla = pd.read_csv("data/tabla_posiciones.csv")
    st.header("📊 Tabla de Posiciones 2026 - Colombia")
    st.dataframe(df_tabla, use_container_width=True)
except:
    st.warning("Tabla de posiciones no disponible. Ejecuta 'actualizar_datos.py' primero.")

# ============================================
# 3. TOP GOLEADORES (SOLO COLOMBIA)
# ============================================
try:
    df_goles = pd.read_csv("data/top_goleadores.csv")
    st.header("⚽ Top Goleadores 2026 - Colombia")
    st.dataframe(df_goles.head(10), use_container_width=True)
    fig = px.bar(df_goles.head(10), x='Jugador', y='Goles', title='Top 10 Goleadores Liga BetPlay')
    st.plotly_chart(fig)
except:
    st.warning("Top goleadores no disponible.")

# ============================================
# 4. SCOUTING AVANZADO (MULTILIGA)
# ============================================
st.header("🔍 Scouting Avanzado - Compara jugadores en todas las ligas latinoamericanas")

# Obtener valores únicos para filtros
all_leagues = sorted(ref['league_name'].dropna().unique())
all_countries = sorted(ref['country'].dropna().unique())
all_positions = sorted(ref['position'].dropna().unique())
all_players = sorted(ref['player_name'].dropna().unique())

# Filtros laterales
col1, col2, col3, col4 = st.columns(4)
with col1:
    ligas_seleccion = st.multiselect("Liga(s) a considerar", all_leagues, default=all_leagues)
with col2:
    paises_seleccion = st.multiselect("País(es) a considerar", all_countries, default=[])
with col3:
    posiciones_seleccion = st.multiselect("Posición(es)", all_positions, default=[])
with col4:
    jugador_ref = st.selectbox("Jugador de referencia", all_players)

top_n = st.slider("Número de recomendaciones", 3, 15, 5)

if st.button("🔍 Buscar jugadores similares", type="primary"):
    # Construir máscara de filtros
    mask = (ref['league_name'].isin(ligas_seleccion) if ligas_seleccion else True)
    if paises_seleccion:
        mask &= (ref['country'].isin(paises_seleccion))
    if posiciones_seleccion:
        mask &= (ref['position'].isin(posiciones_seleccion))
    
    df_candidates = ref[mask].copy()
    df_candidates = df_candidates[df_candidates['player_name'] != jugador_ref]
    
    if df_candidates.empty:
        st.warning("No hay jugadores que cumplan con los filtros seleccionados.")
    else:
        # Obtener vector del jugador referencia
        idx_ref = ref[ref['player_name'] == jugador_ref].index[0]
        vec_ref = scaler.transform([ref.loc[idx_ref, features].fillna(0).values])[0]
        X_candidates = scaler.transform(df_candidates[features].fillna(0).values)
        sim = cosine_similarity([vec_ref], X_candidates)[0]
        df_candidates['similitud'] = sim
        results = df_candidates.sort_values('similitud', ascending=False).head(top_n)
        
        # Agregar fila del jugador referencia
        ref_row = ref[ref['player_name'] == jugador_ref].copy()
        ref_row['similitud'] = 1.0
        # Asegurar las mismas columnas
        final_cols = ['player_name', 'team_name', 'league_name', 'country', 'position',
                      'goals_p90', 'assists_p90', 'shots_p90', 'passes_p90', 'tackles_p90', 'duels_won_p90', 'similitud']
        ref_row = ref_row[final_cols]
        results = results[final_cols]
        final_df = pd.concat([ref_row, results], ignore_index=True).round(2)
        
        st.success(f"Jugadores similares a **{jugador_ref}** (primera fila):")
        st.dataframe(final_df, use_container_width=True)
        
        # Gráfico comparativo de similitudes (opcional)
        fig_sim = px.bar(results, x='player_name', y='similitud', title=f'Similitud con {jugador_ref}',
                         labels={'player_name': 'Jugador', 'similitud': 'Similitud (coseno)'})
        st.plotly_chart(fig_sim)

# ============================================
# 5. RIESGO DE LESIÓN (DEMO)
# ============================================
st.header("⚠️ Riesgo de Lesión (Demo)")
st.info("Modelo conceptual. Con datos GPS del club se puede personalizar.")
c1, c2 = st.columns(2)
with c1:
    mins = st.number_input("Minutos en la semana", 0, 180, 90)
    sprints = st.number_input("Sprints", 0, 50, 12)
with c2:
    rest = st.number_input("Días de descanso", 1, 7, 3)
    acc = st.number_input("Aceleraciones", 0, 100, 30)
riesgo = np.clip(0.1 + (mins/180)*0.3 + (sprints/50)*0.2 + (1/rest)*0.1 + (acc/100)*0.1, 0, 1)
st.metric("Probabilidad de lesión", f"{riesgo:.1%}")
if riesgo > 0.6:
    st.error("⚠️ Alto riesgo")
elif riesgo > 0.3:
    st.warning("📉 Riesgo moderado")
else:
    st.success("✅ Riesgo bajo")
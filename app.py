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
# Tabla de posiciones (generada desde API si no existe)
# ============================================
try:
    df_tabla = pd.read_csv("data/tabla_posiciones.csv")
    st.header("📊 Tabla de Posiciones 2026")
    st.dataframe(df_tabla, use_container_width=True)
except:
    import requests
    API_KEY = "ebb8f00138af0df132bbda386d55981c"
    headers = {"x-apisports-key": API_KEY}
    url = "https://v3.football.api-sports.io/standings?league=239&season=2026"
    r = requests.get(url, headers=headers)
    if r.status_code == 200:
        data = r.json()
        if data['response']:
            standings_arrays = data['response'][0]['league']['standings']
            tabla_general = [arr for arr in standings_arrays if len(arr) == 20][0]
            equipos_tabla = []
            for t in tabla_general:
                equipos_tabla.append({
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
            df_tabla = pd.DataFrame(equipos_tabla)
            df_tabla.to_csv("data/tabla_posiciones.csv", index=False)
            st.header("📊 Tabla de Posiciones 2026")
            st.dataframe(df_tabla, use_container_width=True)
        else:
            st.warning("No se pudo obtener la tabla de posiciones")
    else:
        st.warning("No se pudo obtener la tabla de posiciones")

# ============================================
# Cargar datos unificados (todas las ligas)
# ============================================
df_players = pd.read_csv("data/players_unificado.csv")
scaler = joblib.load("models/scaler_unificado.pkl")
ref = joblib.load("models/referencia_unificado.pkl")
features = ['goals_p90', 'assists_p90', 'shots_p90', 'passes_p90', 'tackles_p90', 'duels_won_p90']
ref_full = ref.copy()

# ============================================
# Top goleadores de Colombia (Primera A)
# ============================================
st.header("⚽ Top Goleadores - Colombia (Primera A)")
df_colombia = df_players[df_players['league_name'] == 'Primera A']
top_col = df_colombia.nlargest(10, 'goals')[['player_name', 'team_name', 'goals']]
top_col.columns = ['Jugador', 'Equipo', 'Goles']
st.dataframe(top_col, use_container_width=True)
fig = px.bar(top_col, x='Jugador', y='Goles', title='Top 10 Goleadores Liga BetPlay 2026')
st.plotly_chart(fig)

# ============================================
# Scouting avanzado (con filtro por liga)
# ============================================
st.header("🔍 Scouting Avanzado - Encuentra jugadores similares")

# Obtener valores únicos (convertir a string para evitar TypeError)
all_leagues = sorted(ref_full['league_name'].astype(str).unique())
all_positions = sorted(ref_full['position'].astype(str).unique())
all_teams = sorted(ref_full['team_name'].astype(str).unique())
all_players = sorted(ref_full['player_name'].astype(str).unique())

# Selección por defecto: Primera A (Colombia)
default_league = ['Primera A'] if 'Primera A' in all_leagues else all_leagues[:1]

col1, col2, col3, col4 = st.columns(4)
with col1:
    liga_sel = st.multiselect("Liga(s)", all_leagues, default=default_league)
with col2:
    pos_sel = st.multiselect("Posición(es)", all_positions)
with col3:
    eq_sel = st.multiselect("Equipo(s)", all_teams)
with col4:
    jug_ref = st.selectbox("Jugador de referencia", all_players)

top_n = st.slider("Número de recomendaciones", 3, 15, 5)

if st.button("🔍 Buscar similares", type="primary"):
    mask = (ref_full['league_name'].astype(str).isin(liga_sel) if liga_sel else True) & \
           (ref_full['position'].astype(str).isin(pos_sel) if pos_sel else True) & \
           (ref_full['team_name'].astype(str).isin(eq_sel) if eq_sel else True)
    df_cand = ref_full[mask].copy()
    df_cand = df_cand[df_cand['player_name'] != jug_ref]
    if df_cand.empty:
        st.warning("No hay jugadores con esos filtros")
    else:
        idx = ref_full[ref_full['player_name'] == jug_ref].index[0]
        vec_ref = scaler.transform([ref_full.loc[idx, features].fillna(0).values])[0]
        X = scaler.transform(df_cand[features].fillna(0).values)
        sim = cosine_similarity([vec_ref], X)[0]
        df_cand['similitud'] = sim
        resultados = df_cand.sort_values('similitud', ascending=False).head(top_n)

        ref_data = ref_full[ref_full['player_name'] == jug_ref].copy()
        ref_data['similitud'] = 1.0
        cols = ['player_name', 'team_name', 'league_name', 'country', 'position',
                'goals_p90', 'assists_p90', 'shots_p90', 'passes_p90', 'tackles_p90', 'duels_won_p90', 'similitud']
        ref_data = ref_data[cols]
        resultados = resultados[cols]
        final = pd.concat([ref_data, resultados], ignore_index=True).round(2)
        st.success(f"Jugadores similares a **{jug_ref}** (primera fila):")
        st.dataframe(final, use_container_width=True)

# ============================================
# Riesgo de lesión (demo)
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
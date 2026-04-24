import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(layout="wide")
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

st.title("⚽ Futbol Analytics Colombia")
st.markdown("### Sistema de Scouting Inteligente y Prevención de Lesiones")
st.markdown("---")

# ============================================
# 1. TABLA DE POSICIONES
# ============================================
try:
    df_tabla = pd.read_csv("data/tabla_posiciones.csv")
    st.header("📊 Tabla de Posiciones 2026")
    st.dataframe(df_tabla, use_container_width=True)
except:
    st.warning("Tabla de posiciones no disponible (ejecuta 'actualizar_datos.py')")

# ============================================
# 2. TOP GOLEADORES (MANUAL)
# ============================================
st.header("⚽ Top Goleadores 2026")
goleadores = [
    {"Jugador": "Andrey Estupiñán", "Goles": 12, "Equipo": "Deportivo Pasto"},
    {"Jugador": "Yeison Guzmán", "Goles": 10, "Equipo": "América de Cali"},
    {"Jugador": "Alfredo Morelos", "Goles": 9, "Equipo": "Atlético Nacional"},
    {"Jugador": "Jorge Rivaldo", "Goles": 9, "Equipo": "Águilas Doradas"},
    {"Jugador": "Dayro Moreno", "Goles": 9, "Equipo": "Once Caldas"},
    {"Jugador": "Luis Muriel", "Goles": 8, "Equipo": "Junior"},
]
df_goles = pd.DataFrame(goleadores)
st.dataframe(df_goles, use_container_width=True)
fig = px.bar(df_goles, x='Jugador', y='Goles', title='Top Goleadores BetPlay 2026')
st.plotly_chart(fig)

# ============================================
# 3. SCOUTING AVANZADO (VERSIÓN CON MANEJO DE DUPLICADOS)
# ============================================
st.header("🔍 Scouting Avanzado - Encuentra jugadores similares en cualquier liga")

try:
    # Cargar datos
    df_players = pd.read_csv("data/players_internacional.csv").drop_duplicates(subset=['player_id'])
    scaler = joblib.load("models/scaler_internacional.pkl")
    ref = joblib.load("models/referencia_internacional.pkl")
    features = ['goles_p90', 'asistencias_p90', 'tiros_p90', 'pases_p90', 'entradas_p90', 'duelos_ganados_p90']

    # Verificar columnas necesarias en df_players
    required = ['player_name', 'position', 'league_name', 'country']
    for col in required:
        if col not in df_players.columns:
            st.error(f"Falta la columna '{col}' en players_internacional.csv. Re-ejecuta 'scout_mundial.py'")
            st.stop()

    # Merge para añadir metadata
    ref_full = ref.merge(df_players[required], on='player_name', how='left')

    # Después del merge, las columnas duplicadas tienen sufijos _x y _y.
    # Vamos a construir las columnas definitivas a partir de las que vienen de df_players (sufijo _y)
    # o si no existen, usar las originales.
    for col in ['league_name', 'country', 'position']:
        if f'{col}_y' in ref_full.columns:
            ref_full[col] = ref_full[f'{col}_y'].fillna('Sin especificar' if col != 'position' else 'No especificada')
        elif f'{col}_x' in ref_full.columns:
            ref_full[col] = ref_full[f'{col}_x'].fillna('Sin especificar' if col != 'position' else 'No especificada')
        elif col in ref_full.columns:
            ref_full[col] = ref_full[col].fillna('Sin especificar' if col != 'position' else 'No especificada')
        else:
            ref_full[col] = 'Sin especificar' if col != 'position' else 'No especificada'

    # Renombrar para claridad
    ref_full = ref_full.rename(columns={
        'position': 'posicion_raw',
        'league_name': 'liga_raw',
        'country': 'pais_raw'
    })
    ref_full['posicion'] = ref_full['posicion_raw'].fillna('No especificada').astype(str).replace('nan', 'No especificada')
    ref_full['liga'] = ref_full['liga_raw'].fillna('Sin especificar').astype(str).replace('nan', 'Sin especificar')
    ref_full['pais'] = ref_full['pais_raw'].fillna('Sin especificar').astype(str).replace('nan', 'Sin especificar')

    # Columnas auxiliares para ordenar sin problemas de tipos
    ref_full['posicion_clean'] = ref_full['posicion'].astype(str)
    ref_full['liga_clean'] = ref_full['liga'].astype(str)

    # Filtros
    col1, col2, col3 = st.columns(3)
    with col1:
        posiciones = sorted(ref_full['posicion_clean'].unique())
        default_pos = [p for p in posiciones if 'delantero' in p.lower() or 'attacker' in p.lower()]
        pos_seleccion = st.multiselect("Posición(es)", posiciones, default=default_pos[:1] if default_pos else posiciones[:1])
    with col2:
        ligas = sorted(ref_full['liga_clean'].unique())
        ligas_seleccion = st.multiselect("Liga(s)", ligas, default=ligas[:3] if len(ligas)>=3 else ligas)
    with col3:
        jugadores = sorted(ref_full['player_name'].unique())
        jugador_ref = st.selectbox("Jugador de referencia", jugadores)

    top_n = st.slider("Número de recomendaciones", 3, 15, 5)

    if st.button("🔍 Buscar similares", type="primary"):
        mask = (ref_full['posicion_clean'].isin(pos_seleccion)) & (ref_full['liga_clean'].isin(ligas_seleccion))
        df_candidatos = ref_full[mask].copy()
        df_candidatos = df_candidatos[df_candidatos['player_name'] != jugador_ref]

        if df_candidatos.empty:
            st.warning("No hay jugadores con esos filtros")
        else:
            idx_ref = ref_full[ref_full['player_name'] == jugador_ref].index[0]
            vector_ref = scaler.transform([ref_full.loc[idx_ref, features].fillna(0).values])[0]
            X_cand = scaler.transform(df_candidatos[features].fillna(0).values)
            sim = cosine_similarity([vector_ref], X_cand)[0]
            df_candidatos['similitud'] = sim
            resultados = df_candidatos.sort_values('similitud', ascending=False).head(top_n)
            cols_mostrar = ['player_name', 'liga', 'pais', 'posicion', 'goles_p90', 'asistencias_p90', 'similitud']
            st.success(f"Jugadores similares a **{jugador_ref}** en {', '.join(ligas_seleccion)} que juegan de {', '.join(pos_seleccion)}:")
            st.dataframe(resultados[cols_mostrar].round(2), use_container_width=True)

except Exception as e:
    st.error(f"Error en el módulo de scouting: {e}")
    st.info("Asegúrate de haber ejecutado 'scout_mundial.py' y 'entrenar_internacional.py' correctamente.")

# ============================================
# 4. RIESGO DE LESIÓN (DEMO)
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
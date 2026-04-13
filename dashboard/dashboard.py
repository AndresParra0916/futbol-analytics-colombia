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

@st.cache_data(ttl=3600)
def cargar_posiciones():
    try:
        return pd.read_csv('data/tabla_posiciones.csv')
    except:
        return None

@st.cache_data(ttl=3600)
def cargar_jugadores():
    try:
        return pd.read_csv('data/jugadores_stats.csv')
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
    if nombre not in ref['Nombre'].values:
        return None
    features = ['goles_p90', 'asistencias_p90', 'recuperaciones_p90', 
                'duelos_ganados_p90', 'pases_progresivos_p90']
    X = scaler.transform(ref[features].fillna(0).values)
    idx = ref[ref['Nombre'] == nombre].index[0]
    sim = cosine_similarity([X[idx]], X)[0]
    top_idx = np.argsort(sim)[::-1][1:top_n+1]
    return ref.iloc[top_idx][['Nombre', 'Equipo'] + features].assign(similitud=sim[top_idx].round(3))

opcion = st.sidebar.radio("Menú", ["📊 Tabla de Posiciones", "⚽ Top Goleadores", "🔍 Scouting", "⚠️ Riesgo de Lesión"])

if opcion == "📊 Tabla de Posiciones":
    st.header("Tabla de Posiciones")
    df = cargar_posiciones()
    if df is not None:
        st.dataframe(df, use_container_width=True)
    else:
        st.warning("Datos no disponibles. Ejecuta primero el workflow.")

elif opcion == "⚽ Top Goleadores":
    st.header("Top Goleadores")
    df = cargar_jugadores()
    if df is not None:
        top = df.nlargest(10, 'Goles')[['Nombre', 'Equipo', 'Goles', 'goles_p90']]
        st.dataframe(top, use_container_width=True)
        fig = px.bar(top, x='Nombre', y='Goles', title='Goles por Jugador')
        st.plotly_chart(fig)
    else:
        st.warning("Datos no disponibles.")

elif opcion == "🔍 Scouting":
    st.header("🔍 Motor de Scouting - Encontrar Jugadores Similares")
    df_jug = cargar_jugadores()
    scaler, ref = cargar_scouting()
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

elif opcion == "⚠️ Riesgo de Lesión":
    st.header("Predicción de Riesgo de Lesión")
    modelo_path = 'models/modelo_lesiones.pkl'
    if os.path.exists(modelo_path):
        modelo = joblib.load(modelo_path)
        st.success("✅ Usando modelo entrenado con datos GPS")
        with st.form(key="lesion_form"):
            col1, col2 = st.columns(2)
            with col1:
                minutos = st.number_input("Minutos en la semana", 0, 180, 90)
                sprints = st.number_input("Sprints", 0, 50, 12)
                aceleraciones = st.number_input("Aceleraciones", 0, 100, 30)
            with col2:
                distancia = st.number_input("Distancia total (m)", 0, 20000, 9000)
                descanso = st.number_input("Días de descanso", 1, 7, 3)
                acwr = st.number_input("ACWR (carga aguda/crónica)", 0.5, 2.5, 1.2, 0.05)
            submitted = st.form_submit_button("Calcular riesgo")
            if submitted:
                # El modelo espera 6 características: minutos, acwr, sprints, aceleraciones, distancia, descanso
                entrada = np.array([[minutos, acwr, sprints, aceleraciones, distancia, descanso]])
                proba = modelo.predict_proba(entrada)[0][1]
                st.metric("Probabilidad de lesión en la próxima semana", f"{proba:.1%}")
                if proba > 0.6:
                    st.error("⚠️ Alto riesgo. Considera reducir carga o aumentar descanso.")
                elif proba > 0.3:
                    st.warning("📉 Riesgo moderado. Monitorear fatiga.")
                else:
                    st.success("✅ Riesgo bajo.")
    else:
        st.warning("Modelo no encontrado. Asegúrate de que 'modelo_lesiones.pkl' esté en la carpeta 'models'.")

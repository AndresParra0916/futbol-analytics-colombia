import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import joblib

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

opcion = st.sidebar.radio("Menú", ["📊 Tabla de Posiciones", "⚽ Top Goleadores", "⚠️ Riesgo de Lesión"])

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

elif opcion == "⚠️ Riesgo de Lesión":
    st.header("Predicción de Riesgo de Lesión")
    modelo_path = 'models/modelo_lesiones_gps.pkl'
    if os.path.exists(modelo_path):
        modelo = joblib.load(modelo_path)
        st.success("✅ Usando modelo entrenado con datos GPS")
        col1, col2 = st.columns(2)
        with col1:
            minutos = st.number_input("Minutos en la semana", 0, 180, 90)
            sprints = st.number_input("Sprints", 0, 50, 12)
            aceleraciones = st.number_input("Aceleraciones", 0, 100, 30)
        with col2:
            distancia = st.number_input("Distancia total (m)", 0, 20000, 9000)
            descanso = st.number_input("Días de descanso", 1, 7, 3)
            acwr = st.number_input("ACWR (carga aguda/crónica)", 0.5, 2.5, 1.2, 0.05)
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
        st.warning("Modelo GPS no encontrado. Ejecuta: python integrar_gps.py data/datos_gps_prueba.csv")

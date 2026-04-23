import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os

print("1. Cargando datos...")

# Leer plantilla maestra
plantilla = pd.read_csv('data/plantilla_maestra.csv', encoding='utf-8')
print(f"   Plantilla: {len(plantilla)} jugadores")

# Leer estadísticas con separador correcto
estadisticas = pd.read_csv('data/estadisticas_jugadores.csv', sep=';', encoding='utf-8')
print(f"   Estadísticas: {len(estadisticas)} registros")
print(f"   Columnas encontradas: {list(estadisticas.columns)}")

if estadisticas.empty:
    print("❌ No hay datos. Agrega registros.")
    exit()

# Convertir columnas numéricas (las que existen)
columnas_numericas = ['minutos', 'goles', 'asistencias']
for col in columnas_numericas:
    if col in estadisticas.columns:
        estadisticas[col] = pd.to_numeric(estadisticas[col], errors='coerce').fillna(0)

# Lista de columnas adicionales que podrías tener (opcionales)
opcionales = ['recuperaciones', 'duelos_ganados', 'pases_progresivos']
for col in opcionales:
    if col not in estadisticas.columns:
        estadisticas[col] = 0  # crear columna con ceros

# 2. Agrupar por jugador
stats_jugador = estadisticas.groupby(['id_jugador', 'nombre', 'equipo']).agg({
    'minutos': 'sum',
    'goles': 'sum',
    'asistencias': 'sum',
    'recuperaciones': 'sum',
    'duelos_ganados': 'sum',
    'pases_progresivos': 'sum'
}).reset_index()

# Evitar división por cero
stats_jugador['minutos'] = stats_jugador['minutos'].replace(0, 1)

# Calcular métricas por 90 minutos
stats_jugador['goles_p90'] = stats_jugador['goles'] / (stats_jugador['minutos'] / 90)
stats_jugador['asistencias_p90'] = stats_jugador['asistencias'] / (stats_jugador['minutos'] / 90)
stats_jugador['recuperaciones_p90'] = stats_jugador['recuperaciones'] / (stats_jugador['minutos'] / 90)
stats_jugador['duelos_ganados_p90'] = stats_jugador['duelos_ganados'] / (stats_jugador['minutos'] / 90)
stats_jugador['pases_progresivos_p90'] = stats_jugador['pases_progresivos'] / (stats_jugador['minutos'] / 90)

stats_jugador = stats_jugador.fillna(0)

# Guardar archivo para el dashboard
stats_jugador.to_csv('data/jugadores_stats.csv', index=False)
print(f"✅ Métricas calculadas para {len(stats_jugador)} jugadores.")

# 3. Modelo de scouting (usando solo las métricas que tienen variación)
features = ['goles_p90', 'asistencias_p90', 'recuperaciones_p90', 
            'duelos_ganados_p90', 'pases_progresivos_p90']

# Verificar si alguna columna tiene todos los valores iguales (cero)
X = stats_jugador[features].fillna(0).values
if X.std(axis=0).sum() == 0:
    print("⚠️ Las métricas adicionales son todas cero. Usando solo goles y asistencias para el scouting.")
    features = ['goles_p90', 'asistencias_p90']
    X = stats_jugador[features].fillna(0).values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

joblib.dump(scaler, 'models/scaler_scouting.pkl')
joblib.dump(stats_jugador[['nombre', 'equipo'] + features], 'models/referencia_scouting.pkl')
print("✅ Modelo de scouting actualizado.")

# 4. Función de recomendación
def recomendar_similares(nombre, top_n=5):
    scaler = joblib.load('models/scaler_scouting.pkl')
    ref = joblib.load('models/referencia_scouting.pkl')
    if nombre not in ref['nombre'].values:
        return None
    idx = ref[ref['nombre'] == nombre].index[0]
    X = scaler.transform(ref[features].fillna(0).values)
    sim = cosine_similarity([X[idx]], X)[0]
    top_idx = np.argsort(sim)[::-1][1:top_n+1]
    return ref.iloc[top_idx][['nombre', 'equipo'] + features].assign(similitud=sim[top_idx].round(3))

# Ejemplo (si existe algún jugador)
if len(stats_jugador) > 0:
    ejemplo = stats_jugador.iloc[0]['nombre']
    print(f"\n🔍 Jugadores similares a {ejemplo}:")
    sim = recomendar_similares(ejemplo)
    if sim is not None:
        print(sim.to_string(index=False))
    else:
        print("No se encontraron recomendaciones.")
else:
    print("No hay jugadores con estadísticas para entrenar el modelo.")

print("\n🎉 Proceso completado. Ahora puedes ejecutar el dashboard.")
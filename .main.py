import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os

# Cargar datos manuales
print("Cargando plantilla...")
plantilla = pd.read_csv('data/plantilla_maestra.csv', encoding='utf-8')
print("Cargando estadísticas...")
estadisticas = pd.read_csv('data/estadisticas_jugadores.csv', encoding='utf-8')

if estadisticas.empty:
    print("No hay datos en estadisticas_jugadores.csv. Agrega registros y vuelve a ejecutar.")
    exit()

# Agrupar por jugador
stats_jugador = estadisticas.groupby(['id_jugador', 'nombre', 'equipo']).agg({
    'minutos': 'sum',
    'goles': 'sum',
    'asistencias': 'sum',
    'recuperaciones': 'sum',
    'duelos_ganados': 'sum',
    'pases_progresivos': 'sum'
}).reset_index()

# Calcular por 90 minutos
stats_jugador['goles_p90'] = stats_jugador['goles'] / (stats_jugador['minutos'] / 90)
stats_jugador['asistencias_p90'] = stats_jugador['asistencias'] / (stats_jugador['minutos'] / 90)
stats_jugador['recuperaciones_p90'] = stats_jugador['recuperaciones'] / (stats_jugador['minutos'] / 90)
stats_jugador['duelos_ganados_p90'] = stats_jugador['duelos_ganados'] / (stats_jugador['minutos'] / 90)
stats_jugador['pases_progresivos_p90'] = stats_jugador['pases_progresivos'] / (stats_jugador['minutos'] / 90)
stats_jugador = stats_jugador.fillna(0)

stats_jugador.to_csv('data/jugadores_stats.csv', index=False)
print(f"✅ Métricas calculadas para {len(stats_jugador)} jugadores.")

# Modelo de scouting
features = ['goles_p90', 'asistencias_p90', 'recuperaciones_p90', 'duelos_ganados_p90', 'pases_progresivos_p90']
X = stats_jugador[features].fillna(0).values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, 'models/scaler_scouting.pkl')
joblib.dump(stats_jugador[['nombre', 'equipo'] + features], 'models/referencia_scouting.pkl')
print("✅ Modelo de scouting actualizado.")

# Recomendación de ejemplo
def recomendar(nombre, top=5):
    scaler = joblib.load('models/scaler_scouting.pkl')
    ref = joblib.load('models/referencia_scouting.pkl')
    if nombre not in ref['nombre'].values:
        return None
    idx = ref[ref['nombre'] == nombre].index[0]
    X = scaler.transform(ref[features].fillna(0).values)
    sim = cosine_similarity([X[idx]], X)[0]
    top_idx = np.argsort(sim)[::-1][1:top+1]
    return ref.iloc[top_idx][['nombre', 'equipo'] + features].assign(similitud=sim[top_idx].round(3))

print("\n🔍 Ejemplo: Jugadores similares a Alfredo Morelos")
sim = recomendar('Alfredo Morelos')
if sim is not None:
    print(sim.to_string(index=False))
else:
    print("Alfredo Morelos no está en los datos. Prueba con otro jugador.")

print("\n🎉 Proceso completado.")
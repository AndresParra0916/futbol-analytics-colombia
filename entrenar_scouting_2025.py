import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

print("Cargando datos...")
plantilla = pd.read_csv('data/plantilla_api_2025.csv')
estadisticas = pd.read_csv('data/estadisticas_api_2025_prueba.csv')
print(f"Plantilla: {len(plantilla)} jugadores")
print(f"Estadísticas: {len(estadisticas)} registros")

# Agrupar por jugador
stats_jugador = estadisticas.groupby(['player_id', 'player_name']).agg({
    'minutes': 'sum',
    'goals': 'sum',
    'assists': 'sum',
    'shots': 'sum',
    'passes': 'sum',
    'tackles': 'sum',
    'duels_won': 'sum'
}).reset_index()

stats_jugador['minutes'] = stats_jugador['minutes'].replace(0, 1)

# Métricas por 90 minutos
stats_jugador['goles_p90'] = stats_jugador['goals'] / (stats_jugador['minutes'] / 90)
stats_jugador['asistencias_p90'] = stats_jugador['assists'] / (stats_jugador['minutes'] / 90)
stats_jugador['tiros_p90'] = stats_jugador['shots'] / (stats_jugador['minutes'] / 90)
stats_jugador['pases_p90'] = stats_jugador['passes'] / (stats_jugador['minutes'] / 90)
stats_jugador['entradas_p90'] = stats_jugador['tackles'] / (stats_jugador['minutes'] / 90)
stats_jugador['duelos_ganados_p90'] = stats_jugador['duels_won'] / (stats_jugador['minutes'] / 90)

stats_jugador = stats_jugador.fillna(0)

# Características para el scouting
features = ['goles_p90', 'asistencias_p90', 'tiros_p90', 'pases_p90', 'entradas_p90', 'duelos_ganados_p90']
X = stats_jugador[features].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

joblib.dump(scaler, 'models/scaler_scouting_2025.pkl')
joblib.dump(stats_jugador[['player_name'] + features], 'models/referencia_scouting_2025.pkl')
print("✅ Modelo de scouting 2025 entrenado")

# Recomendación de ejemplo
def recomendar(nombre, top=5):
    ref = joblib.load('models/referencia_scouting_2025.pkl')
    scaler = joblib.load('models/scaler_scouting_2025.pkl')
    if nombre not in ref['player_name'].values:
        return None
    idx = ref[ref['player_name'] == nombre].index[0]
    X = scaler.transform(ref[features].fillna(0).values)
    sim = cosine_similarity([X[idx]], X)[0]
    top_idx = np.argsort(sim)[::-1][1:top+1]
    return ref.iloc[top_idx][['player_name'] + features].assign(similitud=sim[top_idx].round(3))

print("\n🔍 Ejemplo: Jugadores similares a 'Dayro Moreno'")
sim = recomendar('Dayro Moreno')
if sim is not None:
    print(sim.to_string(index=False))
else:
    print("No se encontró a Dayro Moreno en los datos de 2025")

print("\n🎉 Proceso completado.")
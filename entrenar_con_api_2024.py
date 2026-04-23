import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import joblib

print("Cargando datos...")

# Plantilla
plantilla = pd.read_csv("data/plantilla_api_2024.csv")
print(f"Plantilla: {len(plantilla)} jugadores")

# Estadísticas de partidos de prueba
estadisticas = pd.read_csv("data/estadisticas_api_2024_prueba.csv")
print(f"Estadísticas: {len(estadisticas)} registros")

if estadisticas.empty:
    print("No hay estadísticas. Ejecuta primero api_stats_2024_prueba.py")
    exit()

# Agrupar por jugador y sumar
stats_jugador = estadisticas.groupby(["player_id", "player_name"]).agg({
    "minutes": "sum",
    "goals": "sum",
    "assists": "sum",
    "shots": "sum",
    "passes": "sum",
    "tackles": "sum",
    "duels_won": "sum"
}).reset_index()

# Evitar división por cero
stats_jugador["minutes"] = stats_jugador["minutes"].replace(0, 1)

# Calcular por 90 minutos
stats_jugador["goles_p90"] = stats_jugador["goals"] / (stats_jugador["minutes"] / 90)
stats_jugador["asistencias_p90"] = stats_jugador["assists"] / (stats_jugador["minutes"] / 90)
stats_jugador["tiros_p90"] = stats_jugador["shots"] / (stats_jugador["minutes"] / 90)
stats_jugador["pases_p90"] = stats_jugador["passes"] / (stats_jugador["minutes"] / 90)
stats_jugador["entradas_p90"] = stats_jugador["tackles"] / (stats_jugador["minutes"] / 90)
stats_jugador["duelos_ganados_p90"] = stats_jugador["duels_won"] / (stats_jugador["minutes"] / 90)

stats_jugador = stats_jugador.fillna(0)

# Seleccionar características para el scouting
features = ["goles_p90", "asistencias_p90", "tiros_p90", "pases_p90", "entradas_p90", "duelos_ganados_p90"]
X = stats_jugador[features].values

# Normalizar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Guardar modelo
joblib.dump(scaler, "models/scaler_scouting_api.pkl")
joblib.dump(stats_jugador[["player_name"] + features], "models/referencia_scouting_api.pkl")
print("✅ Modelo de scouting entrenado con datos reales de la API (2024).")

# Recomendación de ejemplo
def recomendar_similares(nombre, top_n=5):
    scaler = joblib.load("models/scaler_scouting_api.pkl")
    ref = joblib.load("models/referencia_scouting_api.pkl")
    if nombre not in ref["player_name"].values:
        return None
    idx = ref[ref["player_name"] == nombre].index[0]
    X = scaler.transform(ref[features].fillna(0).values)
    sim = cosine_similarity([X[idx]], X)[0]
    top_idx = np.argsort(sim)[::-1][1:top_n+1]
    return ref.iloc[top_idx][["player_name"] + features].assign(similitud=sim[top_idx].round(3))

print("\n🔍 Ejemplo: Jugadores similares a 'Alfredo Morelos' (si existe en los datos):")
sim = recomendar_similares("Alfredo Morelos")
if sim is not None:
    print(sim.to_string(index=False))
else:
    print("No se encontró a Alfredo Morelos en los datos de prueba. Prueba con otro jugador.")

print("\n🎉 Proceso completado.")
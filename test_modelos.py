import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

print("Cargando modelos...")
ref = joblib.load('models/referencia_scouting_2025.pkl')
scaler = joblib.load('models/scaler_scouting_2025.pkl')
features = ['goles_p90','asistencias_p90','tiros_p90','pases_p90','entradas_p90','duelos_ganados_p90']

print("Columnas:", ref.columns.tolist())
print("Primeros jugadores:", ref['player_name'].head(3).tolist())

# Buscar a Dayro Moreno
idx = ref[ref['player_name'] == 'Dayro Moreno'].index
if len(idx) > 0:
    print("Dayro Moreno encontrado en índice", idx[0])
    X = scaler.transform(ref[features].fillna(0).values)
    sim = cosine_similarity([X[idx[0]]], X)[0]
    top = np.argsort(sim)[::-1][1:6]
    print("Jugadores similares a Dayro Moreno:")
    for i, t in enumerate(top):
        print(f"{i+1}. {ref.iloc[t]['player_name']} (similitud: {sim[t]:.3f})")
else:
    print("Dayro Moreno no está en los datos. Probando con el primer jugador:")
    primer_jugador = ref.iloc[0]['player_name']
    print("Primer jugador:", primer_jugador)
    idx = ref[ref['player_name'] == primer_jugador].index[0]
    X = scaler.transform(ref[features].fillna(0).values)
    sim = cosine_similarity([X[idx]], X)[0]
    top = np.argsort(sim)[::-1][1:6]
    print(f"Jugadores similares a {primer_jugador}:")
    for i, t in enumerate(top):
        print(f"{i+1}. {ref.iloc[t]['player_name']} (similitud: {sim[t]:.3f})")
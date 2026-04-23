import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# Cargar modelo y referencia
scaler = joblib.load('models/scaler_scouting_2025.pkl')
ref = joblib.load('models/referencia_scouting_2025.pkl')
features = ['goles_p90', 'asistencias_p90', 'tiros_p90', 'pases_p90', 'entradas_p90', 'duelos_ganados_p90']

def recomendar(nombre, top=5):
    if nombre not in ref['player_name'].values:
        return None
    idx = ref[ref['player_name'] == nombre].index[0]
    X = scaler.transform(ref[features].fillna(0).values)
    sim = cosine_similarity([X[idx]], X)[0]
    top_idx = np.argsort(sim)[::-1][1:top+1]
    return ref.iloc[top_idx][['player_name'] + features].assign(similitud=sim[top_idx].round(3))

# Lista de jugadores clave de Santa Fe (ajusta según plantilla real de 2025)
# Puedes cambiar estos nombres por los que tengas en tus datos
jugadores_santa_fe = ['Hugo Rodallega', 'Omar Fernández', 'Jersson González', 'Daniel Torres']

print("=" * 60)
print("INFORME DE SCOUTING - INDEPENDIENTE SANTA FE")
print("=" * 60)
print("Jugadores similares a los referentes del club:\n")

for jugador in jugadores_santa_fe:
    print(f"\n🔍 Basado en: {jugador}")
    similares = recomendar(jugador)
    if similares is not None:
        print(similares.to_string(index=False))
    else:
        print(f"No se encontró a {jugador} en los datos. Intenta con otro nombre.")

# Generar recomendaciones de fichajes (top 5 del campeonato)
print("\n" + "=" * 60)
print("TOP 5 JUGADORES DESTACADOS DE LA LIGA (por goles_p90)")
print("=" * 60)
top_goleadores = ref.nlargest(5, 'goles_p90')[['player_name', 'goles_p90']]
print(top_goleadores.to_string(index=False))

print("\n" + "=" * 60)
print("TOP 5 ASISTENTES (por asistencias_p90)")
print("=" * 60)
top_asistentes = ref.nlargest(5, 'asistencias_p90')[['player_name', 'asistencias_p90']]
print(top_asistentes.to_string(index=False))

# Guardar informe en CSV
informe = []
for jugador in jugadores_santa_fe:
    similares = recomendar(jugador)
    if similares is not None:
        for _, row in similares.iterrows():
            informe.append({
                'Jugador_Referencia': jugador,
                'Jugador_Recomendado': row['player_name'],
                'Similitud': row['similitud'],
                'Goles_p90': row['goles_p90'],
                'Asistencias_p90': row['asistencias_p90']
            })

if informe:
    df_informe = pd.DataFrame(informe)
    df_informe.to_csv('data/informe_scouting_santa_fe.csv', index=False)
    print("\n✅ Informe guardado en 'data/informe_scouting_santa_fe.csv'")
else:
    print("\n⚠️ No se pudo generar el informe. Verifica los nombres de los jugadores.")

print("\n🎉 Listo para tu reunión. Puedes abrir el CSV con Excel y presentar los resultados.")
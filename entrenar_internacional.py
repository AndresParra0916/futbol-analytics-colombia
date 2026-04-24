import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

print("=== ENTRENANDO MODELO DE SCOUTING INTERNACIONAL ===")

df_players = pd.read_csv("data/players_internacional.csv")
df_stats = pd.read_csv("data/stats_internacional.csv")

print(f"Jugadores: {len(df_players)}")
print(f"Estadísticas: {len(df_stats)}")

stats_agrup = df_stats.groupby(['player_id', 'player_name', 'league_name', 'country']).agg({
    'minutes': 'sum',
    'goals': 'sum',
    'assists': 'sum',
    'shots': 'sum',
    'passes': 'sum',
    'tackles': 'sum',
    'duels_won': 'sum'
}).reset_index()

stats_agrup['minutes'] = stats_agrup['minutes'].replace(0, 1)

stats_agrup['goles_p90'] = stats_agrup['goals'] / (stats_agrup['minutes'] / 90)
stats_agrup['asistencias_p90'] = stats_agrup['assists'] / (stats_agrup['minutes'] / 90)
stats_agrup['tiros_p90'] = stats_agrup['shots'] / (stats_agrup['minutes'] / 90)
stats_agrup['pases_p90'] = stats_agrup['passes'] / (stats_agrup['minutes'] / 90)
stats_agrup['entradas_p90'] = stats_agrup['tackles'] / (stats_agrup['minutes'] / 90)
stats_agrup['duelos_ganados_p90'] = stats_agrup['duels_won'] / (stats_agrup['minutes'] / 90)

stats_agrup = stats_agrup.fillna(0)

features = ['goles_p90', 'asistencias_p90', 'tiros_p90', 'pases_p90', 'entradas_p90', 'duelos_ganados_p90']
X = stats_agrup[features].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

joblib.dump(scaler, 'models/scaler_internacional.pkl')
joblib.dump(stats_agrup[['player_name', 'league_name', 'country'] + features], 'models/referencia_internacional.pkl')
print("✅ Modelo internacional guardado")
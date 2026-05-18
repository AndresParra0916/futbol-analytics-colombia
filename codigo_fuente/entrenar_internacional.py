import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

print("=== ENTRENANDO MODELO CON DATOS 2026 ===\n")

df = pd.read_csv("data/players_stats_2026_completo.csv")
print(f"Jugadores con minutos: {len(df)}")

# Calcular métricas por 90 minutos
for col in ['goals', 'assists', 'shots', 'passes', 'tackles', 'duels_won']:
    df[f'{col}_p90'] = df[col] / (df['minutes'] / 90)
df = df.fillna(0)

features = ['goals_p90', 'assists_p90', 'shots_p90', 'passes_p90', 'tackles_p90', 'duels_won_p90']
X = df[features].values

scaler = StandardScaler()
scaler.fit(X)

joblib.dump(scaler, 'models/scaler_2026.pkl')
joblib.dump(df[['player_name', 'team_name', 'position'] + features], 'models/referencia_2026.pkl')
print("✅ Modelos guardados (scaler_2026.pkl, referencia_2026.pkl)")
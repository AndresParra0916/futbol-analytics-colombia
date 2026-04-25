import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

print("=== ENTRENANDO MODELO UNIFICADO (SUDAMÉRICA + MÉXICO) ===\n")

df = pd.read_csv("data/players_unificado.csv")
print(f"Jugadores con minutos: {len(df)}")

# Calcular métricas por 90 minutos
for col in ['goals', 'assists', 'shots', 'passes', 'tackles', 'duels_won']:
    df[f'{col}_p90'] = df[col] / (df['minutes'] / 90)
df = df.fillna(0)

features = ['goals_p90', 'assists_p90', 'shots_p90', 'passes_p90', 'tackles_p90', 'duels_won_p90']
X = df[features].values

scaler = StandardScaler()
scaler.fit(X)

joblib.dump(scaler, 'models/scaler_unificado.pkl')
joblib.dump(df[['player_name', 'team_name', 'league_name', 'country', 'position'] + features], 'models/referencia_unificado.pkl')
print("✅ Modelo guardado (scaler_unificado.pkl, referencia_unificado.pkl)")
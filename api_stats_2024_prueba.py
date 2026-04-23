import requests
import pandas as pd
import time

API_KEY = "ebb8f00138af0df132bbda386d55981c"
headers = {"x-apisports-key": API_KEY}

df_fixtures = pd.read_csv("data/fixtures_2024.csv")
fixture_ids = df_fixtures["fixture_id"].tolist()[:40]   # 40 partidos

all_stats = []

for i, fid in enumerate(fixture_ids):
    print(f"Procesando fixture {i+1} de 40 (ID: {fid})...")
    url = f"https://v3.football.api-sports.io/fixtures/players?fixture={fid}"
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        for team_data in data["response"]:
            team_id = team_data["team"]["id"]
            for player_data in team_data["players"]:
                player = player_data["player"]
                stats = player_data["statistics"][0]
                all_stats.append({
                    "fixture_id": fid,
                    "team_id": team_id,
                    "player_id": player["id"],
                    "player_name": player["name"],
                    "minutes": stats.get("minutes", 0),
                    "goals": stats.get("goals", {}).get("total", 0),
                    "assists": stats.get("goals", {}).get("assists", 0),
                    "shots": stats.get("shots", {}).get("total", 0),
                    "passes": stats.get("passes", {}).get("total", 0),
                    "tackles": stats.get("tackles", {}).get("total", 0),
                    "duels_won": stats.get("duels", {}).get("won", 0),
                    "rating": stats.get("rating", 0)
                })
    else:
        print(f"❌ Error {response.status_code} en fixture {fid}")
    time.sleep(6)   # Esperar 6 segundos para no exceder 10 peticiones/minuto

df = pd.DataFrame(all_stats)
df.to_csv("data/estadisticas_api_2024_prueba.csv", index=False)
print(f"✅ Se guardaron {len(df)} registros de estadísticas (prueba con 40 partidos).")
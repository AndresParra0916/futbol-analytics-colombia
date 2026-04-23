import requests
import pandas as pd

API_KEY = "ebb8f00138af0df132bbda386d55981c"
headers = {"x-apisports-key": API_KEY}

LEAGUE_ID = 239
SEASON = 2025   # Cambiado de 2026 a 2025

url = f"https://v3.football.api-sports.io/teams?league={LEAGUE_ID}&season={SEASON}"
response = requests.get(url, headers=headers)

if response.status_code == 200:
    data = response.json()
    if data["response"]:
        equipos = []
        for team in data["response"]:
            equipos.append({
                "id": team["team"]["id"],
                "nombre": team["team"]["name"],
                "codigo": team["team"]["code"],
                "pais": team["team"]["country"]
            })
        df = pd.DataFrame(equipos)
        df.to_csv("data/equipos_api.csv", index=False)
        print(f"✅ Se guardaron {len(df)} equipos (temporada {SEASON}) en 'data/equipos_api.csv'")
        print(df.head())
    else:
        print(f"⚠️ No se encontraron equipos para temporada {SEASON}.")
else:
    print(f"❌ Error {response.status_code}: {response.text}")
import requests
import pandas as pd
import time

API_KEY = "ebb8f00138af0df132bbda386d55981c"
headers = {"x-apisports-key": API_KEY}

df_equipos = pd.read_csv("data/equipos_api.csv")
team_ids = df_equipos["id"].tolist()
team_names = dict(zip(df_equipos["id"], df_equipos["nombre"]))

all_players = []

for idx, team_id in enumerate(team_ids):
    # Esperar 6 segundos entre cada petición (10 peticiones/minuto = 6 segundos)
    if idx > 0:
        time.sleep(6)
    
    url = f"https://v3.football.api-sports.io/players/squads?team={team_id}"
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        if data["response"]:
            squad = data["response"][0]["players"]
            for player in squad:
                all_players.append({
                    "team_id": team_id,
                    "team_name": team_names[team_id],
                    "player_id": player["id"],
                    "player_name": player["name"],
                    "age": player.get("age"),
                    "number": player.get("number"),
                    "position": player.get("position")
                })
        else:
            print(f"⚠️ No se encontró plantilla para equipo {team_id} ({team_names[team_id]})")
    elif response.status_code == 429:
        print(f"⏳ Demasiadas solicitudes. Esperando 60 segundos antes de reintentar equipo {team_id}...")
        time.sleep(60)
        # Reintentar una vez
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            if data["response"]:
                squad = data["response"][0]["players"]
                for player in squad:
                    all_players.append({
                        "team_id": team_id,
                        "team_name": team_names[team_id],
                        "player_id": player["id"],
                        "player_name": player["name"],
                        "age": player.get("age"),
                        "number": player.get("number"),
                        "position": player.get("position")
                    })
            else:
                print(f"⚠️ Reintento fallido: sin plantilla para {team_names[team_id]}")
        else:
            print(f"❌ Reintento también falló para equipo {team_id}: {response.status_code}")
    else:
        print(f"❌ Error {response.status_code} para equipo {team_id}")

    # Mostrar progreso
    print(f"Procesado {idx+1} de {len(team_ids)} equipos")

df = pd.DataFrame(all_players)
df.to_csv("data/plantilla_api_2024.csv", index=False)
print(f"✅ Se guardaron {len(df)} jugadores en 'data/plantilla_api_2024.csv'")
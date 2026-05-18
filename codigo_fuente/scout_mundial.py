import requests
import pandas as pd
import time

API_KEY = "ebb8f00138af0df132bbda386d55981c"
HEADERS = {"x-apisports-key": API_KEY}
SEASON = 2025          # Usamos 2025 (datos disponibles). Si tu plan de pago tiene 2026, cámbialo a 2026.
LEAGUE_ID = 239        # Liga colombiana

print("=== INICIANDO DESCARGA DE ESTADÍSTICAS DE JUGADORES (TEMPORADA {}) ===\n".format(SEASON))

# 1. Obtener equipos de la liga
url_teams = f"https://v3.football.api-sports.io/teams?league={LEAGUE_ID}&season={SEASON}"
r = requests.get(url_teams, headers=HEADERS)
if r.status_code != 200:
    print("Error obteniendo equipos:", r.status_code)
    exit()
data = r.json()
teams = data["response"]
print(f"✅ {len(teams)} equipos encontrados.\n")

all_players_data = []

# 2. Para cada equipo, obtener jugadores y estadísticas
for idx, team in enumerate(teams, 1):
    team_id = team["team"]["id"]
    team_name = team["team"]["name"]
    print(f"Procesando equipo {idx}/{len(teams)}: {team_name}")

    # Obtener plantilla del equipo
    url_squad = f"https://v3.football.api-sports.io/players/squads?team={team_id}"
    r_squad = requests.get(url_squad, headers=HEADERS)
    if r_squad.status_code != 200:
        print(f"   ⚠️ Error al obtener plantilla de {team_name}")
        continue
    squad_data = r_squad.json()
    if not squad_data["response"]:
        continue
    players = squad_data["response"][0]["players"]

    # Para cada jugador, obtener estadísticas de la temporada
    for player in players:
        player_id = player["id"]
        player_name = player["name"]
        position = player.get("position", "No especificada")

        url_stats = f"https://v3.football.api-sports.io/players?season={SEASON}&player={player_id}"
        r_stats = requests.get(url_stats, headers=HEADERS)
        if r_stats.status_code != 200:
            print(f"   ⚠️ Error al obtener estadísticas de {player_name}")
            continue
        stats_data = r_stats.json()
        if not stats_data["response"]:
            continue

        # Buscar las estadísticas correspondientes a la liga colombiana
        player_stats = None
        for entry in stats_data["response"]:
            if entry['statistics'][0]['league']['id'] == LEAGUE_ID:
                player_stats = entry['statistics'][0]
                break

        if not player_stats:
            # El jugador no tiene estadísticas en esta liga
            continue

        # Extraer datos relevantes
        games = player_stats.get('games', {})
        minutes = games.get('minutes', 0)
        if minutes == 0:
            # Si no tiene minutos, ignorar (no hay datos de rendimiento)
            continue

        goals = player_stats.get('goals', {}).get('total', 0)
        assists = player_stats.get('goals', {}).get('assists', 0)
        shots = player_stats.get('shots', {}).get('total', 0)
        passes = player_stats.get('passes', {}).get('total', 0)
        tackles = player_stats.get('tackles', {}).get('total', 0)
        duels_won = player_stats.get('duels', {}).get('won', 0)

        all_players_data.append({
            "player_id": player_id,
            "player_name": player_name,
            "team_id": team_id,
            "team_name": team_name,
            "position": position,
            "minutes": minutes,
            "goals": goals,
            "assists": assists,
            "shots": shots,
            "passes": passes,
            "tackles": tackles,
            "duels_won": duels_won
        })

    time.sleep(1)  # Pequeña pausa para no saturar la API

# Guardar a CSV
df = pd.DataFrame(all_players_data)
df.to_csv("data/players_stats_2025.csv", index=False)
print(f"\n✅ Datos guardados: {len(df)} jugadores con minutos > 0.")
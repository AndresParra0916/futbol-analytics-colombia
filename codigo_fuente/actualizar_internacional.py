import requests
import pandas as pd
import time

API_KEY = "ebb8f00138af0df132bbda386d55981c"
HEADERS = {"x-apisports-key": API_KEY}
SEASON = 2026

# ============================================
# LISTA DE LIGAS PARA SCOUTING INTERNACIONAL
# ============================================
# ID de liga, nombre, país
LEAGUES = [
    {"id": 71, "name": "Brasileirão", "country": "Brasil"},
    {"id": 128, "name": "Liga Profesional Argentina", "country": "Argentina"},
    {"id": 262, "name": "Liga MX", "country": "México"},
    {"id": 140, "name": "La Liga", "country": "España"},
    {"id": 39, "name": "Premier League", "country": "Inglaterra"},
    {"id": 135, "name": "Serie A", "country": "Italia"},
    {"id": 78, "name": "Bundesliga", "country": "Alemania"},
    {"id": 61, "name": "Ligue 1", "country": "Francia"},
]

print("=== INICIANDO DESCARGA DE DATOS INTERNACIONALES ===")

all_players = []
all_stats = []

for league in LEAGUES:
    league_id = league["id"]
    league_name = league["name"]
    country = league["country"]
    print(f"\n📌 Procesando {league_name} ({country})...")

    # 1. Obtener equipos de la liga
    url_teams = f"https://v3.football.api-sports.io/teams?league={league_id}&season={SEASON}"
    r = requests.get(url_teams, headers=HEADERS)
    if r.status_code != 200:
        print(f"   ❌ Error obteniendo equipos: {r.status_code}")
        continue
    data_teams = r.json()
    teams = data_teams["response"]
    print(f"   ✅ {len(teams)} equipos encontrados")

    # 2. Por cada equipo, obtener plantilla
    for team in teams:
        team_id = team["team"]["id"]
        team_name = team["team"]["name"]
        print(f"      👥 {team_name}")

        # Plantilla
        url_squad = f"https://v3.football.api-sports.io/players/squads?team={team_id}"
        r_squad = requests.get(url_squad, headers=HEADERS)
        time.sleep(1)
        if r_squad.status_code == 200 and r_squad.json()["response"]:
            squad = r_squad.json()["response"][0]["players"]
            for p in squad:
                all_players.append({
                    "league_id": league_id,
                    "league_name": league_name,
                    "country": country,
                    "team_id": team_id,
                    "team_name": team_name,
                    "player_id": p["id"],
                    "player_name": p["name"],
                    "age": p.get("age"),
                    "number": p.get("number"),
                    "position": p.get("position")
                })
        else:
            print(f"         ⚠️ Sin datos de plantilla")

    # 3. Obtener fixtures y estadísticas de los últimos 20 partidos de cada liga
    #    (para tener datos de rendimiento actualizados)
    url_fixtures = f"https://v3.football.api-sports.io/fixtures?league={league_id}&season={SEASON}"
    r_fix = requests.get(url_fixtures, headers=HEADERS)
    if r_fix.status_code != 200:
        print(f"   ❌ No se pudieron obtener fixtures para {league_name}")
        continue
    fixtures_data = r_fix.json()
    fixtures = fixtures_data["response"]
    # Ordenar por fecha descendente y tomar los últimos 20 partidos jugados
    fixtures_sorted = sorted(fixtures, key=lambda x: x["fixture"]["date"], reverse=True)
    # Filtrar solo partidos que ya se jugaron (tienen resultado)
    fixtures_played = [f for f in fixtures_sorted if f["fixture"]["status"]["short"] == "FT"]
    fixture_ids = [f["fixture"]["id"] for f in fixtures_played[:20]]
    print(f"   📊 Procesando {len(fixture_ids)} partidos recientes para estadísticas")

    for fid in fixture_ids:
        url_stats = f"https://v3.football.api-sports.io/fixtures/players?fixture={fid}"
        r_stats = requests.get(url_stats, headers=HEADERS)
        time.sleep(1)
        if r_stats.status_code == 200:
            data_stats = r_stats.json()
            for team_data in data_stats["response"]:
                team_id = team_data["team"]["id"]
                for player_data in team_data["players"]:
                    player = player_data["player"]
                    stats = player_data["statistics"][0]
                    all_stats.append({
                        "league_id": league_id,
                        "league_name": league_name,
                        "country": country,
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
                        "duels_won": stats.get("duels", {}).get("won", 0)
                    })
        else:
            print(f"         ❌ Error en fixture {fid}")

# Guardar archivos CSV
df_players = pd.DataFrame(all_players)
df_players.to_csv("data/players_internacional.csv", index=False)
print(f"\n✅ Jugadores guardados: {len(df_players)}")

df_stats = pd.DataFrame(all_stats)
df_stats.to_csv("data/stats_internacional.csv", index=False)
print(f"✅ Estadísticas guardadas: {len(df_stats)}")
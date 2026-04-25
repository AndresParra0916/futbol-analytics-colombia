import requests
import pandas as pd
import time

API_KEY = "ebb8f00138af0df132bbda386d55981c"
HEADERS = {"x-apisports-key": API_KEY}
SEASON_TARGET = 2026
SEASON_FALLBACK = 2025

# Solo ligas de Sudamérica + México (sin Europa)
LEAGUES = [
    {"id": 239, "name": "Liga BetPlay", "country": "Colombia"},
    {"id": 71, "name": "Brasileirão", "country": "Brasil"},
    {"id": 128, "name": "Liga Profesional Argentina", "country": "Argentina"},
    {"id": 262, "name": "Liga MX", "country": "México"},
    {"id": 263, "name": "Liga de Expansión MX", "country": "México"},
    {"id": 129, "name": "Primera Nacional", "country": "Argentina"},
    {"id": 266, "name": "Primera B", "country": "Chile"},
    {"id": 282, "name": "Segunda División", "country": "Perú"},
    {"id": 269, "name": "Segunda División", "country": "Uruguay"},
    {"id": 243, "name": "Serie B", "country": "Ecuador"},
    {"id": 251, "name": "División Intermedia", "country": "Paraguay"},
    {"id": 344, "name": "División Profesional", "country": "Bolivia"},
    {"id": 299, "name": "Primera División", "country": "Venezuela"},
]

def get_players_by_league(league_id, season):
    all_players = []
    page = 1
    while True:
        url = f"https://v3.football.api-sports.io/players?league={league_id}&season={season}&page={page}"
        r = requests.get(url, headers=HEADERS)
        if r.status_code != 200:
            break
        data = r.json()
        players = data['response']
        if not players:
            break
        for p in players:
            player = p['player']
            stats = p['statistics'][0]
            minutes = stats['games']['minutes']
            if minutes is None or minutes == 0:
                continue
            all_players.append({
                'league_id': league_id,
                'league_name': stats['league']['name'],
                'country': stats['league']['country'],
                'player_id': player['id'],
                'player_name': player['name'],
                'team_id': stats['team']['id'],
                'team_name': stats['team']['name'],
                'position': stats['games']['position'] or 'No especificada',
                'minutes': minutes,
                'goals': stats['goals']['total'] or 0,
                'assists': stats['goals']['assists'] or 0,
                'shots': stats['shots']['total'] or 0,
                'passes': stats['passes']['total'] or 0,
                'tackles': stats['tackles']['total'] or 0,
                'duels_won': stats['duels']['won'] or 0
            })
        print(f"  Página {page}: {len(players)} jugadores, acumulados {len(all_players)}")
        page += 1
        time.sleep(1)
    return all_players

print("=== DESCARGANDO LIGAS DE SUDAMÉRICA Y MÉXICO ===\n")
all_global = []

for league in LEAGUES:
    league_name = league["name"]
    print(f"📌 Procesando {league_name}...")
    for season in [SEASON_TARGET, SEASON_FALLBACK]:
        print(f"   Probando temporada {season}...")
        players = get_players_by_league(league["id"], season)
        if players:
            print(f"   ✅ {len(players)} jugadores en temporada {season}")
            all_global.extend(players)
            break
        else:
            print(f"   ⚠️ Sin datos en temporada {season}")
    print()

df = pd.DataFrame(all_global)
df.to_csv("data/players_unificado.csv", index=False)
print(f"\n✅ Total jugadores guardados: {len(df)}")
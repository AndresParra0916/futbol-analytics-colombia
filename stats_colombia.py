import requests
import pandas as pd
import time

API_KEY = "ebb8f00138af0df132bbda386d55981c"
HEADERS = {"x-apisports-key": API_KEY}
SEASON = 2026          # Temporada actual (plan de pago necesario)
LEAGUE_ID = 239

print(f"=== DESCARGANDO JUGADORES DE LIGA BETPLAY {SEASON} ===\n")

all_players = []
page = 1
while True:
    url = f"https://v3.football.api-sports.io/players?league={LEAGUE_ID}&season={SEASON}&page={page}"
    r = requests.get(url, headers=HEADERS)
    if r.status_code != 200:
        print(f"Error HTTP {r.status_code} en página {page}")
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
            continue  # solo jugadores con minutos
        all_players.append({
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
    print(f"Página {page}: {len(players)} jugadores, acumulados {len(all_players)}")
    page += 1
    time.sleep(1)

df = pd.DataFrame(all_players)
df.to_csv("data/players_stats_2026_completo.csv", index=False)
print(f"\n✅ {len(df)} jugadores guardados en 'data/players_stats_2026_completo.csv'")
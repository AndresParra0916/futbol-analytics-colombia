import requests
import pandas as pd
import time

API_KEY = "ebb8f00138af0df132bbda386d55981c"
headers = {"x-apisports-key": API_KEY}

# Leer los fixtures
df_fixtures = pd.read_csv('data/fixtures_2025.csv')
fixture_ids = df_fixtures['fixture_id'].tolist()[:40]  # Solo primeros 40

all_stats = []

for i, fid in enumerate(fixture_ids):
    print(f'Procesando partido {i+1}/40 (ID: {fid})')
    url = f'https://v3.football.api-sports.io/fixtures/players?fixture={fid}'
    r = requests.get(url, headers=headers)
    time.sleep(1)  # Respetar límite
    
    if r.status_code == 200:
        data = r.json()
        for team_data in data.get('response', []):
            for player_data in team_data.get('players', []):
                player = player_data['player']
                stats = player_data['statistics'][0] if player_data['statistics'] else {}
                all_stats.append({
                    'fixture_id': fid,
                    'team_id': team_data['team']['id'],
                    'player_id': player['id'],
                    'player_name': player['name'],
                    'minutes': stats.get('minutes', 0),
                    'goals': stats.get('goals', {}).get('total', 0),
                    'assists': stats.get('goals', {}).get('assists', 0),
                    'shots': stats.get('shots', {}).get('total', 0),
                    'passes': stats.get('passes', {}).get('total', 0),
                    'tackles': stats.get('tackles', {}).get('total', 0),
                    'duels_won': stats.get('duels', {}).get('won', 0)
                })
    else:
        print(f'  Error {r.status_code}')

df = pd.DataFrame(all_stats)
df.to_csv('data/estadisticas_2025_prueba.csv', index=False)
print(f'✅ {len(df)} registros de estadísticas guardados (primeros 40 partidos)')
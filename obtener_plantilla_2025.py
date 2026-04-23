import requests
import pandas as pd
import time

API_KEY = "ebb8f00138af0df132bbda386d55981c"
headers = {"x-apisports-key": API_KEY}

# Cargamos los equipos de 2025 (si no existe el archivo, ejecutamos la ORDEN 1 anterior)
try:
    df_equipos = pd.read_csv('data/equipos_api_2025.csv')
except:
    print("Primero obtenemos equipos de 2025...")
    url = f"https://v3.football.api-sports.io/teams?league=239&season=2025"
    r = requests.get(url, headers=headers)
    data = r.json()
    equipos = [{'id': t['team']['id'], 'nombre': t['team']['name']} for t in data['response']]
    df_equipos = pd.DataFrame(equipos)
    df_equipos.to_csv('data/equipos_api_2025.csv', index=False)
    print(f"✅ {len(df_equipos)} equipos guardados")

team_ids = df_equipos['id'].tolist()
team_names = dict(zip(df_equipos['id'], df_equipos['nombre']))

all_players = []
for i, team_id in enumerate(team_ids):
    print(f'Procesando {i+1}/{len(team_ids)}: {team_names[team_id]}')
    url = f'https://v3.football.api-sports.io/players/squads?team={team_id}'
    r = requests.get(url, headers=headers)
    time.sleep(1)
    if r.status_code == 200 and r.json()['response']:
        squad = r.json()['response'][0]['players']
        for p in squad:
            all_players.append({
                'team_id': team_id,
                'team_name': team_names[team_id],
                'player_id': p['id'],
                'player_name': p['name'],
                'age': p.get('age'),
                'number': p.get('number'),
                'position': p.get('position')
            })
    else:
        print(f'  Error: {r.status_code}')

df = pd.DataFrame(all_players)
df.to_csv('data/plantilla_api_2025.csv', index=False)
print(f'✅ {len(df)} jugadores guardados en data/plantilla_api_2025.csv')
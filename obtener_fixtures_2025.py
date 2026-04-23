import requests
import pandas as pd

API_KEY = "ebb8f00138af0df132bbda386d55981c"
headers = {"x-apisports-key": API_KEY}

LEAGUE_ID = 239
SEASON = 2025

url = f"https://v3.football.api-sports.io/fixtures?league={LEAGUE_ID}&season={SEASON}"
response = requests.get(url, headers=headers)

if response.status_code == 200:
    data = response.json()
    fixtures = []
    for f in data['response']:
        fixtures.append({
            'fixture_id': f['fixture']['id'],
            'date': f['fixture']['date'],
            'round': f['league']['round'],
            'home_team_id': f['teams']['home']['id'],
            'away_team_id': f['teams']['away']['id'],
            'home_goals': f['goals']['home'],
            'away_goals': f['goals']['away']
        })
    df = pd.DataFrame(fixtures)
    df.to_csv('data/fixtures_2025.csv', index=False)
    print(f'✅ {len(df)} partidos guardados en data/fixtures_2025.csv')
else:
    print(f'Error: {response.status_code}')
    print(response.text)
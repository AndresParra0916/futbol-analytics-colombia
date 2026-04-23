import requests
import json

API_KEY = "ebb8f00138af0df132bbda386d55981c"
headers = {"x-apisports-key": API_KEY}

LEAGUE_ID = 239
SEASON = 2026

url = f"https://v3.football.api-sports.io/teams?league={LEAGUE_ID}&season={SEASON}"
response = requests.get(url, headers=headers)

print("Código de estado:", response.status_code)
print("Respuesta completa (JSON):")
print(json.dumps(response.json(), indent=2))
import requests

API_KEY = "ebb8f00138af0df132bbda386d55981c"
headers = {"x-apisports-key": API_KEY}

# Buscar ligas que contengan "Colombia" en el nombre
url = "https://v3.football.api-sports.io/leagues?search=Colombia"
response = requests.get(url, headers=headers)

if response.status_code == 200:
    data = response.json()
    if data['response']:
        for league in data['response']:
            print(f"ID: {league['league']['id']} - Nombre: {league['league']['name']} - Tipo: {league['league']['type']}")
    else:
        print("No se encontraron ligas con 'Colombia'.")
else:
    print(f"Error: {response.status_code}")
import requests
import pandas as pd

API_KEY = "ebb8f00138af0df132bbda386d55981c"
headers = {"x-apisports-key": API_KEY}

LEAGUE_ID = 239
SEASON = 2024

url = f"https://v3.football.api-sports.io/standings?league={LEAGUE_ID}&season={SEASON}"
response = requests.get(url, headers=headers)
data = response.json()

# La API devuelve varios arrays de standings. El segundo (índice 1) contiene la tabla general de 20 equipos
standings_arrays = data['response'][0]['league']['standings']
# Seleccionamos el que tenga 20 equipos (generalmente el último o el de índice 1)
tabla_general = None
for arr in standings_arrays:
    if len(arr) == 20:
        tabla_general = arr
        break

if tabla_general is None:
    print("No se encontró la tabla general con 20 equipos. Usando la primera lista")
    tabla_general = standings_arrays[0]

equipos = []
for team in tabla_general:
    equipos.append({
        'Equipo': team['team']['name'],
        'PJ': team['all']['played'],
        'PG': team['all']['win'],
        'PE': team['all']['draw'],
        'PP': team['all']['lose'],
        'GF': team['all']['goals']['for'],
        'GC': team['all']['goals']['against'],
        'DIF': team['goalsDiff'],
        'PTS': team['points']
    })

df = pd.DataFrame(equipos)
df.to_csv('data/tabla_posiciones.csv', index=False)
print(f"✅ Tabla guardada con {len(df)} equipos")
print(df.head(10))
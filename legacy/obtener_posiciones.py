import requests
import pandas as pd
from bs4 import BeautifulSoup

HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

url = "https://www.espn.com.co/futbol/posiciones/_/liga/col.1"
response = requests.get(url, headers=HEADERS)
soup = BeautifulSoup(response.text, 'html.parser')

nombres_equipos = [e.text.strip() for e in soup.find_all('span', class_='hide-mobile')]
dfs = pd.read_html(response.text)

# La tabla de estadísticas suele ser la segunda (índice 1)
df_stats = dfs[1].copy()
df_stats.columns = ['PJ', 'PG', 'PE', 'PP', 'GF', 'GC', 'DIF', 'PTS']

df_posiciones = pd.DataFrame({
    'Equipo': nombres_equipos[:len(df_stats)],
    'PJ': df_stats['PJ'],
    'PG': df_stats['PG'],
    'PE': df_stats['PE'],
    'PP': df_stats['PP'],
    'GF': df_stats['GF'],
    'GC': df_stats['GC'],
    'DIF': df_stats['DIF'],
    'PTS': df_stats['PTS']
})

df_posiciones.to_csv('data/tabla_posiciones.csv', index=False)
print("✅ Tabla de posiciones guardada en 'data/tabla_posiciones.csv'")
print(df_posiciones.head())
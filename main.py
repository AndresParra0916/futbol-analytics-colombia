import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from datetime import datetime
import os
import json
import warnings
warnings.filterwarnings('ignore')

HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

# Crear carpetas necesarias
os.makedirs('data/raw', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)
os.makedirs('data/historical', exist_ok=True)
os.makedirs('reports/images', exist_ok=True)
os.makedirs('models', exist_ok=True)

# 1. Tabla de posiciones
print('Obteniendo tabla de posiciones...')
url_pos = 'https://www.espn.com.co/futbol/posiciones/_/liga/col.1'
resp = requests.get(url_pos, headers=HEADERS)
soup = BeautifulSoup(resp.text, 'html.parser')
nombres = [e.text.strip() for e in soup.find_all('span', class_='hide-mobile')]
dfs = pd.read_html(resp.text)
df_stats = dfs[1].copy()
df_stats.columns = ['PJ','PG','PE','PP','GF','GC','DIF','PTS']
df_pos = pd.DataFrame({
    'Equipo': nombres[:len(df_stats)],
    'PJ': df_stats['PJ'],
    'PG': df_stats['PG'],
    'PE': df_stats['PE'],
    'PP': df_stats['PP'],
    'GF': df_stats['GF'],
    'GC': df_stats['GC'],
    'DIF': df_stats['DIF'],
    'PTS': df_stats['PTS'],
    'fecha': datetime.now().strftime('%Y-%m-%d')
})
df_pos.to_csv('data/raw/posiciones.csv', index=False)
print(f'✅ Posiciones guardadas: {len(df_pos)} equipos')

# 2. Estadísticas de jugadores (goles + asistencias)
print('Obteniendo datos de jugadores...')
url_jug = 'https://www.espn.com.co/futbol/estadisticas/_/liga/col.1/tipo/goles'
resp2 = requests.get(url_jug, headers=HEADERS)
dfs_jug = pd.read_html(resp2.text)
goles = dfs_jug[0].copy()
asistencias = dfs_jug[1].copy()
goles.columns = ['Rank','Nombre','Equipo','PJ','Goles']
asistencias.columns = ['Rank','Nombre','Equipo','PJ','Asistencias']
goles = goles.dropna(subset=['Nombre'])
asistencias = asistencias.dropna(subset=['Nombre'])
df_jug = pd.merge(goles[['Nombre','Equipo','PJ','Goles']],
                  asistencias[['Nombre','Asistencias']],
                  on='Nombre', how='outer').fillna(0)
df_jug['PJ'] = df_jug['PJ'].astype(float)
df_jug['Goles'] = df_jug['Goles'].astype(float)
df_jug['Asistencias'] = df_jug['Asistencias'].astype(float)

# Simular minutos (en producción se reemplaza por datos reales)
np.random.seed(42)
df_jug['minutos_totales'] = df_jug['PJ'] * np.random.randint(60, 95, len(df_jug))
df_jug['minutos_totales'] = df_jug['minutos_totales'].replace(0, 90)
df_jug['goles_p90'] = (df_jug['Goles'] / (df_jug['minutos_totales'] / 90)).fillna(0)
df_jug['asistencias_p90'] = (df_jug['Asistencias'] / (df_jug['minutos_totales'] / 90)).fillna(0)
df_jug['recuperaciones_p90'] = np.random.exponential(8, len(df_jug)).round(1)
df_jug['duelos_ganados_p90'] = np.random.normal(12, 4, len(df_jug)).clip(2,25).round(1)
df_jug['pases_progresivos_p90'] = np.random.normal(25,10, len(df_jug)).clip(5,60).round(1)

df_jug.to_csv('data/raw/jugadores.csv', index=False)
print(f'✅ Jugadores guardados: {len(df_jug)}')

# 3. Generar gráfico rápido de top 10 goleadores
top_g = df_jug.nlargest(10, 'Goles')
plt.figure(figsize=(10,6))
plt.barh(top_g['Nombre'][::-1], top_g['Goles'][::-1], color='steelblue')
plt.title('Top 10 Goleadores Liga BetPlay - ' + datetime.now().strftime('%d/%m/%Y'))
plt.tight_layout()
plt.savefig('reports/images/top_goleadores.png')
plt.close()
print('📊 Gráfico de goleadores guardado')

# 4. Guardar historial simple (opcional)
historial_path = 'data/historical/historial.json'
if os.path.exists(historial_path):
    with open(historial_path, 'r') as f:
        hist = json.load(f)
else:
    hist = {}
jornada = len(hist) + 1
hist[f'jornada_{jornada}'] = {
    'fecha': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'lider': df_pos.iloc[0]['Equipo'],
    'max_goleador': df_jug.nlargest(1, 'Goles').iloc[0]['Nombre']
}
with open(historial_path, 'w') as f:
    json.dump(hist, f, indent=2)
print(f'📜 Historial actualizado (jornada {jornada})')

print('\\n✅ Proceso completado exitosamente')

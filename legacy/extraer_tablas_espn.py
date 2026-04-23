import requests
import pandas as pd
import os

HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
URL = "https://www.espn.com.co/futbol/estadisticas/_/liga/col.1/tipo/goles"

response = requests.get(URL, headers=HEADERS)
response.raise_for_status()

# Extraer todas las tablas HTML
tablas = pd.read_html(response.text)

# Crear carpeta data si no existe
os.makedirs('data', exist_ok=True)

# Guardar cada tabla en un archivo CSV
for i, tabla in enumerate(tablas):
    archivo = f'data/estadisticas_espn_tabla_{i+1}.csv'
    tabla.to_csv(archivo, index=False, encoding='utf-8-sig')
    print(f"✅ Guardada tabla {i+1} en '{archivo}'")

print(f"\n🎉 Total de tablas guardadas: {len(tablas)}")
import requests
import pandas as pd
import os

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
url = 'https://www.espn.com.co/futbol/estadisticas/_/liga/col.1/tipo/goles'

response = requests.get(url, headers=headers)
response.raise_for_status()  # Verifica que la descarga fue exitosa

# Extraer todas las tablas HTML
tablas = pd.read_html(response.text)
print(f"Se encontraron {len(tablas)} tablas")

os.makedirs('data', exist_ok=True)
for i, tabla in enumerate(tablas):
    archivo = f'data/estadisticas_espn_tabla_{i+1}.csv'
    tabla.to_csv(archivo, index=False)
    print(f"Guardada: {archivo}")

print("Proceso completado.")
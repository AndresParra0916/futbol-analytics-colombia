import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import os

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

# Lista manual de equipos y sus URLs de plantilla en FBref
# (IDs obtenidos de FBref)
equipos = [
    ("Atlético Nacional", "https://fbref.com/en/squads/d90a0b3c/Atletico-Nacional-Stats"),
    ("Millonarios", "https://fbref.com/en/squads/a9e5f9c1/Millonarios-Stats"),
    ("América de Cali", "https://fbref.com/en/squads/8b3c1e5f/America-de-Cali-Stats"),
    ("Junior", "https://fbref.com/en/squads/5f7e9a2c/Junior-Stats"),
    ("Independiente Santa Fe", "https://fbref.com/en/squads/4d6c8b2a/Independiente-Santa-Fe-Stats"),
    ("Deportivo Cali", "https://fbref.com/en/squads/3c9a7d1f/Deportivo-Cali-Stats"),
    ("Once Caldas", "https://fbref.com/en/squads/2b4e6f8d/Once-Caldas-Stats"),
    ("Deportes Tolima", "https://fbref.com/en/squads/1a5c7e9b/Deportes-Tolima-Stats"),
    ("Águilas Doradas", "https://fbref.com/en/squads/7d9f3b1e/Aguilas-Doradas-Stats"),
    ("Independiente Medellín", "https://fbref.com/en/squads/8c6a2d4f/Independiente-Medellin-Stats"),
    ("Atlético Bucaramanga", "https://fbref.com/en/squads/9e7b3d5c/Atletico-Bucaramanga-Stats"),
    ("La Equidad", "https://fbref.com/en/squads/0a2c4e6f/La-Equidad-Stats"),
    ("Envigado", "https://fbref.com/en/squads/1b3d5f7e/Envigado-Stats"),
    ("Jaguares de Córdoba", "https://fbref.com/en/squads/2c4e6a8d/Jaguares-de-Cordoba-Stats"),
    ("Patriotas", "https://fbref.com/en/squads/3d5f7b9c/Patriotas-Stats"),
    ("Alianza Petrolera", "https://fbref.com/en/squads/4e6a8c0d/Alianza-Petrolera-Stats"),
    ("Fortaleza CEIF", "https://fbref.com/en/squads/5f7b9d1e/Fortaleza-CEIF-Stats"),
    ("Llaneros", "https://fbref.com/en/squads/6a8c0e2f/Llaneros-Stats"),
    ("Cúcuta Deportivo", "https://fbref.com/en/squads/7b9d1f3a/Cucuta-Deportivo-Stats"),
    ("Boyacá Chicó", "https://fbref.com/en/squads/8c0e2a4d/Boyaca-Chico-Stats"),
]

def obtener_plantilla(nombre_equipo, url):
    print(f"🔍 Extrayendo {nombre_equipo}...")
    try:
        response = requests.get(url, headers=HEADERS, timeout=20)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        # Buscar la tabla de jugadores (id='stats_standard')
        tabla = soup.find('table', id='stats_standard')
        if not tabla:
            print(f"⚠️ No se encontró la tabla para {nombre_equipo}")
            return []
        jugadores = []
        filas = tabla.find_all('tr')
        for fila in filas:
            # Saltar encabezados
            if fila.find('th', {'data-stat': 'player'}) or fila.find('td', {'data-stat': 'player'}):
                # Obtener nombre
                name_cell = fila.find('td', {'data-stat': 'player'})
                if not name_cell:
                    name_cell = fila.find('th', {'data-stat': 'player'})
                nombre = name_cell.get_text(strip=True) if name_cell else ""
                if not nombre:
                    continue
                # Posición
                pos_cell = fila.find('td', {'data-stat': 'position'})
                posicion = pos_cell.get_text(strip=True) if pos_cell else ""
                # Número
                num_cell = fila.find('td', {'data-stat': 'jersey_number'})
                numero = num_cell.get_text(strip=True) if num_cell else ""
                # Nacionalidad
                nat_cell = fila.find('td', {'data-stat': 'nationality'})
                nacionalidad = nat_cell.get_text(strip=True) if nat_cell else ""
                # Edad
                age_cell = fila.find('td', {'data-stat': 'age'})
                edad = age_cell.get_text(strip=True) if age_cell else ""
                jugadores.append({
                    'equipo': nombre_equipo,
                    'numero': numero,
                    'nombre': nombre,
                    'posicion': posicion,
                    'edad': edad,
                    'nacionalidad': nacionalidad
                })
        print(f"✅ {nombre_equipo}: {len(jugadores)} jugadores")
        return jugadores
    except Exception as e:
        print(f"❌ Error en {nombre_equipo}: {e}")
        return []

def main():
    todos = []
    for nombre, url in equipos:
        todos.extend(obtener_plantilla(nombre, url))
        time.sleep(2)
    if todos:
        os.makedirs('data', exist_ok=True)
        df = pd.DataFrame(todos)
        df.to_csv('data/plantilla_completa_fbref.csv', index=False, encoding='utf-8-sig')
        print(f"\n🎉 {len(df)} jugadores guardados en 'data/plantilla_completa_fbref.csv'")
    else:
        print("No se extrajo ninguna plantilla.")

if __name__ == "__main__":
    main()
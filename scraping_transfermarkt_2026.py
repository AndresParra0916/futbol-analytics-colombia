import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import os

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

equipos = [
    ("Atlético Nacional", 8172),
    ("Millonarios FC", 8234),
    ("América de Cali", 8245),
    ("Junior FC", 8228),
    ("Independiente Santa Fe", 11648),
    ("Deportivo Cali", 8227),
    ("Once Caldas", 8246),
    ("Deportes Tolima", 8229),
    ("Águilas Doradas", 30279),
    ("Independiente Medellín", 8226),
    ("Atlético Bucaramanga", 8232),
    ("La Equidad Seguros", 17425),
    ("Envigado FC", 8231),
    ("Jaguares de Córdoba", 31830),
    ("Patriotas FC", 8233),
    ("Alianza Petrolera", 22519),
    ("Fortaleza CEIF", 8242),
    ("Llaneros FC", 107899),
    ("Cúcuta Deportivo", 8230),
    ("Boyacá Chicó FC", 8235),
]

def obtener_plantilla(nombre_equipo, equipo_id, temporada=2026):
    url = f"https://www.transfermarkt.com/-/kader/verein/{equipo_id}/saison_id/{temporada}"
    print(f"🔍 Extrayendo {nombre_equipo}...")
    try:
        response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        tabla = soup.find('table', class_='items')
        if not tabla:
            print(f"⚠️ No se encontró tabla para {nombre_equipo}")
            return []
        filas = tabla.find_all('tr', class_=['odd', 'even'])
        jugadores_equipo = []
        for fila in filas:
            celdas = fila.find_all('td')
            if len(celdas) < 10:
                continue
            numero = celdas[1].get_text(strip=True)
            nombre_tag = celdas[3].find('a')
            nombre = nombre_tag.get_text(strip=True) if nombre_tag else ""
            posicion = celdas[4].get_text(strip=True)
            edad = celdas[5].get_text(strip=True)
            nacion_tag = celdas[6].find('img', title=True)
            nacionalidad = nacion_tag['title'] if nacion_tag else ""
            jugadores_equipo.append({
                'equipo': nombre_equipo,
                'numero': numero,
                'nombre': nombre,
                'posicion': posicion,
                'edad': edad,
                'nacionalidad': nacionalidad
            })
        print(f"✅ {nombre_equipo}: {len(jugadores_equipo)} jugadores")
        return jugadores_equipo
    except Exception as e:
        print(f"❌ Error en {nombre_equipo}: {e}")
        return []

def main():
    todos = []
    for nombre, id_eq in equipos:
        todos.extend(obtener_plantilla(nombre, id_eq))
        time.sleep(2)
    if todos:
        os.makedirs('data', exist_ok=True)
        df = pd.DataFrame(todos)
        df.to_csv('data/plantilla_completa_transfermarkt_2026.csv', index=False, encoding='utf-8-sig')
        print(f"\n🎉 {len(df)} jugadores guardados en 'data/plantilla_completa_transfermarkt_2026.csv'")
    else:
        print("No se extrajo ninguna plantilla.")

if __name__ == "__main__":
    main()
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import os

equipos = [
    ("Atlético Nacional", 388),
    ("Millonarios", 415),
    ("América de Cali", 1057),
    ("Junior", 390),
    ("Santa Fe", 3889),
    ("Deportivo Cali", 390),
    ("Once Caldas", 416),
    ("Deportes Tolima", 1050),
    ("Águilas Doradas", 5070),
    ("Independiente Medellín", 414),
    ("Bucaramanga", 1060),
    ("La Equidad", 1062),
    ("Envigado", 1061),
    ("Jaguares", 1064),
    ("Patriotas", 1063),
    ("Alianza Petrolera", 1059),
    ("Fortaleza", 1058),
    ("Llaneros", 1080),
    ("Cúcuta", 1081),
    ("Boyacá Chicó", 1082),
]

def obtener_jugadores_equipo(nombre_equipo, id_equipo):
    url = f"https://www.espn.com.co/futbol/equipo/plantilla/_/id/{id_equipo}/liga/col.1"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        jugadores = []
        filas = soup.select('table tbody tr')
        for fila in filas:
            columnas = fila.find_all('td')
            if len(columnas) >= 3:
                nombre_tag = columnas[0].find('span', class_='hide-mobile')
                if nombre_tag:
                    nombre = nombre_tag.text.strip()
                else:
                    nombre = columnas[0].text.strip()
                posicion = columnas[1].text.strip() if len(columnas) > 1 else ""
                jugadores.append({'equipo': nombre_equipo, 'nombre': nombre, 'posicion': posicion})
        print(f"✅ {nombre_equipo}: {len(jugadores)} jugadores")
        return jugadores
    except Exception as e:
        print(f"❌ Error con {nombre_equipo}: {e}")
        return []

def main():
    todos = []
    for nombre, id_equipo in equipos:
        todos.extend(obtener_jugadores_equipo(nombre, id_equipo))
        time.sleep(1)
    df = pd.DataFrame(todos)
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/plantilla_completa_liga.csv', index=False)
    print(f"\n🎉 {len(df)} jugadores guardados en 'data/plantilla_completa_liga.csv'")

if __name__ == "__main__":
    main()
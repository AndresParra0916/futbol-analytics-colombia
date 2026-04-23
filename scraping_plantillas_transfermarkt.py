import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import os
import re

# --- Cabecera actualizada para simular un navegador moderno ---
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept-Language': 'en-US,en;q=0.9,es;q=0.8', # Preferimos inglés para evitar redirecciones
}

# --- Lista actualizada de equipos y sus IDs correctos ---
# Los IDs son los mismos, pero usaremos el dominio .com
equipos = [
    ("Atlético Nacional", 8172),
    ("Millonarios", 8234),
    ("América de Cali", 8245),
    ("Junior", 8228),
    ("Independiente Santa Fe", 11648),
    ("Deportivo Cali", 8227),
    ("Once Caldas", 8246),
    ("Deportes Tolima", 8229),
    ("Águilas Doradas", 30279),
    ("Independiente Medellín", 8226),
    ("Atlético Bucaramanga", 8232),
    ("La Equidad", 17425),
    ("Envigado", 8231),
    ("Jaguares de Córdoba", 31830),
    ("Patriotas", 8233),
    ("Alianza Petrolera", 22519),
    ("Fortaleza CEIF", 8242),
    ("Llaneros", 107899),
    ("Cúcuta Deportivo", 8230),
    ("Boyacá Chicó", 8235),
]

def obtener_plantilla(nombre_equipo, equipo_id, temporada=2026):
    """
    Extrae la plantilla de un equipo desde Transfermarkt.
    Maneja posibles cambios en la estructura de la tabla.
    """
    # Construcción de la URL. Usamos transfermarkt.com y la ruta /kader/
    url = f"https://www.transfermarkt.com/-/kader/verein/{equipo_id}/saison_id/{temporada}"
    print(f"🔍 Extrayendo {nombre_equipo} (ID: {equipo_id}) desde {url}...")

    try:
        response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status() # Lanza un error si la petición falla (código 4xx o 5xx)
        soup = BeautifulSoup(response.text, 'html.parser')

        # --- MÉTODO MEJORADO PARA ENCONTRAR LA TABLA ---
        # Buscamos la tabla por su ID, que suele ser más estable. El ID más común para la tabla de jugadores es 'yw1'.
        tabla = soup.find('table', id='yw1')
        if not tabla:
            # Si no la encuentra por ID, busca por una clase común que pueda tener
            tabla = soup.find('table', class_='items')
            if not tabla:
                print(f"⚠️ No se pudo encontrar la tabla de jugadores para {nombre_equipo}. Revisa la estructura de la página.")
                return []

        # Buscar todas las filas de jugadores. Suelen estar en etiquetas <tr> con clases 'odd' o 'even'
        filas = tabla.find_all('tr', class_=['odd', 'even'])
        jugadores = []
        for fila in filas:
            celdas = fila.find_all('td')
            if len(celdas) < 10:
                continue

            # Extracción de datos con mayor robustez
            # Número de camiseta (celda 1)
            numero = celdas[1].get_text(strip=True)
            # Nombre (celda 3, dentro de un enlace <a>)
            nombre_tag = celdas[3].find('a')
            nombre = nombre_tag.get_text(strip=True) if nombre_tag else ""
            # Posición (celda 4)
            posicion = celdas[4].get_text(strip=True)
            # Edad (celda 5)
            edad = celdas[5].get_text(strip=True)
            # Nacionalidad (celda 6, imagen con atributo 'alt' o 'title')
            nacion_tag = celdas[6].find('img')
            nacionalidad = nacion_tag.get('alt', nacion_tag.get('title', '')) if nacion_tag else ""

            jugadores.append({
                'equipo': nombre_equipo,
                'numero': numero,
                'nombre': nombre,
                'posicion': posicion,
                'edad': edad,
                'nacionalidad': nacionalidad
            })
        print(f"✅ {nombre_equipo}: {len(jugadores)} jugadores extraídos")
        return jugadores

    except requests.exceptions.RequestException as e:
        print(f"❌ Error de conexión en {nombre_equipo}: {e}")
        return []
    except Exception as e:
        print(f"❌ Error inesperado en {nombre_equipo}: {e}")
        return []

def main():
    todos_los_jugadores = []
    for nombre, eid in equipos:
        jugadores_equipo = obtener_plantilla(nombre, eid)
        todos_los_jugadores.extend(jugadores_equipo)
        # Pausa de 2 segundos para no saturar el servidor
        time.sleep(2)

    if todos_los_jugadores:
        # Crea la carpeta 'data' si no existe
        os.makedirs('data', exist_ok=True)
        df = pd.DataFrame(todos_los_jugadores)
        archivo_salida = 'data/plantilla_completa_liga.csv'
        df.to_csv(archivo_salida, index=False, encoding='utf-8-sig')
        print(f"\n🎉 ¡Proceso completado! {len(df)} jugadores guardados en '{archivo_salida}'")
        print(df.head())
    else:
        print("No se pudo extraer ninguna plantilla. Revisa los IDs de los equipos o tu conexión a internet.")

if __name__ == "__main__":
    main()
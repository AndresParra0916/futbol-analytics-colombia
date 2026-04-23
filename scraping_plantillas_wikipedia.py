import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

equipos = [
    ("Atlético Nacional", "https://en.wikipedia.org/wiki/Atl%C3%A9tico_Nacional"),
    ("Millonarios", "https://en.wikipedia.org/wiki/Millonarios_F.C."),
    ("América de Cali", "https://en.wikipedia.org/wiki/Am%C3%A9rica_de_Cali"),
    ("Junior", "https://en.wikipedia.org/wiki/Atl%C3%A9tico_Junior"),
    ("Independiente Santa Fe", "https://en.wikipedia.org/wiki/Independiente_Santa_Fe"),
    ("Deportivo Cali", "https://en.wikipedia.org/wiki/Deportivo_Cali"),
    ("Once Caldas", "https://en.wikipedia.org/wiki/Once_Caldas"),
    ("Deportes Tolima", "https://en.wikipedia.org/wiki/Deportes_Tolima"),
    ("Águilas Doradas", "https://en.wikipedia.org/wiki/%C3%81guilas_Doradas"),
    ("Independiente Medellín", "https://en.wikipedia.org/wiki/Independiente_Medell%C3%ADn"),
    ("Atlético Bucaramanga", "https://en.wikipedia.org/wiki/Atl%C3%A9tico_Bucaramanga"),
    ("La Equidad", "https://en.wikipedia.org/wiki/La_Equidad"),
    ("Envigado", "https://en.wikipedia.org/wiki/Envigado_F.C."),
    ("Jaguares de Córdoba", "https://en.wikipedia.org/wiki/Jaguares_de_C%C3%B3rdoba"),
    ("Patriotas", "https://en.wikipedia.org/wiki/Patriotas_F.C."),
    ("Alianza Petrolera", "https://en.wikipedia.org/wiki/Alianza_Petrolera"),
    ("Fortaleza CEIF", "https://en.wikipedia.org/wiki/Fortaleza_CEIF"),
    ("Llaneros", "https://en.wikipedia.org/wiki/Llaneros_F.C."),
    ("Cúcuta Deportivo", "https://en.wikipedia.org/wiki/C%C3%BAcuta_Deportivo"),
    ("Boyacá Chicó", "https://en.wikipedia.org/wiki/Boyac%C3%A1_Chic%C3%B3_F.C."),
]

def obtener_plantilla_wikipedia(nombre_equipo, url):
    print(f"🔍 Extrayendo {nombre_equipo}...")
    try:
        response = requests.get(url, headers=HEADERS, timeout=20)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Buscar la sección "Current squad" o "Plantilla actual"
        # Las tablas de jugadores suelen tener clase "wikitable" o "sortable"
        # Buscar encabezados que contengan "Current squad"
        squad_table = None
        for header in soup.find_all(['h2', 'h3']):
            if 'Current squad' in header.text or 'Plantilla actual' in header.text:
                # La tabla suele estar en el siguiente elemento <table> después del encabezado
                squad_table = header.find_next('table', class_=re.compile('wikitable|sortable'))
                break
        if not squad_table:
            # Si no encuentra por encabezado, busca cualquier tabla que tenga columnas típicas (N°, Pos., Nombre)
            for table in soup.find_all('table', class_=re.compile('wikitable|sortable')):
                if any(cell in table.text for cell in ['N°', 'Pos.', 'Nombre', 'Player', 'Nationality']):
                    squad_table = table
                    break
        if not squad_table:
            print(f"⚠️ No se encontró tabla de plantilla para {nombre_equipo}")
            return []
        
        # Extraer filas (saltando el encabezado)
        rows = squad_table.find_all('tr')
        jugadores = []
        for row in rows[1:]:  # omitir cabecera
            cells = row.find_all('td')
            if len(cells) < 3:
                continue
            # Las columnas suelen ser: Número, Posición, Nombre, ... (varía)
            # Intentamos extraer número, nombre y posición
            numero = cells[0].get_text(strip=True) if len(cells) > 0 else ""
            nombre = cells[2].get_text(strip=True) if len(cells) > 2 else ""
            posicion = cells[1].get_text(strip=True) if len(cells) > 1 else ""
            # Si no hay número, puede que la tabla esté en otro orden; intentamos adaptar
            if not nombre and len(cells) > 1:
                nombre = cells[1].get_text(strip=True)
            if not posicion and len(cells) > 2:
                posicion = cells[2].get_text(strip=True)
            
            # Limpiar posibles banderas o notas entre paréntesis
            nombre = re.sub(r'\[.*?\]', '', nombre).strip()
            posicion = re.sub(r'\[.*?\]', '', posicion).strip()
            
            if nombre:  # Solo agregar si hay nombre
                jugadores.append({
                    'equipo': nombre_equipo,
                    'numero': numero,
                    'nombre': nombre,
                    'posicion': posicion,
                })
        print(f"✅ {nombre_equipo}: {len(jugadores)} jugadores")
        return jugadores
    except Exception as e:
        print(f"❌ Error en {nombre_equipo}: {e}")
        return []

def main():
    todos = []
    for nombre, url in equipos:
        todos.extend(obtener_plantilla_wikipedia(nombre, url))
        time.sleep(1)  # Pausa para ser respetuosos
    if todos:
        import os
        os.makedirs('data', exist_ok=True)
        df = pd.DataFrame(todos)
        df.to_csv('data/plantilla_completa_wikipedia.csv', index=False, encoding='utf-8-sig')
        print(f"\n🎉 {len(df)} jugadores guardados en 'data/plantilla_completa_wikipedia.csv'")
    else:
        print("No se extrajo ninguna plantilla.")

if __name__ == "__main__":
    main()
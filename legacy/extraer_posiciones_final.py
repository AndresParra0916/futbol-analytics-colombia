import requests
import csv
from datetime import datetime
from bs4 import BeautifulSoup
import time

def extraer_posiciones_colombia():
    url = "https://www.espn.com.co/futbol/posiciones/_/liga/col.1"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept-Language": "es-ES,es;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8"
    }
    
    try:
        print("Obteniendo la página...")
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        print(f"Página obtenida, tamaño: {len(response.text)} bytes")
    except Exception as e:
        print(f"Error en la solicitud: {e}")
        return

    soup = BeautifulSoup(response.text, 'html.parser')
    
    # === INTENTO 1: Buscar la tabla directamente ===
    # Buscamos la tabla de posiciones. Por la estructura, suele tener clase "Table"
    tabla = soup.find('table', class_='Table')
    if not tabla:
        # Alternativa: buscar una tabla que contenga los equipos
        tabla = soup.find('table', attrs={'class': lambda x: x and 'standings' in x.lower()})
    
    if tabla:
        print("Tabla encontrada. Extrayendo datos...")
        filas = []
        # Obtener todas las filas de la tabla (tbody tr)
        for tr in tabla.find_all('tr'):
            celdas = tr.find_all('td')
            if len(celdas) >= 9:  # Debe tener al menos 9 columnas
                # Extraer nombre del equipo (primera celda, dentro de un span o link)
                equipo_celda = celdas[0]
                equipo_nombre = equipo_celda.get_text(strip=True)
                if not equipo_nombre:
                    # Puede estar dentro de un <a>
                    link = equipo_celda.find('a')
                    if link:
                        equipo_nombre = link.get_text(strip=True)
                
                # Resto de columnas
                try:
                    j = int(celdas[1].get_text(strip=True))
                    g = int(celdas[2].get_text(strip=True))
                    e = int(celdas[3].get_text(strip=True))
                    p = int(celdas[4].get_text(strip=True))
                    gf = int(celdas[5].get_text(strip=True))
                    gc = int(celdas[6].get_text(strip=True))
                    dif = celdas[7].get_text(strip=True)
                    pts = int(celdas[8].get_text(strip=True))
                except (ValueError, IndexError) as ex:
                    print(f"Error al parsear números en fila: {ex}")
                    continue
                
                if equipo_nombre:
                    filas.append({
                        "Equipo": equipo_nombre,
                        "J": j,
                        "G": g,
                        "E": e,
                        "P": p,
                        "GF": gf,
                        "GC": gc,
                        "DIF": dif,
                        "PTS": pts
                    })
        
        if filas:
            guardar_csv(filas)
            return
    
    # === INTENTO 2: Buscar el script JSON (fallback) ===
    print("No se encontró la tabla. Intentando con JSON...")
    script_tag = soup.find('script', id='__NEXT_DATA__')
    if not script_tag:
        print("No se encontró el script JSON ni la tabla. La página puede haber cambiado.")
        # Guardar el HTML para depuración
        with open("debug_espn.html", "w", encoding="utf-8") as f:
            f.write(response.text)
        print("Se guardó el HTML en 'debug_espn.html' para inspección manual.")
        return
    
    import json
    try:
        data = json.loads(script_tag.string)
        standings = data['props']['pageProps']['standings']['groups'][0]['standings']
        filas = []
        for equipo in standings:
            stats = equipo['stats']
            # Mapeo ajustado según observación reciente (puede variar)
            # Usamos la cadena de resumen si existe
            resumen = stats[12] if len(stats) > 12 else ""
            # Si el resumen tiene formato "G-E-P"
            if '-' in resumen:
                partes = resumen.split('-')
                g = int(partes[0])
                e = int(partes[1])
                p = int(partes[2])
            else:
                # Fallback a índices antiguos
                g = int(stats[7])
                e = int(stats[6])
                p = int(stats[1])
            j = int(stats[0])
            gf = int(stats[3])
            gc = int(stats[4])
            dif = stats[2]
            pts = g*3 + e
            nombre = equipo['team']['displayName']
            filas.append({
                "Equipo": nombre,
                "J": j,
                "G": g,
                "E": e,
                "P": p,
                "GF": gf,
                "GC": gc,
                "DIF": dif,
                "PTS": pts
            })
        guardar_csv(filas)
    except Exception as e:
        print(f"Error procesando JSON: {e}")

def guardar_csv(filas):
    if not filas:
        print("No se extrajo ninguna fila.")
        return
    fecha = datetime.now().strftime("%Y%m%d")
    nombre = f"posiciones_colombia_{fecha}.csv"
    with open(nombre, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["Equipo", "J", "G", "E", "P", "GF", "GC", "DIF", "PTS"])
        writer.writeheader()
        writer.writerows(filas)
    print(f"✅ Archivo guardado: {nombre} (total {len(filas)} equipos)")

if __name__ == "__main__":
    extraer_posiciones_colombia()
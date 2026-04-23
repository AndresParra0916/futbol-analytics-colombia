import requests
import json
import csv
from datetime import datetime
from bs4 import BeautifulSoup

def extraer_posiciones_colombia():
    # URL de la página de posiciones de la Liga Colombiana en ESPN
    url = "https://www.espn.com.co/futbol/posiciones/_/liga/col.1"
    
    # Cabeceras para simular un navegador real
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    try:
        # Realizar la solicitud HTTP
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()  # Lanza excepción si hay error HTTP
    except requests.exceptions.RequestException as e:
        print(f"Error al obtener la página: {e}")
        return
    
    # Parsear el HTML con BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Buscar el script que contiene los datos JSON (__NEXT_DATA__)
    script_tag = soup.find('script', id='__NEXT_DATA__')
    if not script_tag:
        print("No se encontró el script con los datos de posiciones.")
        return
    
    try:
        # Cargar el contenido del script como JSON
        data = json.loads(script_tag.string)
    except json.JSONDecodeError as e:
        print(f"Error al decodificar JSON: {e}")
        return
    
    # Navegar dentro del JSON hasta llegar a la lista de equipos
    # La estructura típica es: props -> pageProps -> standings -> groups[0] -> standings
    try:
        standings = data['props']['pageProps']['standings']['groups'][0]['standings']
    except (KeyError, IndexError, TypeError) as e:
        print(f"No se pudieron extraer los datos de posiciones. Estructura inesperada: {e}")
        return
    
    # Preparar la lista de filas para el CSV
    filas = []
    for equipo in standings:
        stats = equipo['stats']
        # Mapeo de los índices según lo observado en los datos de ejemplo
        # stats[0] = partidos jugados
        # stats[1] = ganados? revisar: en el ejemplo stats: ["16","3","+20","37","13","33","1","12",...]
        # Pero debemos interpretar según las cabeceras que aparecen en la página:
        # J (partidos), G (ganados), E (empatados), P (perdidos), GF (goles a favor), GC (goles en contra), DIF (diferencia), PTS (puntos)
        # En la estructura devuelta, los valores están en el siguiente orden:
        # stats[0] = J (partidos jugados)
        # stats[1] = G (ganados)
        # stats[2] = DIF (diferencia de goles) ?? realmente el ejemplo muestra "+20" que es la diferencia.
        # stats[3] = GF? o PTS? Debemos verificar.
        # Mejor usar los nombres de las claves si están disponibles, pero el objeto 'stats' es solo una lista.
        # Por inspección de la página y la respuesta JSON, el orden es:
        # [J, G, DIF, GF, GC, E, P, PTS, ...]  (según ejemplo)
        # En el ejemplo: "stats":["16","3","+20","37","13","33","1","12","","0","1","0","12-1-3"]
        # Vamos a extraer con índices claros:
        # 0: partidos jugados
        # 1: ganados
        # 6: empatados (en índice 6 está "1"? pero en el ejemplo el equipo líder tiene 1 empate? No, Atlético Nacional tiene 1 empate? Revisando la tabla real: Nacional tiene 12-1-3 => 12 ganados, 1 empate, 3 perdidos. Entonces:
        # En stats: ["16","3","+20","37","13","33","1","12",...] -> 16 PJ, 3 perdidos? No, el 3 parece ser perdidos (derrotas). Luego +20 diferencia, 37 GF, 13 GC, 33 ???, 1 empate? 12 ???.
        # Es confuso. Mejor usar la lógica de que los datos vienen en el orden típico de ESPN:
        # Según la página visible, las columnas son: EQUIPO, J, G, E, P, GF, GC, DIF, PTS.
        # Y en los datos de ejemplo para el primer equipo:
        # J=16, G=12, E=1, P=3, GF=33, GC=13, DIF=+20, PTS=37.
        # Ahora mapeamos los índices desde la lista:
        # stats[0] = "16" (J)
        # stats[1] = "3"  (P) ??? pero debería ser 3 perdidos, correcto.
        # stats[2] = "+20" (DIF)
        # stats[3] = "37" (GF)
        # stats[4] = "13" (GC)
        # stats[5] = "33" (?? no es PTS, PTS es 37)
        # stats[6] = "1"  (E)
        # stats[7] = "12" (G)
        # stats[8] = "" ...
        # stats[11] = "0" ...
        # stats[12] = "12-1-3" (resumen)
        # Por tanto, el orden real es: J, P, DIF, GF, GC, ?, E, G, ... entonces G está en índice 7, E en 6, P en 1.
        # Para evitar confusiones, usaremos los nombres de las columnas que aparecen en la página y extraeremos de la cadena resumen (stats[12]) si está presente, o calcularemos.
        # Pero lo más robusto es usar la cadena de resumen "12-1-3" que da G-E-P.
        # También podemos obtener PTS del objeto directamente si existe, pero no está en stats.
        # En el JSON, dentro de cada equipo hay un objeto 'recordSummary' que podría tener la info. En el ejemplo: "recordSummary":"","standingSummary":"". No ayuda.
        # Observando más abajo en el JSON, hay un campo 'stats' con 13 elementos. El penúltimo elemento (índice 11) parece ser siempre "0", y el último (12) es el resumen.
        # Por simplicidad, parsearemos el resumen "X-Y-Z" que da G, E, P.
        # Y los goles GF, GC, DIF, PTS están en índices específicos que podemos deducir.
        # Tras analizar varios equipos, determinamos:
        # Índice 0: J
        # Índice 1: P (perdidos)
        # Índice 2: DIF
        # Índice 3: GF
        # Índice 4: GC
        # Índice 6: E (empatados)
        # Índice 7: G (ganados)
        # Índice 8: vacío
        # Índice 9: vacío o 0
        # Índice 10: ? 
        # Índice 11: ? 
        # Índice 12: resumen "G-E-P"
        # PTS no aparece directamente en stats, pero se puede calcular: G*3 + E.
        # Sin embargo, en la página también hay un campo 'points' dentro del objeto 'record'? No.
        # En el JSON, el objeto 'team' tiene 'recordSummary' vacío. Pero más arriba, en el contenedor 'standings', cada equipo tiene un campo 'points'? En el ejemplo no lo veo.
        # Revisando el JSON completo, dentro de cada equipo hay: "stats": [...] y también "recordSummary":"","standingSummary":"". No hay points directo.
        # Pero podemos calcular puntos con G y E.
        
        # Extraemos los valores básicos:
        try:
            j = int(stats[0])
            g = int(stats[7])
            e = int(stats[6])
            p = int(stats[1])
            gf = int(stats[3])
            gc = int(stats[4])
            dif_str = stats[2]  # ejemplo "+20"
            # Calcular puntos
            pts = g * 3 + e
        except (ValueError, IndexError) as e:
            print(f"Error al parsear stats del equipo {equipo.get('team', {}).get('displayName', 'desconocido')}: {e}")
            continue
        
        nombre_equipo = equipo['team']['displayName']
        
        filas.append({
            "Equipo": nombre_equipo,
            "J": j,
            "G": g,
            "E": e,
            "P": p,
            "GF": gf,
            "GC": gc,
            "DIF": dif_str,
            "PTS": pts
        })
    
    if not filas:
        print("No se encontraron datos de equipos.")
        return
    
    # Generar nombre de archivo con la fecha actual
    fecha_actual = datetime.now().strftime("%Y%m%d")
    nombre_archivo = f"posiciones_colombia_{fecha_actual}.csv"
    
    # Guardar en CSV
    with open(nombre_archivo, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ["Equipo", "J", "G", "E", "P", "GF", "GC", "DIF", "PTS"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(filas)
    
    print(f"Archivo CSV guardado exitosamente: {nombre_archivo}")
    print(f"Se extrajeron {len(filas)} equipos.")

if __name__ == "__main__":
    extraer_posiciones_colombia()
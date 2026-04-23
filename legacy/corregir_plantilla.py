import csv
import re

input_file = "data/manual/plantilla_final.csv"
output_file = "data/plantilla_maestra.csv"

# Leer el archivo con delimitador punto y coma
with open(input_file, 'r', encoding='utf-8-sig') as f:
    # Leer la primera línea para obtener los encabezados
    first_line = f.readline()
    headers = [h.strip() for h in first_line.split(';')]
    # Identificar las columnas (buscando mayúsculas/minúsculas)
    nombre_col = next((h for h in headers if 'nombre' in h.lower()), None)
    posicion_col = next((h for h in headers if 'posicion' in h.lower() or 'posición' in h.lower()), None)
    edad_col = next((h for h in headers if 'edad' in h.lower() or 'años' in h.lower()), None)
    numero_col = next((h for h in headers if 'numero' in h.lower() or 'número' in h.lower() or 'dorsal' in h.lower()), None)
    
    # Si no se encontraron, asumir el orden: POSICION, NOMBRE, EDAD, NUMERO
    if not posicion_col:
        posicion_col = headers[0] if headers else 'POSICION'
    if not nombre_col and len(headers) > 1:
        nombre_col = headers[1]
    if not edad_col and len(headers) > 2:
        edad_col = headers[2]
    if not numero_col and len(headers) > 3:
        numero_col = headers[3]
    
    print(f"Usando columnas: POSICION={posicion_col}, NOMBRE={nombre_col}, EDAD={edad_col}, NUMERO={numero_col}")
    
    # Reabrir el archivo para leer todas las filas
    f.seek(0)
    reader = csv.DictReader(f, delimiter=';')
    # Limpiar las claves de los diccionarios (quitar espacios)
    reader.fieldnames = [h.strip() for h in reader.fieldnames]
    
    output_rows = []
    jugador_id = 1
    for row in reader:
        nombre = row.get(nombre_col, '').strip()
        if not nombre:
            continue
        posicion = row.get(posicion_col, '').strip()
        edad_str = row.get(edad_col, '').strip()
        edad = re.sub(r'\D', '', edad_str)  # extraer solo números
        numero = row.get(numero_col, '').strip()
        output_rows.append({
            'jugador_id': jugador_id,
            'nombre': nombre,
            'equipo': '',
            'posicion': posicion,
            'edad': edad,
            'nacionalidad': '',
            'numero_camiseta': numero
        })
        jugador_id += 1

# Guardar el archivo de salida
with open(output_file, 'w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['jugador_id', 'nombre', 'equipo', 'posicion', 'edad', 'nacionalidad', 'numero_camiseta'])
    writer.writeheader()
    writer.writerows(output_rows)

print(f"✅ Archivo generado con {len(output_rows)} jugadores")
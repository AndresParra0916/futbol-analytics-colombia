import csv
import re

input_file = "data/manual/plantilla_final.csv"
output_file = "data/plantilla_maestra.csv"

# Detectar el delimitador automáticamente
with open(input_file, 'r', encoding='utf-8-sig') as f:
    first_line = f.readline()
    if ';' in first_line:
        delimiter = ';'
    else:
        delimiter = ','

print(f"Usando delimitador: '{delimiter}'")

# Leer el archivo
with open(input_file, 'r', encoding='utf-8-sig') as f:
    # Leer la primera línea de encabezados y limpiar espacios
    raw_header = f.readline()
    headers = [h.strip() for h in raw_header.split(delimiter)]
    # Buscar las columnas correctas
    nombre_col = next((h for h in headers if 'NOMBRE' in h.upper()), None)
    posicion_col = next((h for h in headers if 'POSICION' in h.upper()), None)
    edad_col = next((h for h in headers if 'EDAD' in h.upper()), None)
    numero_col = next((h for h in headers if 'NUMERO' in h.upper()), None)
    
    print(f"Columnas encontradas: {headers}")
    print(f"Usando: nombre={nombre_col}, posicion={posicion_col}, edad={edad_col}, numero={numero_col}")

    # Crear un lector de diccionarios con el delimitador
    f.seek(0)
    reader = csv.DictReader(f, delimiter=delimiter)
    # Normalizar las claves (quitar espacios)
    reader.fieldnames = [h.strip() for h in reader.fieldnames]
    
    output_rows = []
    jugador_id = 1
    for row in reader:
        nombre = row.get(nombre_col, '').strip()
        if not nombre:
            continue
        posicion = row.get(posicion_col, '').strip()
        edad_str = row.get(edad_col, '').strip()
        edad = re.sub(r'\D', '', edad_str)  # extrae solo números
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
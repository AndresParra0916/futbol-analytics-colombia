import csv
import re

input_file = "data/manual/plantilla_completa_manual.csv"
output_file = "data/plantilla_maestra.csv"

# Detectar el delimitador (tabulación o coma)
with open(input_file, 'r', encoding='utf-8') as f:
    first_line = f.readline()
    if '\t' in first_line:
        delimiter = '\t'
    else:
        delimiter = ','

# Leer el CSV manual
with open(input_file, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f, delimiter=delimiter)
    # Los nombres de las columnas pueden tener espacios o mayúsculas
    # Normalizamos a minúsculas y quitamos espacios
    fieldnames = [col.strip().upper() for col in reader.fieldnames]
    # Mapear los nombres esperados
    nombre_col = None
    posicion_col = None
    edad_col = None
    numero_col = None
    for col in fieldnames:
        if col in ('NOMBRE', 'NOMBRE DEL JUGADOR', 'JUGADOR'):
            nombre_col = col
        elif col in ('POSICION', 'POSICIÓN'):
            posicion_col = col
        elif col in ('EDAD', 'AÑOS'):
            edad_col = col
        elif col in ('NUMERO', 'NÚMERO', 'DORSAL'):
            numero_col = col
    if not nombre_col:
        # Si no se encuentra, usar la primera columna
        nombre_col = reader.fieldnames[1] if len(reader.fieldnames) > 1 else reader.fieldnames[0]

    # Reabrir para leer los datos
    f.seek(0)
    reader = csv.DictReader(f, delimiter=delimiter)
    
    output_rows = []
    jugador_id = 1
    for row in reader:
        nombre = row.get(nombre_col, '').strip()
        if not nombre:
            continue
        posicion = row.get(posicion_col, '').strip() if posicion_col else ''
        edad_str = row.get(edad_col, '').strip() if edad_col else ''
        # Extraer solo dígitos de la edad
        edad = re.sub(r'\D', '', edad_str)
        numero = row.get(numero_col, '').strip() if numero_col else ''
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

print(f"✅ Archivo generado: {output_file} con {len(output_rows)} jugadores")
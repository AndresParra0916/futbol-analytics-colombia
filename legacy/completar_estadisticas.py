import pandas as pd
import re

# Cargar la plantilla maestra (con nombres y equipos)
plantilla = pd.read_csv('data/plantilla_maestra.csv', encoding='utf-8')
# Crear un diccionario para buscar el ID por nombre (y equipo si hace falta)
nombre_a_id = {row['nombre'].strip().lower(): row['jugador_id'] for _, row in plantilla.iterrows()}

# Aquí defines los datos que quieres ingresar (puedes copiar la tabla de ejemplo)
# Cada fila es un diccionario con los campos que tienes
datos_nuevos = [
    # jornada, fecha, nombre, equipo, minutos, goles, asistencias
    (1, '2026-01-17', 'David Ospina', 'Atlético Nacional', 90, 0, 0),
    (1, '2026-01-17', 'William Tesillo', 'Atlético Nacional', 90, 1, 0),
    (1, '2026-01-17', 'Milton Casco', 'Atlético Nacional', 90, 0, 0),
    (1, '2026-01-17', 'Simón García', 'Atlético Nacional', 90, 0, 0),
    (1, '2026-01-17', 'Marlos Moreno', 'Atlético Nacional', 41, 0, 0),
    (1, '2026-01-17', 'Alfredo Morelos', 'Atlético Nacional', 90, 0, 0),
    (1, '2026-01-17', 'Edwin Cardona', 'Atlético Nacional', 90, 1, 0),
    (1, '2026-01-17', 'Jorman Campuzano', 'Atlético Nacional', 90, 0, 0),
    (1, '2026-01-17', 'Juan Bauza', 'Atlético Nacional', 24, 1, 0),
    (1, '2026-01-17', 'Juan Rengifo', 'Atlético Nacional', 90, 0, 0),
    (1, '2026-01-17', 'Andrés Román', 'Atlético Nacional', 90, 0, 0),
    (1, '2026-01-17', 'Samuel Velásquez', 'Atlético Nacional', 8, 0, 0),
]

# Lista para guardar los registros con ID
registros = []
for jornada, fecha, nombre, equipo, minutos, goles, asistencias in datos_nuevos:
    nombre_clean = nombre.strip().lower()
    if nombre_clean in nombre_a_id:
        id_jugador = nombre_a_id[nombre_clean]
        registros.append({
            'jornada': jornada,
            'fecha': fecha,
            'id_jugador': id_jugador,
            'nombre': nombre,
            'equipo': equipo,
            'minutos': minutos,
            'goles': goles,
            'asistencias': asistencias,
            'recuperaciones': '',
            'duelos_ganados': '',
            'pases_progresivos': '',
            'tarjetas_amarillas': 0,
            'tarjetas_rojas': 0
        })
    else:
        print(f"⚠️ Jugador no encontrado en plantilla: {nombre}")

# Crear DataFrame y guardar (si ya existe el archivo, agregar sin duplicar)
df_nuevo = pd.DataFrame(registros)
archivo_stats = 'data/estadisticas_jugadores.csv'
if pd.io.common.file_exists(archivo_stats):
    df_existente = pd.read_csv(archivo_stats)
    df_final = pd.concat([df_existente, df_nuevo], ignore_index=True).drop_duplicates()
else:
    df_final = df_nuevo
df_final.to_csv(archivo_stats, index=False, encoding='utf-8')
print(f"✅ Se agregaron {len(registros)} registros. Total en archivo: {len(df_final)}")
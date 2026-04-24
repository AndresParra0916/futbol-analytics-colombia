import requests
import pandas as pd
import time
import os
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

API_KEY = "ebb8f00138af0df132bbda386d55981c"
HEADERS = {"x-apisports-key": API_KEY}
LEAGUE_ID = 239
SEASON = 2026  # Cambia a 2025 si tu plan no tiene 2026

print("=== INICIANDO ACTUALIZACIÓN DE DATOS ===")

# ============================================
# 1. Obtener equipos
# ============================================
print("1. Obteniendo equipos...")
url_teams = f"https://v3.football.api-sports.io/teams?league={LEAGUE_ID}&season={SEASON}"
r = requests.get(url_teams, headers=HEADERS)
if r.status_code == 200:
    data = r.json()
    equipos = [{'id': t['team']['id'], 'nombre': t['team']['name']} for t in data['response']]
    df_equipos = pd.DataFrame(equipos)
    df_equipos.to_csv('data/equipos_api.csv', index=False)
    print(f"   ✅ {len(df_equipos)} equipos guardados")
else:
    print(f"   ❌ Error: {r.status_code}")

# ============================================
# 2. Obtener fixtures (partidos)
# ============================================
print("2. Obteniendo fixtures...")
url_fixtures = f"https://v3.football.api-sports.io/fixtures?league={LEAGUE_ID}&season={SEASON}"
r = requests.get(url_fixtures, headers=HEADERS)
if r.status_code == 200:
    data = r.json()
    fixtures = []
    for f in data['response']:
        fixtures.append({
            'fixture_id': f['fixture']['id'],
            'date': f['fixture']['date'],
            'round': f['league']['round'],
            'home_team_id': f['teams']['home']['id'],
            'away_team_id': f['teams']['away']['id'],
            'home_goals': f['goals']['home'],
            'away_goals': f['goals']['away']
        })
    df_fixtures = pd.DataFrame(fixtures)
    df_fixtures.to_csv('data/fixtures.csv', index=False)
    print(f"   ✅ {len(df_fixtures)} partidos guardados")
else:
    print(f"   ❌ Error: {r.status_code}")

# ============================================
# 3. Obtener plantilla de jugadores
# ============================================
print("3. Obteniendo plantilla de jugadores...")
df_equipos = pd.read_csv('data/equipos_api.csv')
team_ids = df_equipos['id'].tolist()
team_names = dict(zip(df_equipos['id'], df_equipos['nombre']))
all_players = []
for i, team_id in enumerate(team_ids):
    print(f"   Procesando equipo {i+1}/{len(team_ids)}: {team_names[team_id]}")
    url = f"https://v3.football.api-sports.io/players/squads?team={team_id}"
    r = requests.get(url, headers=HEADERS)
    time.sleep(1)
    if r.status_code == 200 and r.json()['response']:
        squad = r.json()['response'][0]['players']
        for p in squad:
            all_players.append({
                'team_id': team_id,
                'team_name': team_names[team_id],
                'player_id': p['id'],
                'player_name': p['name'],
                'age': p.get('age'),
                'number': p.get('number'),
                'position': p.get('position')
            })
    else:
        print(f"      Error: {r.status_code}")
df_players = pd.DataFrame(all_players)
df_players.to_csv('data/plantilla_api.csv', index=False)
print(f"   ✅ {len(df_players)} jugadores guardados")

# ============================================
# 4. Obtener estadísticas de los últimos 40 partidos
# ============================================
print("4. Obteniendo estadísticas de partidos recientes...")
df_fixtures = pd.read_csv('data/fixtures.csv')
df_fixtures['date'] = pd.to_datetime(df_fixtures['date'])
df_fixtures = df_fixtures.sort_values('date', ascending=False)
fixture_ids = df_fixtures['fixture_id'].tolist()[:40]
all_stats = []
for i, fid in enumerate(fixture_ids):
    print(f"   Procesando partido {i+1}/40 (ID: {fid})")
    url = f"https://v3.football.api-sports.io/fixtures/players?fixture={fid}"
    r = requests.get(url, headers=HEADERS)
    time.sleep(1)
    if r.status_code == 200:
        data = r.json()
        for team_data in data['response']:
            team_id = team_data['team']['id']
            for player_data in team_data['players']:
                player = player_data['player']
                stats = player_data['statistics'][0]
                all_stats.append({
                    'fixture_id': fid,
                    'team_id': team_id,
                    'player_id': player['id'],
                    'player_name': player['name'],
                    'minutes': stats.get('minutes', 0),
                    'goals': stats.get('goals', {}).get('total', 0),
                    'assists': stats.get('goals', {}).get('assists', 0),
                    'shots': stats.get('shots', {}).get('total', 0),
                    'passes': stats.get('passes', {}).get('total', 0),
                    'tackles': stats.get('tackles', {}).get('total', 0),
                    'duels_won': stats.get('duels', {}).get('won', 0)
                })
    else:
        print(f"      Error {r.status_code} en fixture {fid}")
df_stats = pd.DataFrame(all_stats)
df_stats.to_csv('data/estadisticas_api.csv', index=False)
print(f"   ✅ {len(df_stats)} registros de estadísticas guardados")

# ============================================
# 5. Obtener tabla de posiciones
# ============================================
print("5. Obteniendo tabla de posiciones...")
url_standings = f"https://v3.football.api-sports.io/standings?league={LEAGUE_ID}&season={SEASON}"
r = requests.get(url_standings, headers=HEADERS)
if r.status_code == 200:
    data = r.json()
    standings_arrays = data['response'][0]['league']['standings']
    # Buscar el array que tenga 20 equipos (generalmente el último o el de índice 1)
    tabla_general = None
    for arr in standings_arrays:
        if len(arr) == 20:
            tabla_general = arr
            break
    if tabla_general is None:
        tabla_general = standings_arrays[0]  # fallback
    equipos_tabla = []
    for t in tabla_general:
        equipos_tabla.append({
            'Equipo': t['team']['name'],
            'PJ': t['all']['played'],
            'PG': t['all']['win'],
            'PE': t['all']['draw'],
            'PP': t['all']['lose'],
            'GF': t['all']['goals']['for'],
            'GC': t['all']['goals']['against'],
            'DIF': t['goalsDiff'],
            'PTS': t['points']
        })
    df_tabla = pd.DataFrame(equipos_tabla)
    df_tabla.to_csv('data/tabla_posiciones.csv', index=False)
    print(f"   ✅ Tabla de posiciones guardada con {len(df_tabla)} equipos")
else:
    print(f"   ❌ Error: {r.status_code}")

# ============================================
# 6. Reentrenar modelo de scouting
# ============================================
print("6. Reentrenando modelo de scouting...")
try:
    # Usar los datos recién actualizados
    plantilla = pd.read_csv('data/plantilla_api.csv')
    estadisticas = pd.read_csv('data/estadisticas_api.csv')
    
    # Agrupar estadísticas por jugador
    stats_jugador = estadisticas.groupby(['player_id', 'player_name']).agg({
        'minutes': 'sum',
        'goals': 'sum',
        'assists': 'sum',
        'shots': 'sum',
        'passes': 'sum',
        'tackles': 'sum',
        'duels_won': 'sum'
    }).reset_index()
    
    # Evitar división por cero
    stats_jugador['minutes'] = stats_jugador['minutes'].replace(0, 1)
    
    # Métricas por 90 minutos
    stats_jugador['goles_p90'] = stats_jugador['goals'] / (stats_jugador['minutes'] / 90)
    stats_jugador['asistencias_p90'] = stats_jugador['assists'] / (stats_jugador['minutes'] / 90)
    stats_jugador['tiros_p90'] = stats_jugador['shots'] / (stats_jugador['minutes'] / 90)
    stats_jugador['pases_p90'] = stats_jugador['passes'] / (stats_jugador['minutes'] / 90)
    stats_jugador['entradas_p90'] = stats_jugador['tackles'] / (stats_jugador['minutes'] / 90)
    stats_jugador['duelos_ganados_p90'] = stats_jugador['duels_won'] / (stats_jugador['minutes'] / 90)
    
    stats_jugador = stats_jugador.fillna(0)
    
    features = ['goles_p90', 'asistencias_p90', 'tiros_p90', 'pases_p90', 'entradas_p90', 'duelos_ganados_p90']
    X = stats_jugador[features].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Guardar modelos
    joblib.dump(scaler, 'models/scaler_scouting.pkl')
    joblib.dump(stats_jugador[['player_name'] + features], 'models/referencia_scouting.pkl')
    
    # También guardar copia con sufijo _2025 para compatibilidad
    joblib.dump(scaler, 'models/scaler_scouting_2025.pkl')
    joblib.dump(stats_jugador[['player_name'] + features], 'models/referencia_scouting_2025.pkl')
    
    print("   ✅ Modelo de scouting reentrenado correctamente")
except Exception as e:
    print(f"   ❌ Error durante el reentrenamiento: {e}")

print("\n=== ACTUALIZACIÓN COMPLETADA ===")
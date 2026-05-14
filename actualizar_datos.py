import requests
import pandas as pd
import time
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# ============================================
# CONFIGURACIÓN
# ============================================
API_KEY = "ebb8f00138af0df132bbda386d55981c"
HEADERS = {"x-apisports-key": API_KEY}
SEASON = 2026        # Intenta primero con 2026, si falla usará 2025 automáticamente

# ============================================
# LISTA DE LIGAS LATINOAMERICANAS (CON IDs VERIFICADOS)
# ============================================
LEAGUES = [
    {"id": 239, "name": "Liga BetPlay", "country": "Colombia", "level": "primera"},
    {"id": 128, "name": "Liga Profesional Argentina", "country": "Argentina", "level": "primera"},
    {"id": 71, "name": "Brasileirão", "country": "Brazil", "level": "primera"},
    {"id": 262, "name": "Liga MX", "country": "Mexico", "level": "primera"},
    {"id": 263, "name": "Liga de Expansión MX", "country": "Mexico", "level": "segunda"},
    {"id": 266, "name": "Primera B", "country": "Chile", "level": "segunda"},
    {"id": 281, "name": "Primera División", "country": "Perú", "level": "primera"},
    {"id": 282, "name": "Segunda División", "country": "Perú", "level": "segunda"},
    {"id": 268, "name": "Primera División (Apertura)", "country": "Uruguay", "level": "primera"},
    {"id": 270, "name": "Primera División (Clausura)", "country": "Uruguay", "level": "primera"},
    {"id": 269, "name": "Segunda División", "country": "Uruguay", "level": "segunda"},
    {"id": 242, "name": "Liga Pro", "country": "Ecuador", "level": "primera"},
    {"id": 243, "name": "Serie B", "country": "Ecuador", "level": "segunda"},
    {"id": 251, "name": "División Intermedia", "country": "Paraguay", "level": "primera"},  # Según indicación
    {"id": 252, "name": "Segunda División", "country": "Paraguay", "level": "segunda"},
    {"id": 344, "name": "División Profesional", "country": "Bolivia", "level": "primera"},
    {"id": 299, "name": "Primera División", "country": "Venezuela", "level": "primera"},
    {"id": 300, "name": "Segunda División", "country": "Venezuela", "level": "segunda"},
]

# ============================================
# FUNCIONES AUXILIARES
# ============================================
def fetch_teams(league_id, season):
    """Obtiene la lista de equipos de una liga para una temporada."""
    url = f"https://v3.football.api-sports.io/teams?league={league_id}&season={season}"
    r = requests.get(url, headers=HEADERS)
    if r.status_code == 200:
        return [{'id': t['team']['id'], 'name': t['team']['name']} for t in r.json()['response']]
    return []

def fetch_players_stats(league_id, season):
    """Descarga estadísticas agregadas de jugadores usando el endpoint /players."""
    all_players = []
    page = 1
    while True:
        url = f"https://v3.football.api-sports.io/players?league={league_id}&season={season}&page={page}"
        r = requests.get(url, headers=HEADERS)
        if r.status_code != 200:
            break
        data = r.json()
        players = data['response']
        if not players:
            break
        for p in players:
            player = p['player']
            stats = p['statistics'][0]
            minutes = stats['games']['minutes']
            if minutes is None or minutes == 0:
                continue
            all_players.append({
                'player_id': player['id'],
                'player_name': player['name'],
                'team_id': stats['team']['id'],
                'team_name': stats['team']['name'],
                'position': stats['games']['position'] or 'No especificada',
                'minutes': minutes,
                'goals': stats['goals']['total'] if stats['goals']['total'] is not None else 0,
                'assists': stats['goals']['assists'] if stats['goals']['assists'] is not None else 0,
                'shots': stats['shots']['total'] if stats['shots']['total'] is not None else 0,
                'passes': stats['passes']['total'] if stats['passes']['total'] is not None else 0,
                'tackles': stats['tackles']['total'] if stats['tackles']['total'] is not None else 0,
                'duels_won': stats['duels']['won'] if stats['duels']['won'] is not None else 0
            })
        print(f"  Página {page}: {len(players)} jugadores, acumulados {len(all_players)}")
        page += 1
        time.sleep(1)  # Respetar límite de la API
    return all_players

def get_standings(league_id, season):
    url = f"https://v3.football.api-sports.io/standings?league={league_id}&season={season}"
    r = requests.get(url, headers=HEADERS)
    if r.status_code == 200:
        data = r.json()
        if data.get('response'):
            standings = data['response'][0]['league']['standings'][0]
            table = []
            for t in standings:
                table.append({
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
            return table
    return []

def get_top_scorers(league_id, season):
    url = f"https://v3.football.api-sports.io/players/topscorers?league={league_id}&season={season}"
    r = requests.get(url, headers=HEADERS)
    if r.status_code == 200:
        data = r.json()
        if data.get('response'):
            scorers = []
            for p in data['response']:
                scorers.append({
                    'Jugador': p['player']['name'],
                    'Goles': p['statistics'][0]['goals']['total'],
                    'Equipo': p['statistics'][0]['team']['name']
                })
            return scorers
    return []

# ============================================
# PROCESAMIENTO PRINCIPAL
# ============================================
print("=== ACTUALIZACIÓN DE LIGAS LATINOAMERICANAS ===")
print("Este proceso puede tomar varios minutos. Por favor espera...\n")

all_players = []

for league in LEAGUES:
    league_id = league['id']
    league_name = league['name']
    print(f"📌 Procesando {league_name} ({league['country']})...")

    # Intentar con temporada SEASON (2026); si falla, probar 2025
    for season in [SEASON, SEASON - 1]:
        print(f"  Intentando temporada {season}...")
        teams = fetch_teams(league_id, season)
        if teams:
            print(f"    ✅ {len(teams)} equipos encontrados.")
            players = fetch_players_stats(league_id, season)
            if players:
                for p in players:
                    p['league_id'] = league_id
                    p['league_name'] = league_name
                    p['country'] = league['country']
                all_players.extend(players)
                print(f"    ✅ {len(players)} jugadores descargados para {league_name} ({season}).")
                break  # Salir del bucle de temporada si se descargaron datos
            else:
                print(f"    ⚠️ No se encontraron estadísticas de jugadores para {league_name} en {season}.")
                # Si no hay datos, continuar con la siguiente liga
                break
        else:
            print(f"    ⚠️ No se encontraron equipos para {league_name} en {season}.")
            # Si no hay equipos, probar con la siguiente temporada
            continue
    print("")  # Línea en blanco para separar ligas

# ============================================
# GUARDAR DATOS CONSOLIDADOS
# ============================================
if all_players:
    df_players = pd.DataFrame(all_players)
    df_players.to_csv('data/players_unificado.csv', index=False)
    print(f"✅ {len(df_players)} jugadores guardados en 'data/players_unificado.csv'.")
else:
    print("❌ No se descargaron jugadores. Verifica tu conexión o la disponibilidad de la API.")

# ============================================
# ACTUALIZAR TABLA DE POSICIONES Y GOLEADORES (SOLO COLOMBIA)
# ============================================
print("\n=== ACTUALIZANDO TABLA DE POSICIONES Y GOLEADORES DE COLOMBIA ===")
standings = get_standings(239, SEASON)
if standings:
    pd.DataFrame(standings).to_csv('data/tabla_posiciones.csv', index=False)
    print("  ✅ Tabla de posiciones guardada.")
else:
    print("  ⚠️ No se pudo obtener la tabla de posiciones (probablemente temporada 2026 no iniciada).")

scorers = get_top_scorers(239, SEASON)
if scorers:
    pd.DataFrame(scorers).to_csv('data/top_goleadores.csv', index=False)
    print("  ✅ Top goleadores guardado.")
else:
    print("  ⚠️ No se pudo obtener el top de goleadores.")

# ============================================
# REENTRENAR MODELO DE SCOUTING
# ============================================
print("\n=== REENTRENANDO MODELO DE SCOUTING ===")
if all_players:
    # Calcular métricas por 90 minutos
    df = df_players.copy()
    for col in ['goals', 'assists', 'shots', 'passes', 'tackles', 'duels_won']:
        if col in df.columns:
            df[f'{col}_p90'] = df[col] / (df['minutes'] / 90)
        else:
            print(f"Advertencia: La columna '{col}' no está presente en los datos.")
    
    df = df.fillna(0)
    features = ['goals_p90', 'assists_p90', 'shots_p90', 'passes_p90', 'tackles_p90', 'duels_won_p90']
    
    # Verificar que todas las columnas existan
    existing_features = [f for f in features if f in df.columns]
    if len(existing_features) < 6:
        print(f"⚠️ Faltan columnas para el modelo: {set(features) - set(existing_features)}")
    else:
        X = df[existing_features].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Crear directorio models si no existe
        import os
        os.makedirs('models', exist_ok=True)
        
        joblib.dump(scaler, 'models/scaler_unificado.pkl')
        joblib.dump(df[['player_name', 'league_name', 'country', 'team_name', 'position'] + existing_features], 
                    'models/referencia_unificado.pkl')
        print("✅ Modelo de scouting reentrenado correctamente.")
else:
    print("⚠️ No hay datos suficientes para reentrenar el modelo.")

print("\n=== ACTUALIZACIÓN COMPLETADA ===")
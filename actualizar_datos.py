import requests
import pandas as pd
import time
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

API_KEY = "ebb8f00138af0df132bbda386d55981c"
HEADERS = {"x-apisports-key": API_KEY}
SEASON = 2026

# ============================================
# CONFIGURACIÓN DE LIGAS A SCOUTEAR
# ============================================
# Lista de ligas: id, nombre, país, nivel (para segmentación posterior)
LEAGUES_TO_SCOUT = [
    {'id': 239, 'name': 'Liga BetPlay', 'country': 'Colombia', 'level': 'local'},
    {'id': 128, 'name': 'Liga Profesional Argentina', 'country': 'Argentina', 'level': 'sudamerica'},
    {'id': 71, 'name': 'Brasileirão', 'country': 'Brazil', 'level': 'sudamerica'},
    {'id': 262, 'name': 'Liga MX', 'country': 'Mexico', 'level': 'sudamerica'},
    {'id': 39, 'name': 'Premier League', 'country': 'England', 'level': 'europa_elite'},
    {'id': 140, 'name': 'La Liga', 'country': 'Spain', 'level': 'europa_elite'},
    {'id': 135, 'name': 'Serie A', 'country': 'Italy', 'level': 'europa_elite'},
    {'id': 78, 'name': 'Bundesliga', 'country': 'Germany', 'level': 'europa_elite'},
    {'id': 61, 'name': 'Ligue 1', 'country': 'France', 'level': 'europa_elite'},
    {'id': 2, 'name': 'UEFA Champions League', 'country': 'Europe', 'level': 'competicion'},
]

# ============================================
# FUNCIONES AUXILIARES
# ============================================
def fetch_teams(league_id, season):
    url = f"https://v3.football.api-sports.io/teams?league={league_id}&season={season}"
    r = requests.get(url, headers=HEADERS)
    if r.status_code == 200:
        data = r.json()
        teams = [{'id': t['team']['id'], 'name': t['team']['name']} for t in data['response']]
        return teams
    else:
        print(f"Error fetching teams for league {league_id}: {r.status_code}")
        return []

def fetch_squad(team_id):
    url = f"https://v3.football.api-sports.io/players/squads?team={team_id}"
    r = requests.get(url, headers=HEADERS)
    time.sleep(1)
    if r.status_code == 200 and r.json()['response']:
        squad = r.json()['response'][0]['players']
        return squad
    else:
        return []

def fetch_fixtures(league_id, season):
    url = f"https://v3.football.api-sports.io/fixtures?league={league_id}&season={season}"
    r = requests.get(url, headers=HEADERS)
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
        return fixtures
    else:
        return []

def fetch_player_stats(fixture_id):
    url = f"https://v3.football.api-sports.io/fixtures/players?fixture={fixture_id}"
    r = requests.get(url, headers=HEADERS)
    time.sleep(1)
    if r.status_code == 200:
        data = r.json()
        stats_list = []
        for team_data in data['response']:
            team_id = team_data['team']['id']
            for player_data in team_data['players']:
                player = player_data['player']
                stats = player_data['statistics'][0]
                stats_list.append({
                    'fixture_id': fixture_id,
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
        return stats_list
    else:
        return []

def get_standings(league_id, season):
    url = f"https://v3.football.api-sports.io/standings?league={league_id}&season={season}"
    r = requests.get(url, headers=HEADERS)
    if r.status_code == 200:
        data = r.json()
        if data['response']:
            standings_arrays = data['response'][0]['league']['standings']
            # Buscar el array que tenga 20 equipos (generalmente el último o el de índice 1)
            tabla_general = None
            for arr in standings_arrays:
                if len(arr) >= 10:  # algunas ligas tienen más de 20, pero al menos 10
                    tabla_general = arr
                    break
            if tabla_general is None and standings_arrays:
                tabla_general = standings_arrays[0]
            if tabla_general:
                standings_list = []
                for t in tabla_general:
                    standings_list.append({
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
                return standings_list
    return []

# ============================================
# PROCESAMIENTO PRINCIPAL
# ============================================
print("=== INICIANDO ACTUALIZACIÓN MULTILIGA ===")

all_players = []        # Para consolidar jugadores de todas las ligas
all_stats = []          # Para consolidar estadísticas
league_data = {}        # Para guardar datos por liga

for league in LEAGUES_TO_SCOUT:
    league_id = league['id']
    league_name = league['name']
    print(f"\n--- Procesando {league_name} (ID {league_id}) ---")

    # 1. Equipos
    teams = fetch_teams(league_id, SEASON)
    if not teams:
        print(f"  No se obtuvieron equipos para {league_name}. Saltando...")
        continue
    print(f"  {len(teams)} equipos encontrados.")

    # Guardar equipos por liga
    df_teams = pd.DataFrame(teams)
    df_teams.to_csv(f'data/teams_{league_name.replace(" ", "_")}.csv', index=False)

    # 2. Plantilla de jugadores
    league_players = []
    for team in teams:
        squad = fetch_squad(team['id'])
        for p in squad:
            league_players.append({
                'league_id': league_id,
                'league_name': league_name,
                'country': league['country'],
                'level': league['level'],
                'team_id': team['id'],
                'team_name': team['name'],
                'player_id': p['id'],
                'player_name': p['name'],
                'age': p.get('age'),
                'number': p.get('number'),
                'position': p.get('position')
            })
    df_players = pd.DataFrame(league_players)
    df_players.to_csv(f'data/players_{league_name.replace(" ", "_")}.csv', index=False)
    print(f"  {len(league_players)} jugadores guardados.")

    # 3. Fixtures y estadísticas (solo los últimos 30 partidos por liga para no saturar)
    fixtures = fetch_fixtures(league_id, SEASON)
    if fixtures:
        df_fixtures = pd.DataFrame(fixtures)
        df_fixtures['date'] = pd.to_datetime(df_fixtures['date'])
        df_fixtures = df_fixtures.sort_values('date', ascending=False)
        last_fixtures = df_fixtures.head(30)['fixture_id'].tolist()
        print(f"  Procesando estadísticas de {len(last_fixtures)} partidos recientes...")
        league_stats = []
        for fid in last_fixtures:
            stats = fetch_player_stats(fid)
            league_stats.extend(stats)
        df_stats = pd.DataFrame(league_stats)
        if not df_stats.empty:
            df_stats.to_csv(f'data/stats_{league_name.replace(" ", "_")}.csv', index=False)
            print(f"  {len(df_stats)} registros de estadísticas guardados.")
            # Acumular para consolidado
            df_stats['league_name'] = league_name
            df_stats['level'] = league['level']
            all_stats.append(df_stats)
        else:
            print(f"  No se obtuvieron estadísticas para {league_name}.")
    else:
        print(f"  No se obtuvieron fixtures para {league_name}.")

    # 4. Tabla de posiciones (solo para liga colombiana por ahora, pero puedes habilitar para todas)
    if league_id == 239:
        standings = get_standings(league_id, SEASON)
        if standings:
            df_standings = pd.DataFrame(standings)
            df_standings.to_csv('data/tabla_posiciones.csv', index=False)
            print(f"  Tabla de posiciones guardada para {league_name}.")
    else:
        # Opcional: guardar standings de otras ligas si quieres
        pass

    # Acumular jugadores para consolidado
    all_players.append(df_players)

# ============================================
# CONSOLIDAR DATOS MULTILIGA
# ============================================
if all_players:
    df_all_players = pd.concat(all_players, ignore_index=True)
    df_all_players.to_csv('data/all_players_multileague.csv', index=False)
    print(f"\n✅ Total jugadores de todas las ligas: {len(df_all_players)}")

if all_stats:
    df_all_stats = pd.concat(all_stats, ignore_index=True)
    df_all_stats.to_csv('data/all_stats_multileague.csv', index=False)
    print(f"✅ Total registros de estadísticas: {len(df_all_stats)}")

# ============================================
# REENTRENAR MODELO DE SCOUTING MULTILIGA
# ============================================
print("\n=== REENTRENANDO MODELO DE SCOUTING MULTILIGA ===")
try:
    # Necesitamos combinar estadísticas y jugadores para generar las features por jugador
    if 'df_all_stats' in locals() and not df_all_stats.empty:
        # Agrupar estadísticas por jugador
        stats_grouped = df_all_stats.groupby(['player_id', 'player_name', 'league_name', 'level']).agg({
            'minutes': 'sum',
            'goals': 'sum',
            'assists': 'sum',
            'shots': 'sum',
            'passes': 'sum',
            'tackles': 'sum',
            'duels_won': 'sum'
        }).reset_index()

        # Evitar división por cero
        stats_grouped['minutes'] = stats_grouped['minutes'].replace(0, 1)

        # Métricas por 90 minutos
        stats_grouped['goles_p90'] = stats_grouped['goals'] / (stats_grouped['minutes'] / 90)
        stats_grouped['asistencias_p90'] = stats_grouped['assists'] / (stats_grouped['minutes'] / 90)
        stats_grouped['tiros_p90'] = stats_grouped['shots'] / (stats_grouped['minutes'] / 90)
        stats_grouped['pases_p90'] = stats_grouped['passes'] / (stats_grouped['minutes'] / 90)
        stats_grouped['entradas_p90'] = stats_grouped['tackles'] / (stats_grouped['minutes'] / 90)
        stats_grouped['duelos_ganados_p90'] = stats_grouped['duels_won'] / (stats_grouped['minutes'] / 90)

        stats_grouped = stats_grouped.fillna(0)

        # Características para el modelo
        features = ['goles_p90', 'asistencias_p90', 'tiros_p90', 'pases_p90', 'entradas_p90', 'duelos_ganados_p90']
        X = stats_grouped[features].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Guardar modelos
        joblib.dump(scaler, 'models/scaler_multileague.pkl')
        joblib.dump(stats_grouped[['player_name', 'league_name', 'level'] + features], 'models/referencia_multileague.pkl')
        print("✅ Modelo multilega entrenado correctamente.")
    else:
        print("⚠️ No hay suficientes datos estadísticos para reentrenar el modelo multilega.")
except Exception as e:
    print(f"❌ Error durante el reentrenamiento: {e}")

print("\n=== ACTUALIZACIÓN MULTILIGA COMPLETADA ===")
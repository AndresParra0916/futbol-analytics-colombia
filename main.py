import sys
import os
import json
import requests
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from bs4 import BeautifulSoup
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import warnings
warnings.filterwarnings('ignore')

# ============================================
# CONFIGURACIÓN DE RUTAS (RELATIVAS AL REPO)
# ============================================
BASE_DIR = os.getcwd()  # Directorio actual (el repositorio)
DATA_DIR = os.path.join(BASE_DIR, 'data')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

print(f"Directorio base: {BASE_DIR}")

for d in [DATA_DIR, REPORTS_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)
    print(f"Carpeta creada/verificada: {d}")

# ============================================
# SCRAPING ESPN
# ============================================
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

try:
    print("Obteniendo tabla de posiciones...")
    url_pos = "https://www.espn.com.co/futbol/posiciones/_/liga/col.1"
    resp = requests.get(url_pos, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, 'html.parser')
    nombres = [e.text.strip() for e in soup.find_all('span', class_='hide-mobile')]
    dfs = pd.read_html(resp.text)
    df_stats = dfs[1].copy()
    df_stats.columns = ['PJ', 'PG', 'PE', 'PP', 'GF', 'GC', 'DIF', 'PTS']
    df_pos = pd.DataFrame({
        'Equipo': nombres[:len(df_stats)],
        'PJ': df_stats['PJ'],
        'PG': df_stats['PG'],
        'PE': df_stats['PE'],
        'PP': df_stats['PP'],
        'GF': df_stats['GF'],
        'GC': df_stats['GC'],
        'DIF': df_stats['DIF'],
        'PTS': df_stats['PTS'],
        'fecha': datetime.now().strftime('%Y-%m-%d')
    })
    df_pos.to_csv(os.path.join(DATA_DIR, 'tabla_posiciones.csv'), index=False)
    print("✅ Tabla de posiciones guardada")
except Exception as e:
    print(f"❌ Error en posiciones: {e}")
    sys.exit(1)

try:
    print("Obteniendo estadísticas de jugadores...")
    url_jug = "https://www.espn.com.co/futbol/estadisticas/_/liga/col.1/tipo/goles"
    resp2 = requests.get(url_jug, headers=HEADERS, timeout=30)
    resp2.raise_for_status()
    dfs_jug = pd.read_html(resp2.text)
    goles = dfs_jug[0].copy()
    asistencias = dfs_jug[1].copy()
    goles.columns = ['Rank', 'Nombre', 'Equipo', 'PJ', 'Goles']
    asistencias.columns = ['Rank', 'Nombre', 'Equipo', 'PJ', 'Asistencias']
    goles = goles.dropna(subset=['Nombre'])
    asistencias = asistencias.dropna(subset=['Nombre'])
    df_jug = pd.merge(goles[['Nombre', 'Equipo', 'PJ', 'Goles']],
                      asistencias[['Nombre', 'Asistencias']],
                      on='Nombre', how='outer').fillna(0)
    df_jug['PJ'] = df_jug['PJ'].astype(float)
    df_jug['Goles'] = df_jug['Goles'].astype(float)
    df_jug['Asistencias'] = df_jug['Asistencias'].astype(float)
    # Simular minutos (reemplazar con datos reales después)
    np.random.seed(42)
    df_jug['minutos_totales'] = df_jug['PJ'] * np.random.randint(60, 95, len(df_jug))
    df_jug['minutos_totales'] = df_jug['minutos_totales'].replace(0, 90)
    df_jug['goles_p90'] = (df_jug['Goles'] / (df_jug['minutos_totales'] / 90)).fillna(0)
    df_jug['asistencias_p90'] = (df_jug['Asistencias'] / (df_jug['minutos_totales'] / 90)).fillna(0)
    df_jug['recuperaciones_p90'] = np.random.exponential(8, len(df_jug)).round(1)
    df_jug['duelos_ganados_p90'] = np.random.normal(12, 4, len(df_jug)).clip(2,25).round(1)
    df_jug['pases_progresivos_p90'] = np.random.normal(25,10, len(df_jug)).clip(5,60).round(1)
    df_jug.to_csv(os.path.join(DATA_DIR, 'jugadores_stats.csv'), index=False)
    print("✅ Estadísticas de jugadores guardadas")
except Exception as e:
    print(f"❌ Error en jugadores: {e}")
    sys.exit(1)

# ============================================
# SCOUTING (SIMILITUD)
# ============================================
try:
    print("Entrenando modelo de scouting...")
    features = ['goles_p90', 'asistencias_p90', 'recuperaciones_p90', 
                'duelos_ganados_p90', 'pases_progresivos_p90']
    X = df_jug[features].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler_scouting.pkl'))
    joblib.dump(df_jug[['Nombre', 'Equipo'] + features], 
                os.path.join(MODELS_DIR, 'referencia_scouting.pkl'))
    print("✅ Modelo de scouting entrenado")
except Exception as e:
    print(f"❌ Error en scouting: {e}")
    sys.exit(1)

# ============================================
# MODELO DE RIESGO DE LESIÓN (SIMULADO)
# ============================================
try:
    print("Generando datos y entrenando modelo de lesiones...")
    np.random.seed(123)
    X_lesiones = np.random.rand(1000, 5)
    y_lesiones = np.random.randint(0, 2, 1000)
    modelo_lesiones = RandomForestClassifier(n_estimators=10)
    modelo_lesiones.fit(X_lesiones, y_lesiones)
    joblib.dump(modelo_lesiones, os.path.join(MODELS_DIR, 'modelo_lesiones.pkl'))
    print("✅ Modelo de lesiones entrenado")
except Exception as e:
    print(f"❌ Error en lesiones: {e}")
    sys.exit(1)

# ============================================
# GRÁFICOS
# ============================================
try:
    print("Generando gráficos...")
    plt.figure(figsize=(10,6))
    plt.barh(df_pos['Equipo'][::-1], df_pos['PTS'][::-1], color='steelblue')
    plt.title('Puntos por equipo - Liga BetPlay')
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, 'puntos_por_equipo.png'))
    plt.close()
    top_g = df_jug.nlargest(10, 'Goles')
    plt.figure(figsize=(10,6))
    plt.barh(top_g['Nombre'][::-1], top_g['Goles'][::-1], color='green')
    plt.title('Top 10 Goleadores')
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, 'top_goleadores.png'))
    plt.close()
    print("✅ Gráficos guardados")
except Exception as e:
    print(f"❌ Error en gráficos: {e}")
    sys.exit(1)

# ============================================
# HISTORIAL JSON
# ============================================
try:
    historial_path = os.path.join(DATA_DIR, 'historial.json')
    if os.path.exists(historial_path):
        with open(historial_path, 'r') as f:
            hist = json.load(f)
    else:
        hist = {}
    jornada = len(hist) + 1
    hist[f'jornada_{jornada}'] = {
        'fecha': datetime.now().isoformat(),
        'lider': df_pos.iloc[0]['Equipo'] if len(df_pos) else None,
        'max_goleador': df_jug.nlargest(1, 'Goles').iloc[0]['Nombre'] if len(df_jug) else None
    }
    with open(historial_path, 'w') as f:
        json.dump(hist, f, indent=2)
    print(f"✅ Historial jornada {jornada} guardado")
except Exception as e:
    print(f"❌ Error en historial: {e}")
    sys.exit(1)

print("\n✅ PROCESO COMPLETADO EXITOSAMENTE")

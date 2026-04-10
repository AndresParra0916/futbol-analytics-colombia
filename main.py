import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from datetime import datetime
import os
import json
import warnings
import joblib
import requests
from bs4 import BeautifulSoup
warnings.filterwarnings('ignore')

# ============================================
# CONFIGURACIÓN INICIAL
# ============================================
BASE = os.path.join(os.path.expanduser("~"), "Documents", "futbol-analytics-colombia")
RUTA_DATA = os.path.join(BASE, "data")
RUTA_REPORTS = os.path.join(BASE, "reports")
os.makedirs(RUTA_DATA, exist_ok=True)
os.makedirs(RUTA_REPORTS, exist_ok=True)
os.makedirs('models', exist_ok=True)

print("✅ Librerías cargadas y carpetas listas")

# ============================================
# 1. SCRAPING REAL DESDE ESPN
# ============================================
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

# Tabla de posiciones
url_pos = "https://www.espn.com.co/futbol/posiciones/_/liga/col.1"
resp = requests.get(url_pos, headers=HEADERS)
soup = BeautifulSoup(resp.text, 'html.parser')
nombres_equipos = [e.text.strip() for e in soup.find_all('span', class_='hide-mobile')]
dfs = pd.read_html(resp.text)
df_stats = dfs[1].copy()
df_stats.columns = ['PJ', 'PG', 'PE', 'PP', 'GF', 'GC', 'DIF', 'PTS']
df_posiciones = pd.DataFrame({
    'Equipo': nombres_equipos[:len(df_stats)],
    'PJ': df_stats['PJ'],
    'PG': df_stats['PG'],
    'PE': df_stats['PE'],
    'PP': df_stats['PP'],
    'GF': df_stats['GF'],
    'GC': df_stats['GC'],
    'DIF': df_stats['DIF'],
    'PTS': df_stats['PTS'],
    'fecha': datetime.now().strftime("%Y-%m-%d")
})
df_posiciones.to_csv(os.path.join(RUTA_DATA, "tabla_posiciones.csv"), index=False)
print("✅ Tabla de posiciones guardada")

# Datos de jugadores (goles y asistencias)
url_jug = "https://www.espn.com.co/futbol/estadisticas/_/liga/col.1/tipo/goles"
resp2 = requests.get(url_jug, headers=HEADERS)
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

# Simular minutos (por ahora; reemplazar con datos reales después)
np.random.seed(42)
df_jug['minutos_totales'] = df_jug['PJ'] * np.random.randint(60, 95, len(df_jug))
df_jug['minutos_totales'] = df_jug['minutos_totales'].replace(0, 90)
df_jug['goles_p90'] = (df_jug['Goles'] / (df_jug['minutos_totales'] / 90)).fillna(0)
df_jug['asistencias_p90'] = (df_jug['Asistencias'] / (df_jug['minutos_totales'] / 90)).fillna(0)
df_jug['recuperaciones_p90'] = np.random.exponential(8, len(df_jug)).round(1)
df_jug['duelos_ganados_p90'] = np.random.normal(12, 4, len(df_jug)).clip(2,25).round(1)
df_jug['pases_progresivos_p90'] = np.random.normal(25,10, len(df_jug)).clip(5,60).round(1)

df_jug.to_csv(os.path.join(RUTA_DATA, "jugadores_stats.csv"), index=False)
print("✅ Estadísticas de jugadores guardadas")

# ============================================
# 2. MOTOR DE SCOUTING (SIMILITUD)
# ============================================
def entrenar_scouting(df):
    features = ['goles_p90', 'asistencias_p90', 'recuperaciones_p90', 
                'duelos_ganados_p90', 'pases_progresivos_p90']
    X = df[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, 'models/scaler_scouting.pkl')
    joblib.dump(df[['Nombre', 'Equipo'] + features], 'models/referencia_scouting.pkl')
    print("✅ Modelo de scouting entrenado")

def recomendar_similares(nombre_jugador, top_n=5):
    try:
        scaler = joblib.load('models/scaler_scouting.pkl')
        ref = joblib.load('models/referencia_scouting.pkl')
    except:
        print("Modelo no encontrado, entrenando...")
        entrenar_scouting(df_jug)
        scaler = joblib.load('models/scaler_scouting.pkl')
        ref = joblib.load('models/referencia_scouting.pkl')
    if nombre_jugador not in ref['Nombre'].values:
        print(f"Jugador '{nombre_jugador}' no encontrado")
        return None
    features = ['goles_p90', 'asistencias_p90', 'recuperaciones_p90', 
                'duelos_ganados_p90', 'pases_progresivos_p90']
    X = scaler.transform(ref[features].values)
    idx = ref[ref['Nombre'] == nombre_jugador].index[0]
    sim = cosine_similarity([X[idx]], X)[0]
    top_idx = np.argsort(sim)[::-1][1:top_n+1]
    return ref.iloc[top_idx][['Nombre', 'Equipo'] + features].assign(similitud=sim[top_idx].round(3))

# Entrenar scouting
entrenar_scouting(df_jug)

# Ejemplo: recomendaciones
print("\n🔍 Jugadores similares a Andrey Estupiñán:")
print(recomendar_similares("Andrey Estupiñán").to_string(index=False))

# ============================================
# 3. MODELO DE RIESGO DE LESIÓN
# ============================================
def generar_datos_lesiones_simulados():
    np.random.seed(123)
    registros = []
    for jugador in range(200):
        resistencia_base = np.random.uniform(0.4, 1.0)
        propension_lesion = np.random.uniform(0.03, 0.25)
        for semana in range(1, 21):
            minutos_semana = np.random.randint(0, 180)
            carga_cronica = np.random.randint(200, 400)
            acwr = minutos_semana / (carga_cronica / 4 + 0.1)
            acwr = np.clip(acwr, 0.5, 2.5)
            sprints = np.random.poisson(minutos_semana / 15)
            aceleraciones = np.random.poisson(minutos_semana / 8)
            dias_descanso = np.random.choice([1,2,3,4,5,6,7], p=[0.15,0.25,0.3,0.15,0.08,0.04,0.03])
            fatiga = (minutos_semana / 90) * (2 - resistencia_base) + max(0, acwr - 1.2) * 1.5
            fatiga = np.clip(fatiga, 0, 4)
            riesgo = propension_lesion + fatiga * 0.15 + max(0, acwr - 1.3) * 0.25
            if dias_descanso <= 2: riesgo += 0.12
            if minutos_semana > 120: riesgo += 0.1
            if sprints > 20: riesgo += 0.05
            riesgo = np.clip(riesgo, 0, 0.9)
            lesion = 1 if np.random.random() < riesgo else 0
            registros.append([jugador, semana, minutos_semana, acwr, sprints, 
                             aceleraciones, dias_descanso, fatiga, lesion])
    cols = ['jugador_id', 'semana', 'minutos_semana', 'acwr', 'sprints', 
            'aceleraciones', 'dias_descanso', 'fatiga_acumulada', 'lesion_semana_siguiente']
    return pd.DataFrame(registros, columns=cols)

def entrenar_modelo_lesiones():
    df = generar_datos_lesiones_simulados()
    features = ['minutos_semana', 'acwr', 'sprints', 'aceleraciones', 'dias_descanso', 'fatiga_acumulada']
    X = df[features]
    y = df['lesion_semana_siguiente']
    modelo = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    modelo.fit(X, y)
    joblib.dump(modelo, 'models/modelo_lesiones.pkl')
    print("✅ Modelo de lesiones entrenado")
    return modelo

if not os.path.exists('models/modelo_lesiones.pkl'):
    entrenar_modelo_lesiones()
else:
    print("✅ Modelo de lesiones ya existe")

def predecir_riesgo_lesion(minutos, acwr, sprints, aceleraciones, descanso, fatiga):
    modelo = joblib.load('models/modelo_lesiones.pkl')
    entrada = np.array([[minutos, acwr, sprints, aceleraciones, descanso, fatiga]])
    prob = modelo.predict_proba(entrada)[0][1]
    riesgo = "🔴 ALTO" if prob > 0.6 else "🟡 MODERADO" if prob > 0.3 else "🟢 BAJO"
    return prob, riesgo

# Ejemplo
print("\n⚕️ Ejemplo de predicción de lesión:")
prob, riesgo = predecir_riesgo_lesion(160, 1.8, 25, 70, 1, 2.5)
print(f"Riesgo: {prob:.1%} {riesgo}")

# ============================================
# 4. GENERAR GRÁFICOS Y REPORTES FINALES
# ============================================
# Gráfico de puntos por equipo
plt.figure(figsize=(10,6))
plt.barh(df_posiciones['Equipo'][::-1], df_posiciones['PTS'][::-1], color='steelblue')
plt.title('Puntos por equipo - Liga BetPlay')
plt.tight_layout()
plt.savefig(os.path.join(RUTA_REPORTS, 'puntos_por_equipo.png'))
plt.close()

# Top 10 goleadores
top_g = df_jug.nlargest(10, 'Goles')
plt.figure(figsize=(10,6))
plt.barh(top_g['Nombre'][::-1], top_g['Goles'][::-1], color='green')
plt.title('Top 10 Goleadores')
plt.tight_layout()
plt.savefig(os.path.join(RUTA_REPORTS, 'top_goleadores.png'))
plt.close()

print("📊 Gráficos guardados en 'reports/'")

# Guardar historial simple
historial_path = os.path.join(RUTA_DATA, 'historial.json')
if os.path.exists(historial_path):
    with open(historial_path, 'r') as f:
        hist = json.load(f)
else:
    hist = {}
jornada = len(hist) + 1
hist[f'jornada_{jornada}'] = {
    'fecha': datetime.now().isoformat(),
    'lider': df_posiciones.iloc[0]['Equipo'],
    'max_goleador': df_jug.nlargest(1, 'Goles').iloc[0]['Nombre']
}
with open(historial_path, 'w') as f:
    json.dump(hist, f, indent=2)
print(f"📜 Historial jornada {jornada} guardado")

print("\n✅ PROCESO COMPLETADO EXITOSAMENTE")

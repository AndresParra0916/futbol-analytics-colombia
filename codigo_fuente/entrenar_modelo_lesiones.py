# entrenar_modelo_lesiones.py
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
import os

# ============================================
# CONFIGURACIÓN
# ============================================
INPUT_CSV = "data/gps_data.csv"          # Datos reales del club (formato específico)
SYNTHETIC_SIZE = 800                     # Si no hay datos, generar sintéticos
MODEL_PATH = "models/modelo_lesiones.pkl"
SCALER_PATH = "models/scaler_lesiones.pkl"

# ============================================
# 1. CARGAR O GENERAR DATOS (con todas las variables)
# ============================================
if os.path.exists(INPUT_CSV):
    print(f"📂 Cargando datos reales desde {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    df['date'] = pd.to_datetime(df['date'])
else:
    print("⚠️ No se encontró archivo de datos GPS reales.")
    print(f"🔧 Generando {SYNTHETIC_SIZE} registros sintéticos para demostración...")
    np.random.seed(42)
    players = [f"Jugador_{i}" for i in range(1, 21)]
    dates = pd.date_range('2026-01-01', '2026-05-31', freq='D')
    synthetic_data = []
    for player in np.random.choice(players, SYNTHETIC_SIZE):
        date = np.random.choice(dates)
        minutes = np.random.choice([0, 60, 75, 90, 120], p=[0.1, 0.3, 0.3, 0.2, 0.1])
        # Métricas GPS
        high_intensity_dist = np.random.normal(800, 200) if minutes > 30 else 0
        sprints = np.random.poisson(15) if minutes > 30 else 0
        accelerations = np.random.poisson(30) if minutes > 30 else 0
        decelerations = np.random.poisson(35) if minutes > 30 else 0
        # Días de descanso (simulado: días desde último partido/entrenamiento intenso)
        rest_days = np.random.choice([0, 1, 2, 3, 4, 5], p=[0.1, 0.2, 0.3, 0.2, 0.1, 0.1])
        # Carga semanal total (simulada)
        weekly_load = np.random.normal(3500, 800)
        # Variable de fatiga (simulada)
        fatigue = np.random.normal(5, 2)
        # Probabilidad de lesión (función de las variables)
        injury_prob = 0.02 + 0.003 * (high_intensity_dist / 100) + 0.01 * sprints + 0.005 * accelerations \
                      + 0.005 * decelerations - 0.02 * rest_days + 0.0001 * weekly_load + 0.01 * fatigue
        injury_prob = np.clip(injury_prob, 0.01, 0.6)
        injury = 1 if np.random.random() < injury_prob else 0
        synthetic_data.append([player, date, minutes, high_intensity_dist, sprints,
                               accelerations, decelerations, rest_days, weekly_load, fatigue, injury])
    df = pd.DataFrame(synthetic_data, columns=['player_name', 'date', 'minutes',
                                                'high_intensity_dist', 'sprints',
                                                'accelerations', 'decelerations',
                                                'rest_days', 'weekly_load', 'fatigue', 'injury'])
    print(f"✅ {len(df)} registros sintéticos generados.\n")

# ============================================
# 2. AGREGACIÓN POR JUGADOR Y SEMANA
# ============================================
df['week'] = df['date'].dt.isocalendar().week
df['year'] = df['date'].dt.year
# Agrupar por jugador, año, semana (sumar métricas, promediar descanso, etc.)
weekly = df.groupby(['player_name', 'year', 'week']).agg({
    'minutes': 'sum',
    'high_intensity_dist': 'sum',
    'sprints': 'sum',
    'accelerations': 'sum',
    'decelerations': 'sum',
    'rest_days': 'mean',          # promedio de días de descanso en la semana
    'weekly_load': 'sum',
    'fatigue': 'mean',
    'injury': 'max'               # si hubo al menos una lesión
}).reset_index()

# ============================================
# 3. CALCULAR ACWR Y OTRAS CARACTERÍSTICAS
# ============================================
def acwr(series, acute=1, chronic=4):
    chronic_avg = series.rolling(chronic, min_periods=1).mean()
    acute_val = series.rolling(acute, min_periods=1).mean()
    ratio = acute_val / chronic_avg
    return ratio.replace([np.inf, -np.inf], 1).fillna(1)

for metric in ['high_intensity_dist', 'sprints', 'accelerations', 'decelerations', 'weekly_load']:
    weekly[f'{metric}_acwr'] = weekly.groupby('player_name')[metric].transform(acwr)

# Características finales: ACWR de las métricas + días de descanso + fatiga + minutos
features = ['high_intensity_dist_acwr', 'sprints_acwr', 'accelerations_acwr', 'decelerations_acwr',
            'weekly_load_acwr', 'rest_days', 'fatigue', 'minutes']
X = weekly[features].fillna(1).values
y = weekly['injury'].values

print(f"📊 Datos preparados: {len(X)} registros semanales, {X.shape[1]} características.")
print(f"⚽ Proporción de lesiones: {y.mean():.2%}")

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Escalar
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================
# 4. ENTRENAR MODELO RANDOM FOREST
# ============================================
model = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42, class_weight='balanced')
model.fit(X_train_scaled, y_train)

# Evaluación
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]
print("\n📈 Evaluación del modelo:")
print(classification_report(y_test, y_pred))
print(f"AUC-ROC: {roc_auc_score(y_test, y_proba):.3f}")

# ============================================
# 5. GUARDAR MODELO Y ESCALADOR
# ============================================
os.makedirs('models', exist_ok=True)
joblib.dump(model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)
print(f"\n✅ Modelo guardado en {MODEL_PATH}")
print(f"✅ Escalador guardado en {SCALER_PATH}")
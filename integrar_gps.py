"""
Script para integrar datos GPS reales de un club y entrenar modelo de lesiones.
Uso: python integrar_gps.py ruta_del_archivo.csv
"""

import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt
import os

# ============================================
# 1. CARGAR Y VALIDAR DATOS
# ============================================
def cargar_datos(ruta_csv):
    """Carga el archivo CSV y verifica columnas esenciales."""
    df = pd.read_csv(ruta_csv)
    
    # Columnas obligatorias
    obligatorias = ['jugador_id', 'fecha', 'minutos_semana', 'dias_descanso', 'lesion_semana_siguiente']
    for col in obligatorias:
        if col not in df.columns:
            raise ValueError(f"Falta la columna obligatoria: {col}")
    
    # Columnas opcionales (si no existen, se crean con valores por defecto)
    if 'sprints' not in df.columns:
        df['sprints'] = 0
        print("⚠️ Columna 'sprints' no encontrada. Se usará 0.")
    if 'aceleraciones' not in df.columns:
        df['aceleraciones'] = 0
        print("⚠️ Columna 'aceleraciones' no encontrada. Se usará 0.")
    if 'distancia_total' not in df.columns:
        df['distancia_total'] = 0
        print("⚠️ Columna 'distancia_total' no encontrada. Se usará 0.")
    
    return df

# ============================================
# 2. CALCULAR ACWR (RATIO AGUDA/CRÓNICA)
# ============================================
def calcular_acwr(df):
    """
    Calcula el ACWR (carga aguda/crónica) por jugador.
    Carga aguda = minutos de la semana actual.
    Carga crónica = promedio de minutos de las últimas 4 semanas.
    """
    df = df.sort_values(['jugador_id', 'fecha'])
    df['acwr'] = 0.0
    
    for jugador in df['jugador_id'].unique():
        mask = df['jugador_id'] == jugador
        minutos = df.loc[mask, 'minutos_semana'].values
        acwr = []
        for i in range(len(minutos)):
            if i < 4:
                acwr.append(0.5)  # valor por defecto si no hay historial
            else:
                carga_aguda = minutos[i]
                carga_cronica = np.mean(minutos[i-4:i])
                acwr.append(carga_aguda / (carga_cronica + 0.1))
        df.loc[mask, 'acwr'] = acwr
    return df

# ============================================
# 3. ENTRENAR MODELO DE RANDOM FOREST
# ============================================
def entrenar_modelo(df):
    """Entrena y guarda el modelo de lesiones."""
    features = ['minutos_semana', 'acwr', 'sprints', 'aceleraciones', 
            'distancia_total', 'dias_descanso']
    X = df[features].fillna(0).values
    y = df['lesion_semana_siguiente'].values
    
    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Entrenar Random Forest
    modelo = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    modelo.fit(X_train, y_train)
    
    # Evaluar
    y_pred = modelo.predict(X_test)
    auc = roc_auc_score(y_test, modelo.predict_proba(X_test)[:, 1])
    precision = modelo.score(X_test, y_test)
    
    print(f"✅ Modelo entrenado con {len(df)} registros.")
    print(f"📊 Precisión en prueba: {precision:.2%}")
    print(f"📊 AUC ROC: {auc:.3f}")
    
    # Mostrar importancia de variables
    importancias = pd.DataFrame({
        'variable': features,
        'importancia': modelo.feature_importances_
    }).sort_values('importancia', ascending=False)
    print("\n📈 Importancia de variables:")
    print(importancias.to_string(index=False))
    
    # Guardar modelo
    joblib.dump(modelo, 'models/modelo_lesiones_gps.pkl')
    print("💾 Modelo guardado en 'models/modelo_lesiones_gps.pkl'")
    
    return modelo, importancias

# ============================================
# 4. GENERAR REPORTE PARA EL CLUB
# ============================================
def generar_reporte(df, importancias, modelo, X_test, y_test):
    """Crea un reporte simple en texto y gráfico."""
    # Gráfico de importancia de variables
    plt.figure(figsize=(8,5))
    plt.barh(importancias['variable'], importancias['importancia'], color='steelblue')
    plt.xlabel('Importancia')
    plt.title('Factores que más influyen en lesiones')
    plt.tight_layout()
    plt.savefig('reports/importancia_variables_gps.png')
    plt.close()
    
    # Reporte textual
    with open('reports/reporte_gps_club.txt', 'w') as f:
        f.write("=== REPORTE DE ANÁLISIS DE RIESGO DE LESIÓN ===\n\n")
        f.write(f"Total de registros procesados: {len(df)}\n")
        f.write(f"Total de jugadores únicos: {df['jugador_id'].nunique()}\n")
        f.write(f"Tasa de lesiones en el período: {df['lesion_semana_siguiente'].mean():.1%}\n\n")
        f.write("Variables más importantes para predecir lesiones:\n")
        for _, row in importancias.iterrows():
            f.write(f"- {row['variable']}: {row['importancia']:.3f}\n")
        f.write("\nEl modelo puede predecir el riesgo individual con una precisión del ")
        f.write(f"{modelo.score(X_test, y_test):.1%}.\n")
    
    print("📄 Reporte guardado en 'reports/reporte_gps_club.txt'")
    print("📊 Gráfico guardado en 'reports/importancia_variables_gps.png'")

# ============================================
# 5. FUNCIÓN PRINCIPAL (ejecutar desde terminal)
# ============================================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ Uso: python integrar_gps.py ruta_del_archivo.csv")
        sys.exit(1)
    
    ruta = sys.argv[1]
    print(f"🔄 Procesando archivo: {ruta}")
    
    # Crear carpetas necesarias
    os.makedirs('models', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    
    try:
        # 1. Cargar datos
        df = cargar_datos(ruta)
        print(f"✅ Datos cargados: {len(df)} registros, {df['jugador_id'].nunique()} jugadores")
        
        # 2. Calcular ACWR
        df = calcular_acwr(df)
        print("✅ ACWR calculado")
        
        # 3. Entrenar modelo
        modelo, importancias = entrenar_modelo(df)
        
        # 4. Generar reporte (necesita datos de prueba)
        # Para obtener X_test, y_test, re-ejecutamos split (no es óptimo pero funciona)
        features = ['minutos_semana', 'acwr', 'sprints', 'aceleraciones', 
                    'distancia_total', 'dias_descanso']
        X = df[features].fillna(0).values
        y = df['lesion_semana_siguiente'].values
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        generar_reporte(df, importancias, modelo, X_test, y_test)
        
        print("\n🎉 ¡Proceso completado! Ahora puedes actualizar el dashboard para usar este modelo.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

import matplotlib.pyplot as plt
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Cargar modelo y referencia
ref = joblib.load('models/referencia_scouting_2025.pkl')
scaler = joblib.load('models/scaler_scouting_2025.pkl')
features = ['goles_p90', 'asistencias_p90', 'tiros_p90', 'pases_p90', 'entradas_p90', 'duelos_ganados_p90']

def generar_radar(nombre_jugador):
    if nombre_jugador not in ref['player_name'].values:
        print(f"Jugador '{nombre_jugador}' no encontrado")
        return
    
    idx = ref[ref['player_name'] == nombre_jugador].index[0]
    valores = ref.iloc[idx][features].values
    valores_norm = scaler.transform([valores])[0]
    
    # Crear gráfico radar
    angulos = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist()
    valores_norm = np.append(valores_norm, valores_norm[0])
    angulos += angulos[:1]
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': 'polar'})
    ax.plot(angulos, valores_norm, 'o-', linewidth=2, color='#1f77b4')
    ax.fill(angulos, valores_norm, alpha=0.25, color='#1f77b4')
    ax.set_xticks(angulos[:-1])
    ax.set_xticklabels(features, fontsize=8)
    ax.set_title(f'Perfil de {nombre_jugador}\n(métricas normalizadas)', size=12, pad=20)
    plt.tight_layout()
    
    # Guardar imagen
    nombre_archivo = f'data/radar_{nombre_jugador.replace(" ", "_")}.png'
    plt.savefig(nombre_archivo, dpi=150, bbox_inches='tight')
    print(f"✅ Gráfico guardado: {nombre_archivo}")
    plt.close()

# Ejemplo: generar radar de un jugador destacado
generar_radar('Dayro Moreno')
print("Puedes cambiar el nombre en el script para generar más gráficos.")
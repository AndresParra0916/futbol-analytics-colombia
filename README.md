# ⚽ Futbol Analytics Colombia

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://futbol-analytics-colombia-opu2t4oemfqkbkxnvpwfeg.streamlit.app)
[![GitHub Actions](https://github.com/AndresParra0916/futbol-analytics-colombia/actions/workflows/actualizar_jornada.yml/badge.svg)](https://github.com/AndresParra0916/futbol-analytics-colombia/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Sistema de análisis de datos para clubes de fútbol profesional colombiano.**
- 🔍 Scouting inteligente por similitud estadística
- ⚕️ Predicción de riesgo de lesiones por fatiga (modelo ACWR)
- 📊 Automatización semanal con GitHub Actions
- 🌐 Dashboard interactivo en tiempo real

## 🚀 Demo en vivo
[**Ver dashboard interactivo**](https://futbol-analytics-colombia-opu2t4oemfqkbkxnvpwfeg.streamlit.app)

## 📊 Características principales
- **Scraping real** de estadísticas de la Liga BetPlay (ESPN)
- **Motor de similitud** (cosine similarity) para encontrar jugadores con perfiles equivalentes
- **Modelo predictivo** de lesiones basado en carga aguda/crónica (ACWR), minutos, sprints y descanso
- **Automatización completa** – se actualiza cada lunes y jueves sin intervención manual
- **Dashboard** con tablas, gráficos interactivos y recomendaciones

## 🛠️ Tecnologías utilizadas
| Capa | Tecnología |
|------|-------------|
| Data scraping | Python, BeautifulSoup, Requests |
| Procesamiento | Pandas, NumPy |
| Machine Learning | Scikit-learn (StandardScaler, Cosine Similarity, Random Forest) |
| Visualización | Matplotlib, Seaborn, Plotly |
| Dashboard | Streamlit |
| Automatización | GitHub Actions |
| Control de versiones | Git, GitHub |

## 📈 Ejemplo de salida – Scouting
Recomendaciones para **Andrey Estupiñán** (máximo goleador):
| Jugador | Equipo | Goles/90 | Similitud |
|---------|--------|----------|------------|
| Alfredo Morelos | Atlético Nacional | 0.93 | 98.4% |
| Steven Rodríguez | Deportivo Cali | 0.59 | 85.3% |
| Jaime Peralta | Cúcuta Deportivo | 0.62 | 80.7% |

## ⚕️ Modelo de lesiones (simulado con lógica real)
- Variables: minutos semanales, ACWR, sprints, aceleraciones, días de descanso, fatiga acumulada
- Algoritmo: Random Forest
- Próximo paso: integración con datos GPS reales de clubes

## 🔄 Automatización
- El script se ejecuta automáticamente cada **lunes y jueves a las 10:00 AM (hora Colombia)**
- Los resultados se guardan en el repositorio y el dashboard se actualiza en tiempo real

## 📁 Estructura del proyecto

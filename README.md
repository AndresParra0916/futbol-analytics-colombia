# ⚽ Futbol Analytics Colombia

Sistema de análisis de datos para clubes de fútbol colombiano.

## 🚀 Características
- **Scraping automático** de datos reales de la Liga BetPlay (ESPN).
- **Motor de scouting** por similitud para encontrar jugadores con perfiles similares.
- **Modelo predictivo de riesgo de lesiones** basado en fatiga y carga (ACWR).
- **Generación de reportes y gráficos** automáticos.
- **Automatización semanal** con GitHub Actions.

## 📊 Ejemplo de salida
- Tabla de posiciones actualizada (`data/tabla_posiciones.csv`)
- Top goleadores y asistentes (`data/jugadores_stats.csv`)
- Recomendaciones de jugadores similares
- Gráficos de puntos por equipo y top goleadores (`reports/`)

## 🛠️ Tecnologías
Python, pandas, scikit-learn, BeautifulSoup, matplotlib, GitHub Actions.

## 📅 Automatización
El sistema se actualiza cada lunes y jueves a las 10 AM (hora Colombia).

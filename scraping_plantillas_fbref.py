import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

def get_team_links():
    """Obtiene los enlaces de todos los equipos desde la página de la liga."""
    url = "https://fbref.com/en/comps/24/2026/2026-Categoria-Primera-A-Stats"
    print("🔍 Obteniendo enlaces de equipos...")
    response = requests.get(url, headers=HEADERS, timeout=20)
    soup = BeautifulSoup(response.text, 'html.parser')
    # Buscar la tabla de equipos (clase 'stats_table')
    table = soup.find('table', class_='stats_table')
    if not table:
        print("No se encontró la tabla de equipos.")
        return []
    links = []
    for link in table.find_all('a', href=True):
        href = link['href']
        if '/squads/' in href:
            full_url = 'https://fbref.com' + href
            team_name = link.text.strip()
            if team_name and full_url not in [l[1] for l in links]:
                links.append((team_name, full_url))
    print(f"✅ Encontrados {len(links)} equipos.")
    return links

def get_squad(team_name, squad_url):
    """Extrae la plantilla completa desde la página del equipo."""
    print(f"🔍 Extrayendo plantilla de {team_name}...")
    try:
        response = requests.get(squad_url, headers=HEADERS, timeout=20)
        soup = BeautifulSoup(response.text, 'html.parser')
        # La tabla de jugadores suele tener id 'stats_standard' o clase 'stats_table'
        table = soup.find('table', id='stats_standard')
        if not table:
            table = soup.find('table', class_='stats_table')
        if not table:
            print(f"⚠️ No se encontró tabla para {team_name}")
            return []
        # Extraer filas
        rows = table.find_all('tr')
        players = []
        for row in rows:
            # Saltar filas de encabezado (th)
            if row.find('th'):
                continue
            cells = row.find_all('td')
            if len(cells) < 4:
                continue
            # Las columnas típicas: Número (opcional), Posición, Nombre, Nacionalidad, Edad, etc.
            # En FBref, la primera celda (th) a veces tiene el número y nombre
            # Usamos la celda de nombre (suele ser la segunda o tercera)
            name_cell = row.find('th', {'data-stat': 'player'})
            if name_cell:
                nombre = name_cell.get_text(strip=True)
            else:
                # Alternativa: buscar en td con data-stat='player'
                name_cell = row.find('td', {'data-stat': 'player'})
                nombre = name_cell.get_text(strip=True) if name_cell else ""
            if not nombre:
                continue
            # Posición
            pos_cell = row.find('td', {'data-stat': 'position'})
            posicion = pos_cell.get_text(strip=True) if pos_cell else ""
            # Número de camiseta (si existe)
            num_cell = row.find('td', {'data-stat': 'jersey_number'})
            numero = num_cell.get_text(strip=True) if num_cell else ""
            # Nacionalidad
            nat_cell = row.find('td', {'data-stat': 'nationality'})
            nacionalidad = nat_cell.get_text(strip=True) if nat_cell else ""
            # Edad
            age_cell = row.find('td', {'data-stat': 'age'})
            edad = age_cell.get_text(strip=True) if age_cell else ""
            players.append({
                'equipo': team_name,
                'numero': numero,
                'nombre': nombre,
                'posicion': posicion,
                'edad': edad,
                'nacionalidad': nacionalidad
            })
        print(f"✅ {team_name}: {len(players)} jugadores")
        return players
    except Exception as e:
        print(f"❌ Error en {team_name}: {e}")
        return []

def main():
    team_links = get_team_links()
    if not team_links:
        return
    all_players = []
    for team_name, squad_url in team_links:
        players = get_squad(team_name, squad_url)
        all_players.extend(players)
        time.sleep(2)  # Pausa para no sobrecargar
    if all_players:
        import os
        os.makedirs('data', exist_ok=True)
        df = pd.DataFrame(all_players)
        df.to_csv('data/plantilla_completa_fbref.csv', index=False, encoding='utf-8-sig')
        print(f"\n🎉 {len(df)} jugadores guardados en 'data/plantilla_completa_fbref.csv'")
    else:
        print("No se extrajo ninguna plantilla.")

if __name__ == "__main__":
    main()
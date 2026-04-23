import requests

API_KEY = "ebb8f00138af0df132bbda386d55981c"
headers = {"x-apisports-key": API_KEY}

# Endpoint simple para obtener el estado de la API
url = "https://v3.football.api-sports.io/status"
response = requests.get(url, headers=headers)

print("Código de estado:", response.status_code)
if response.status_code == 200:
    print("✅ Conexión exitosa con la API.")
    print("Respuesta:", response.json())
else:
    print("❌ Error:", response.text)
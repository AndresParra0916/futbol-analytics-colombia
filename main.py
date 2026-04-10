import sys
import traceback
import os

try:
    print("Iniciando script...")
    
    # Crear directorio para logs
    os.makedirs('logs', exist_ok=True)
    
    # Prueba 1: escribir un archivo simple
    with open('logs/prueba.txt', 'w') as f:
        f.write('El script se ejecutó correctamente\n')
    print("✅ Archivo prueba.txt creado")
    
    # Aquí puedes ir añadiendo tu código real paso a paso
    
    print("Script finalizado sin errores")
    
except Exception as e:
    # Capturar cualquier error y guardarlo en un archivo
    error_msg = traceback.format_exc()
    with open('logs/error.log', 'w') as f:
        f.write(error_msg)
    print(f"❌ ERROR: {str(e)}")
    print("Error guardado en logs/error.log")
    # Hacer que el script falle para que GitHub Actions lo marque como error
    sys.exit(1)

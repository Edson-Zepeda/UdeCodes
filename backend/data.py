import pandas as pd
import sqlite3
from pathlib import Path
import sys

# --- 1. CONFIGURACIÓN ---
ROOT = Path.cwd()
UNIFIED_DB = 'Datos_Tabulares_SQLITE.sqlite'

# Archivos de origen
FILES = {
    'ProductivityData': '[HackMTY2025]_ProductivityEstimation_Dataset_v1.xlsx',
    'ExpirationData': '[HackMTY2025]_ExpirationDateManagement_Dataset_v1.xlsx',
    'ConsumptionData': '[HackMTY2025]_ConsumptionPrediction_Dataset_v1.xlsx',
    'EmployeeEfficency': '[HackMTY2025]_EmployeeEfficiency_Dataset_v1.xlsx',
}

def verify_files():
    """Verifica que existan los archivos necesarios"""
    missing = []
    for name, file in FILES.items():
        path = ROOT / file
        if not path.exists():
            missing.append(file)
    
    if missing:
        print("❌ Archivos faltantes:")
        for file in missing:
            print(f"   - {file}")
        return False
    return True

def main():
    print(f"📂 Directorio de trabajo: {ROOT}")
    
    # Verificar archivos
    if not verify_files():
        return 1

    try:
        # Cargar datos
        dfs = {}
        for name, file in FILES.items():
            print(f"📊 Cargando {file}...")
            dfs[name] = pd.read_excel(ROOT / file)

        # Crear base de datos
        print(f"💾 Creando base de datos {UNIFIED_DB}...")
        with sqlite3.connect(UNIFIED_DB) as conn:
            for table_name, df in dfs.items():
                df.to_sql(table_name, conn, if_exists='replace', index=False)
                print(f"✅ Tabla '{table_name}' creada con {len(df)} filas")

        print(f"\n🎉 Base de datos creada exitosamente: {UNIFIED_DB}")
        return 0

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return 1

if __name__ == "__main__":
    # Asegurar que pandas está instalado
    try:
        import pandas as pd
    except ImportError:
        print("❌ Pandas no está instalado. Ejecuta:")
        print("python -m pip install pandas openpyxl")
        sys.exit(1)
    
    sys.exit(main())
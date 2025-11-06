# Artefactos del Backend

El backend busca el modelo de consumo en:

- `backend/models/consumption_prediction_xgb.pkl`

Para generarlo, ejecuta el notebook de consumo y asegúrate de guardar el artefacto exactamente en esa ruta.

## Generar localmente (Windows PowerShell)

1. Crear entorno virtual e instalar dependencias

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

2. Ejecutar el notebook de forma no interactiva

```powershell
python -m jupyter nbconvert --execute --inplace notebooks\consumption_prediction.ipynb
```

3. Verificar que el archivo exista

```powershell
Get-Item backend\models\consumption_prediction_xgb.pkl
```

4. Hacer commit con Git LFS

```powershell
git lfs install
git lfs track "backend/models/*.pkl"
# verifica que .gitattributes contenga el patrón anterior
Get-Content .gitattributes

git add .gitattributes backend/models/consumption_prediction_xgb.pkl
git commit -m "Agregar artefacto del modelo de consumo (.pkl)"
git push
```

Una vez publicado, Railway redeplegará y `/predict/financial-impact` devolverá valores distintos de cero.

## Notas
- Si el artefacto no está presente, la API devuelve ceros (sin error) como fallback seguro.
- Los archivos grandes se almacenan vía Git LFS para mantener el repositorio ligero.


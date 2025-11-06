# Guía de Demo

Este proyecto incluye un backend de demo ligero para que puedas ejecutar el frontend sin depender del artefacto pesado del modelo.

Pasos:

1. Copia el logo a la carpeta `public` del frontend (para que cargue la imagen del header):
   - Desde la raíz del repositorio existe `Gategroup_logo.png`.
   - Copia el archivo a `frontend/public` como `Gategroup_logo.png`.

2. Instala dependencias del frontend y ejecuta Vite (desde la raíz del repo):

```powershell
cd frontend
npm install
npm run dev
```

3. Inicia el backend de demo (servidor ligero). Abre una nueva terminal y ejecuta:

```powershell
cd backend
# requiere uvicorn instalado en tu entorno de Python
uvicorn demo_app:app --reload --port 8000
```

4. Abre http://127.0.0.1:5173 (servidor de Vite). El frontend llamará al backend de demo en http://127.0.0.1:8000.

Notas:
- La aplicación completa del backend (`backend/app/main.py`) depende de un artefacto de modelo entrenado y no es necesaria para la demo.
- Si quieres usar el backend completo, ejecuta el notebook de entrenamiento para producir el artefacto en `backend/models/global_flights_forecasting_model.pkl`.


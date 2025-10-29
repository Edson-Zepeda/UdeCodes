Demo setup

This project includes a lightweight demo backend so you can run the frontend without the heavy model artifact.

Steps:

1. Copy the logo to the frontend public folder (so the header image loads):
   - From the repo root there is `Gategroup_logo.png`.
   - Copy it into `frontend/public` as `Gategroup_logo.png`.

2. Install frontend deps and run Vite (from repository root):

```powershell
cd c:/Users/Lenovo/Desktop/UdeCodes/frontend
npm install
npm run dev
```

3. Start the demo backend (new lightweight server). Open a new terminal and run:

```powershell
cd c:/Users/Lenovo/Desktop/UdeCodes/backend
# requires uvicorn to be installed in your Python env
uvicorn demo_app:app --reload --port 8000
```

4. Open http://127.0.0.1:5173 (Vite dev server) and the frontend will call the demo backend at http://127.0.0.1:8000.

Notes:
- The full backend application (`backend/app/main.py`) depends on a trained model artifact and is not required for the demo.
- If you want the full backend, run the training notebook to produce the model artifact at `backend/models/global_flights_forecasting_model.pkl`.

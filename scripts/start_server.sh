#!/usr/bin/env bash
set -euo pipefail

# Pick python or python3
if command -v python >/dev/null 2>&1; then PY=python;
elif command -v python3 >/dev/null 2>&1; then PY=python3;
else echo "[start_server] Python not found in container" >&2; exit 1; fi

mkdir -p backend/models

"$PY" - <<'PY'
import os, os.path, urllib.request
url = os.environ.get('MODEL_URL')
dst = 'backend/models/consumption_prediction_xgb.pkl'
if url and not os.path.exists(dst):
    print(f"[start_server] Downloading model from {url} -> {dst}", flush=True)
    urllib.request.urlretrieve(url, dst)
else:
    print("[start_server] MODEL_URL not set or artifact already present; skipping download.", flush=True)
PY

exec uvicorn backend.app.main:app --host 0.0.0.0 --port "${PORT:-8000}"

#!/usr/bin/env bash
set -euo pipefail

mkdir -p backend/models

python - <<'PY'
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
# Backend artifacts

This backend looks for the consumption model at:

- `backend/models/consumption_prediction_xgb.pkl`

To generate it, execute the consumption notebook and make sure the artifact is saved to that exact path.

## Generate locally (Windows PowerShell)

1. Create venv and install deps:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

2. Run the notebook headlessly:

```powershell
python -m jupyter nbconvert --execute --inplace notebooks\consumption_prediction.ipynb
```

3. Verify file exists:

```powershell
Get-Item backend\models\consumption_prediction_xgb.pkl
```

4. Commit via Git LFS:

```powershell
git lfs install
git lfs track "backend/models/*.pkl"
# ensure .gitattributes contains the pattern above
Get-Content .gitattributes

git add .gitattributes backend/models/consumption_prediction_xgb.pkl
git commit -m "Add consumption model artifact (.pkl)"
git push
```

Once pushed, Railway will redeploy and `/predict/financial-impact` will return non‑zero values.

## Notes
- If the artifact is not present, the API returns zeros (no error) as a safe fallback.
- Large files are stored via Git LFS to keep the repo light.

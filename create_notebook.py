import nbformat as nbf
from textwrap import dedent

nb = nbf.v4.new_notebook()

cells = []

cells.append(nbf.v4.new_markdown_cell("""
# Refined Forecasting Model

Pipeline to engineer advanced features (holidays, lag, rolling windows) and tune XGBoost to minimise MAPE on daily flights.
"""))

cells.append(nbf.v4.new_code_cell(dedent("""
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
import xgboost as xgb
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import make_scorer
""")))

cells.append(nbf.v4.new_code_cell(dedent("""
DATA_DIR = Path('../data')
CSV_GLOB = 'Plant*.csv'

frames = []
for path in sorted(DATA_DIR.glob(CSV_GLOB)):
    match = re.search(r"Plant\\s*([0-9]+)", path.stem)
    if not match:
        continue
    plant_id = int(match.group(1))
    df = pd.read_csv(path, encoding='utf-8-sig', low_memory=False)
    df['plant_id'] = plant_id
    frames.append(df)

if not frames:
    raise RuntimeError('No se encontraron archivos Plant*.csv en ../data.')

df_raw = pd.concat(frames, ignore_index=True)
df_raw.columns = [col.strip().lower().replace(' ', '_') for col in df_raw.columns]
df_raw['day'] = pd.to_datetime(df_raw['day'], errors='coerce')
df_raw = df_raw.dropna(subset=['day']).sort_values(['plant_id', 'day']).reset_index(drop=True)

print('Registros por planta:')
print(df_raw.groupby('plant_id')['day'].agg(['min', 'max', 'count']))
df_raw.head()
""")))

cells.append(nbf.v4.new_code_cell(dedent("""
df = df_raw.copy()

# Temporal context
df['dayofweek'] = df['day'].dt.dayofweek
df['month'] = df['day'].dt.month
df['quarter'] = df['day'].dt.quarter
df['year'] = df['day'].dt.year
df['dayofyear'] = df['day'].dt.dayofyear
df['weekofyear'] = df['day'].dt.isocalendar().week.astype(int)
df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)

# Holidays
cal = USFederalHolidayCalendar()
holidays = cal.holidays(start=df['day'].min(), end=df['day'].max())
df['is_holiday'] = df['day'].isin(holidays).astype(int)

# Cyclical encoding
df['sin_dayofyear'] = np.sin(2 * np.pi * df['dayofyear'] / 365.0)
df['cos_dayofyear'] = np.cos(2 * np.pi * df['dayofyear'] / 365.0)

LAGS = [1, 2, 3, 7, 14]
for lag in LAGS:
    df[f'lag_{lag}'] = df.groupby('plant_id')['flights'].shift(lag)

ROLL_WINDOWS = [3, 7, 14, 28]
for window in ROLL_WINDOWS:
    shifted = df.groupby('plant_id')['flights'].shift(1)
    df[f'roll_mean_{window}'] = shifted.rolling(window=window, min_periods=2).mean()
    df[f'roll_std_{window}'] = shifted.rolling(window=window, min_periods=2).std()

for window in ROLL_WINDOWS:
    df[f'roll_std_{window}'] = df[f'roll_std_{window}'].fillna(0)

required = [f'lag_{lag}' for lag in LAGS] + [f'roll_mean_{window}' for window in ROLL_WINDOWS]
df_features = df.dropna(subset=required).reset_index(drop=True)

FEATURES = [
    'plant_id', 'dayofweek', 'month', 'quarter', 'year', 'dayofyear', 'weekofyear',
    'is_weekend', 'is_holiday', 'sin_dayofyear', 'cos_dayofyear'
] + [f'lag_{lag}' for lag in LAGS] + [f'roll_mean_{window}' for window in ROLL_WINDOWS] + [f'roll_std_{window}' for window in ROLL_WINDOWS]

TARGET = 'flights'
print('Total registros tras ingeniería de variables:', len(df_features))
df_features.head()
""")))

cells.append(nbf.v4.new_code_cell(dedent("""
last_date = df_features['day'].max()
cutoff_date = last_date - pd.DateOffset(months=3)

train_df = df_features[df_features['day'] <= cutoff_date].copy()
test_df = df_features[df_features['day'] > cutoff_date].copy()

print(f"Train: {train_df['day'].min()} -> {train_df['day'].max()} | {len(train_df)} registros")
print(f"Test : {test_df['day'].min()} -> {test_df['day'].max()} | {len(test_df)} registros")

X_train = train_df[FEATURES]
y_train = train_df[TARGET]
X_test = test_df[FEATURES]
y_test = test_df[TARGET]
""")))

cells.append(nbf.v4.new_code_cell(dedent("""
param_distributions = {
    'n_estimators': [300, 400, 500, 600],
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.03, 0.05, 0.07],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.3],
    'reg_lambda': [1, 2, 3]
}

base_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    tree_method='hist',
    random_state=42,
    n_jobs=1
)

ts_cv = TimeSeriesSplit(n_splits=5)
scorer = make_scorer(mean_absolute_percentage_error, greater_is_better=False)

random_search = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_distributions,
    n_iter=20,
    scoring=scorer,
    cv=ts_cv,
    verbose=1,
    random_state=42,
    n_jobs=1
)

random_search.fit(X_train, y_train)

best_model = random_search.best_estimator_
print('Mejores hiperparámetros encontrados:')
print(random_search.best_params_)
""")))

cells.append(nbf.v4.new_code_cell(dedent("""
y_pred_test = best_model.predict(X_test)
y_pred_train = best_model.predict(X_train)

test_mape = mean_absolute_percentage_error(y_test, y_pred_test)
train_mape = mean_absolute_percentage_error(y_train, y_pred_train)

print(f"Test MAPE: {test_mape * 100:.2f}%")
print(f"Train MAPE: {train_mape * 100:.2f}%")

if test_mape <= 0.02:
    print("\\n✅ Objetivo de MAPE ≤ 2% alcanzado.")
else:
    print("\\n⚠️  Objetivo de MAPE ≤ 2% no alcanzado con el dataset actual.")
""")))

cells.append(nbf.v4.new_code_cell(dedent("""
# Entrenar con todo el conjunto y guardar artefacto
best_model.fit(df_features[FEATURES], df_features[TARGET])
output_path = Path('../backend/models/refined_flights_forecasting_model.pkl')
output_path.parent.mkdir(parents=True, exist_ok=True)

payload = {
    'model': best_model,
    'features': FEATURES,
    'best_params': random_search.best_params_,
    'train_mape': train_mape,
    'test_mape': test_mape,
    'cutoff_date': cutoff_date
}

joblib.dump(payload, output_path)
print(f"Modelo guardado en {output_path.resolve()}")
""")))

cells.append(nbf.v4.new_markdown_cell("""
## Resultados y Observaciones

- Se incorporaron variables de calendario, festivos federales, rezagos y ventanas móviles.
- Se utilizó `RandomizedSearchCV` con validación `TimeSeriesSplit` (5 particiones) para ajustar hiperparámetros de XGBoost.
- Con los datos disponibles, el MAPE en el conjunto de prueba continúa por encima del objetivo de 2% (≈10%).
- Recomendación: depurar/normalizar los datos históricos (por ejemplo, coherencia de Plant 2) o explorar modelos específicos por planta que reduzcan la variabilidad.
"""))

nb['cells'] = cells
nb['metadata'] = {
    'kernelspec': {
        'display_name': 'Python 3',
        'language': 'python',
        'name': 'python3'
    },
    'language_info': {
        'name': 'python',
        'version': '3.11'
    }
}

nbf.write(nb, 'notebooks/forecasting_model.ipynb')

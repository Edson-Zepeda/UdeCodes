import re
from pathlib import Path
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
import xgboost as xgb

DATA_DIR = Path('data')
frames = []
for path in sorted(DATA_DIR.glob('Plant*.csv')):
    m = re.search(r'Plant\s*([0-9]+)', path.stem)
    if not m:
        continue
    plant_id = int(m.group(1))
    df = pd.read_csv(path, encoding='utf-8-sig')
    df['plant_id'] = plant_id
    frames.append(df)

if not frames:
    raise SystemExit('No data files found.')

df = pd.concat(frames, ignore_index=True)
df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
df['day'] = pd.to_datetime(df['day'], errors='coerce')
df = df.dropna(subset=['day'])
df = df.sort_values(['plant_id', 'day']).reset_index(drop=True)

# Feature engineering
df['dayofweek'] = df['day'].dt.dayofweek
df['month'] = df['day'].dt.month
df['quarter'] = df['day'].dt.quarter
df['year'] = df['day'].dt.year
df['dayofyear'] = df['day'].dt.dayofyear
df['weekofyear'] = df['day'].dt.isocalendar().week.astype(int)
df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)

cal = USFederalHolidayCalendar()
holidays = cal.holidays(start=df['day'].min(), end=df['day'].max())
df['is_holiday'] = df['day'].isin(holidays).astype(int)

df['sin_dayofyear'] = np.sin(2 * np.pi * df['dayofyear'] / 365.0)
df['cos_dayofyear'] = np.cos(2 * np.pi * df['dayofyear'] / 365.0)

lags = [1, 2, 3, 7, 14]
for lag in lags:
    df[f'lag_{lag}'] = df.groupby('plant_id')['flights'].shift(lag)

roll_windows = [3, 7, 14, 28]
for window in roll_windows:
    shifted = df.groupby('plant_id')['flights'].shift(1)
    df[f'roll_mean_{window}'] = shifted.rolling(window=window, min_periods=2).mean()
    df[f'roll_std_{window}'] = shifted.rolling(window=window, min_periods=2).std()

for col in [f'roll_std_{w}' for w in roll_windows]:
    df[col] = df[col].fillna(0)

required_cols = [f'lag_{lag}' for lag in lags] + [f'roll_mean_{w}' for w in roll_windows]
df = df.dropna(subset=required_cols)

FEATURES = [
    'plant_id', 'dayofweek', 'month', 'quarter', 'year', 'dayofyear', 'weekofyear',
    'is_weekend', 'is_holiday', 'sin_dayofyear', 'cos_dayofyear'
] + [f'lag_{lag}' for lag in lags] + [f'roll_mean_{w}' for w in roll_windows] + [f'roll_std_{w}' for w in roll_windows]

TARGET = 'flights'

last_date = df['day'].max()
cutoff = last_date - pd.DateOffset(months=3)
train = df[df['day'] <= cutoff]
test = df[df['day'] > cutoff]

X_train = train[FEATURES]
y_train = train[TARGET]
X_test = test[FEATURES]
y_test = test[TARGET]

param_distributions = {
    'n_estimators': [300, 400, 500, 600],
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.05, 0.03, 0.07],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.3],
    'reg_lambda': [1, 2, 3]
}

model = xgboost = xgb.XGBRegressor(
    objective='reg:squarederror',
    tree_method='hist',
    random_state=42,
    n_jobs=1
)

from sklearn.metrics import make_scorer

ts_cv = TimeSeriesSplit(n_splits=5)
scorer = make_scorer(mean_absolute_percentage_error, greater_is_better=False)

search = RandomizedSearchCV(model, param_distributions=param_distributions, n_iter=20, scoring=scorer, cv=ts_cv, verbose=1, random_state=42, n_jobs=1)
search.fit(X_train, y_train)

best = search.best_estimator_
print('Best params:', search.best_params_)

pred = best.predict(X_test)
mape = mean_absolute_percentage_error(y_test, pred)
print('Test MAPE %:', mape * 100)

pred_train = best.predict(X_train)
print('Train MAPE %:', mean_absolute_percentage_error(y_train, pred_train) * 100)

best.fit(pd.concat([X_train, X_test]), pd.concat([y_train, y_test]))

import joblib
joblib.dump({'model': best, 'features': FEATURES, 'params': search.best_params_, 'test_mape': mape}, 'tmp_model.pkl')
print('Model saved to tmp_model.pkl')

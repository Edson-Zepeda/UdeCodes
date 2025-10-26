import pandas as pd
import re
from pathlib import Path

frames = []
for path in sorted(Path('data').glob('Plant*.csv')):
    m = re.search(r'Plant\s*([0-9]+)', path.stem)
    if not m:
        continue
    plant_id = int(m.group(1))
    df = pd.read_csv(path, encoding='utf-8-sig')
    df['plant_id'] = plant_id
    frames.append(df)

if not frames:
    raise SystemExit('No data found')

df = pd.concat(frames, ignore_index=True)
df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
df['day'] = pd.to_datetime(df['day'], errors='coerce')
df = df.dropna(subset=['day'])
df.sort_values(['plant_id', 'day'], inplace=True)

df['lag_1'] = df.groupby('plant_id')['flights'].shift(1)
df['delta'] = df['flights'] - df['lag_1']

print(df[df['plant_id'] == 2]['delta'].describe())

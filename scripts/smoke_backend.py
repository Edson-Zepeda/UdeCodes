from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

# Health
r = client.get('/health')
assert r.status_code == 200, r.text
print('/health OK:', r.json())

# Flights demo (always available)
r = client.get('/flights')
assert r.status_code == 200, r.text
print('/flights OK: count', len(r.json().get('flights', r.json())))

# Lots demo
r = client.get('/lots/recommend/demo', params={'flight_id': 'AM109'})
assert r.status_code == 200, r.text
print('/lots/recommend/demo OK: lots', len(r.json().get('lots', [])))

# Financial impact minimal body -> should not 503 even if model missing
payload = {
    'include_details': True,
    'max_details': 5,
    'flight_id': 'AM109',
    'origin': 'MEX'
}
r = client.post('/predict/financial-impact', json=payload)
assert r.status_code == 200, r.text
print('/predict/financial-impact OK:', r.json()['total_impact'])

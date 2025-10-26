from backend.app.main import recommend_lots
print('OK', recommend_lots(flight_id='AM109', origin='DOH').dict()['flight_id'])

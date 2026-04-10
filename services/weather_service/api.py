from fastapi import FastAPI, Query
from services.weather_service.service import get_weather

app = FastAPI()

@app.get("/weather")
def weather_endpoint(
    location: str, 
    period: str = Query("current", enum=["current", "forecast", "historical"]),
    dt: str = Query(None, description="Date in YYYY-MM-DD format for historical data")
):
    return get_weather(location, period, dt)
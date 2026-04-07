from fastapi import FastAPI
from service import get_weather

app = FastAPI()

@app.get("/weather")
def weather(location: str):
    return get_weather(location)
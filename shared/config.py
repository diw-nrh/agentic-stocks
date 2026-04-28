# shared/config.py
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    class AI:
        NOVITA_API_KEY = os.getenv("NOVITA_API_KEY")
        MODEL_NAME = "deepseek/deepseek-v3.2"
        BASE_URL = "https://api.novita.ai/openai"
    
    class Weather:
        URL = os.getenv("WEATHER_SERVICE_URL", "http://localhost:9001")
        OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
    
    class Stock:
        URL = os.getenv("STOCK_SERVICE_URL", "http://localhost:9002")
    
    class News: 
        URL = os.getenv("NEWS_SERVICE_URL", "http://localhost:9003")

    class Memory:
        URL = os.getenv("MEMORY_SERVICE_URL", "http://localhost:9004")
        DATABASE_URL = os.getenv("MEMORY_DB_URL", "postgresql://hermes:hermes_pass@localhost:5432/hermes_db")
        
    @classmethod
    def validate(cls):
        if not cls.AI.NOVITA_API_KEY:
            raise ValueError(" NOVITA_API_KEY in .env")
        if not cls.Weather.OPENWEATHER_API_KEY:
            raise ValueError(" OPENWEATHER_API_KEY in .env")
        
        print(f" Using model: {cls.AI.MODEL_NAME}")
Config.validate()
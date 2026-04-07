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
        URL = "http://localhost:9001"
    
    @classmethod
    def validate(cls):
        if not cls.AI.NOVITA_API_KEY:
            raise ValueError(" NOVITA_API_KEY in .env")
        
        print(f" Using model: {cls.AI.MODEL_NAME}")
Config.validate()
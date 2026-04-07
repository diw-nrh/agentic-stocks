import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Novita AI
    NOVITA_API_KEY = os.getenv("NOVITA_API_KEY")
    MODEL_NAME = "deepseek/deepseek-v3.2"
    BASE_URL = "https://api.novita.ai/openai"
    
    # MCP Server
    MCP_SERVER_URL = "http://localhost:9001"
    
    @classmethod
    def validate(cls):
        if not cls.NOVITA_API_KEY:
            raise ValueError(" .env: NOVITA_API_KEY not found")
        
        print(f"Using model: {cls.MODEL_NAME}")

Config.validate()
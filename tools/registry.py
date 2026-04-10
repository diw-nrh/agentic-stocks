import os
from langchain_core.tools import tool
from enum import Enum
import requests
from shared.utils import get_skill_description
from shared.config import Config

SKILLS_DIR = "skills"

def get_all_skills_prompt() -> str:
    all_prompts = []
    for skill_name in os.listdir(SKILLS_DIR):
        skill_path = os.path.join(SKILLS_DIR, skill_name, "SKILL.md")
        if os.path.exists(skill_path):
            description = get_skill_description(skill_path)
            all_prompts.append(description)
    return "\n\n".join(all_prompts)

class WeatherPeriod(str, Enum):
    CURRENT = "current"
    FORECAST = "forecast"
    HISTORICAL = "historical"

@tool
def get_weather(location: str, period: WeatherPeriod = WeatherPeriod.CURRENT, dt: str = None) -> str:
    """
    Get weather data for any country or city. 
    Supports current, 7-day forecast, and historical data.
    If period is 'historical', you can optionally specify 'dt' in YYYY-MM-DD format.
    """
    # Using Config.Weather.URL to support both local and docker environments
    url = f"{Config.Weather.URL}/weather?location={location}&period={period.value}"
    if dt:
        url += f"&dt={dt}"
    
    try:
        response = requests.get(url)
        return str(response.json())
    except Exception as e:
        return f"Error connecting to weather service: {e}"

TOOLS = [get_weather]
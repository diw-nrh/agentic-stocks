from importlib.resources import path
import os
from langchain_core.tools import tool
from enum import Enum
import requests
from shared.utils import get_skill_description
from shared.config import Config

SKILLS_DIR = "skills"

def get_select_skills_prompt(path: str) -> str:
    all_prompts = []
    skill_path = os.path.join(path, "SKILL.md")
    if os.path.exists(skill_path):
        description = get_skill_description(skill_path)
        all_prompts.append(description)
    return "\n\n".join(all_prompts)

class WeatherPeriod(str, Enum):
    CURRENT = "current"
    FORECAST = "forecast"
    HISTORICAL = "historical"

class StockPeriod(str, Enum):
    CURRENT = "current"
    SCHEDULED = "scheduled"
    RANGE = "range"

@tool
def weather(location: str, period: WeatherPeriod = WeatherPeriod.CURRENT, dt: str = None) -> str:
    """
    Get weather data for any country or city. 
    Supports current, 7-day forecast, and historical data.
    If period is 'historical', you can optionally specify 'dt' in YYYY-MM-DD format.
    """

    url = f"{Config.Weather.URL}/weather?location={location}&period={period.value}"
    if dt:
        url += f"&dt={dt}"
    
    try:
        response = requests.get(url)
        return str(response.json())
    except Exception as e:
        return f"Error connecting to weather service: {e}"

@tool
def stock(symbols: str, period: StockPeriod = StockPeriod.CURRENT, start: str = None, end: str = None) -> str:
    """
    Get stock data for given symbols (e.g. AAPL, MSFT).
    Supports current data, scheduled (end date), or range (start and end date).
    """
    url = f"{Config.Stock.URL}/stocks?symbols={symbols}&period={period.value}"
    if start: url += f"&start={start}"
    if end: url += f"&end={end}"
    
    try:
        response = requests.get(url)
        return str(response.json())
    except Exception as e:
        return f"Error connecting to stock service: {e}"

TOOLS_WEATHER = [weather]
TOOLS_STOCK = [stock]
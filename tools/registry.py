from importlib.resources import path
import os
from langchain_core.tools import tool
from enum import Enum
from shared.utils import get_skill_description
from shared.config import Config
from tools.mcp.weather_tool import weather_tool as mcp_weather
from tools.mcp.stock_tool import stock_tool as mcp_stock
from tools.mcp.news_tool import news_tool as mcp_news

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

class NewsPeriod(str, Enum):
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

    try:
        result = mcp_weather(location=location, period=period.value, dt=dt)
        return str(result)
    except Exception as e:
        return f"Error connecting to weather MCP: {e}"

@tool
def stock(symbols: str, period: StockPeriod = StockPeriod.CURRENT, start: str = None, end: str = None) -> str:
    """
    Get stock data for given symbols (e.g. AAPL, MSFT).
    Supports current data, scheduled (end date), or range (start and end date).
    """
    try:
        result = mcp_stock(symbols=symbols, period=period.value, start=start, end=end)
        return str(result)
    except Exception as e:
        return f"Error connecting to stock MCP: {e}"

@tool
def news(news: str, location: str = "thailand", start: str = None, end: str = None, period: NewsPeriod = NewsPeriod.CURRENT, day: str = "30") -> str:
    """
    Get news data for given news (e.g. weather, stock).
    Supports current data, scheduled (end date), or range (start and end date).
    """
    try:
        result = mcp_news(news=news, location=location, start=start, end=end, period=period.value, day=day)
        return str(result)
    except Exception as e:
        return f"Error connecting to news MCP: {e}"

TOOLS_WEATHER = weather
TOOLS_STOCK = stock
TOOLS_NEWS = news
tools = [TOOLS_WEATHER, TOOLS_STOCK, TOOLS_NEWS]
from langchain_core.tools import tool
from tools.mcp.weather_tool import weather_tool

@tool
def get_weather(location: str) -> str:
    """Get weather by location (e.g., 'Phuket')"""
    return weather_tool(location)

TOOLS = [get_weather]
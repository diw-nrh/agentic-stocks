import asyncio
from .client import call_mcp

def weather_tool(location: str):
    return asyncio.run(
        call_mcp("weather", {"location": location})
    )
import asyncio
from .client import call_mcp

def weather_tool(location: str, period: str = "current", dt: str = None):
    params = {"location": location, "period": period}
    if dt:
        params["dt"] = dt
    return asyncio.run(
        call_mcp("weather", params)
    )
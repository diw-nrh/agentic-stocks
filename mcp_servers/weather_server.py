from mcp.server.fastmcp import FastMCP
import httpx
from shared.config import Config

mcp = FastMCP("weather-server")

@mcp.tool()
async def weather(location: str) -> str:
    async with httpx.AsyncClient() as client:
        endpoint = f"{Config.MCP_SERVER_WEATHER_URL}/weather"
        r = await client.get(endpoint, params={"location": location})

    data = r.json()
    return f"{data['location']} {data['temp']}°C {data['desc']}"

if __name__ == "__main__":
    mcp.run(transport="sse")
from mcp.server.fastmcp import FastMCP
import httpx
from shared.config import Config

mcp = FastMCP("weather-server", host="0.0.0.0", port=8000)

@mcp.tool()
async def weather(location: str, period: str = "current", dt: str = None) -> str:
    async with httpx.AsyncClient() as client:
        endpoint = f"{Config.Weather.URL}/weather"
        params = {"location": location, "period": period}
        if dt:
            params["dt"] = dt
        r = await client.get(endpoint, params=params)

    data = r.json()
    
    if "error" in data:
        return f"Error: {data['error']}"

    display_name = data.get('display_name', 'Unknown')
    temp = data.get('temp', 'N/A')
    condition = data.get('condition', 'N/A')
    
    return f"{display_name} {temp}°C {condition}"

if __name__ == "__main__":
    mcp.run(transport="sse")
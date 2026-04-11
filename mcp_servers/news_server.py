from mcp.server.fastmcp import FastMCP
import httpx
from shared.config import Config

mcp = FastMCP("news-server", host="0.0.0.0", port=8003)

@mcp.tool()
async def news(news: str, location: str = "thailand", start: str = None, end: str = None, period: str = "current", day: str = "30") -> str:
    async with httpx.AsyncClient() as client:
        endpoint = f"{Config.News.URL}/news"
        params = {"news": news, "location": location, "start": start, "end": end, "period": period, "day": day}
        params = {k: v for k, v in params.items() if v is not None}
        r = await client.get(endpoint, params=params)

    data = r.json()
    
    if isinstance(data, dict) and "error" in data:
        return f"Error fetching {news}: {data['error']}"

    return str({"news": news, "data": data})

if __name__ == "__main__":
    mcp.run(transport="sse")
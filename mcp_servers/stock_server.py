from mcp.server.fastmcp import FastMCP
import httpx
from shared.config import Config

mcp = FastMCP("stock-server", host="0.0.0.0", port=8001)

@mcp.tool()
async def stock(symbols: str, period: str = "current", start: str = None, end: str = None) -> str:
    async with httpx.AsyncClient() as client:
        endpoint = f"{Config.Stock.URL}/stocks"
        params = {"symbols": symbols, "period": period}
        if start:
            params["start"] = start
        if end:
            params["end"] = end
        r = await client.get(endpoint, params=params)

    data = r.json()
    
    if isinstance(data, dict) and "error" in data:
        return f"Error fetching {symbols}: {data['error']}"

    return str({"ticker": symbols, "data": data})  # ✅ ติด label ทันที

if __name__ == "__main__":
    mcp.run(transport="sse")
from mcp.server.fastmcp import FastMCP
import httpx
from shared.config import Config

mcp = FastMCP("stock-server")

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
    
    # Check if the response is a dictionary containing an error
    if isinstance(data, dict) and "error" in data:
        return f"Error: {data['error']}"

    # Return the stock data as a string (since it's a list of dictionaries)
    return str(data)

if __name__ == "__main__":
    mcp.run(transport="sse")
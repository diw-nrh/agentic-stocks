import asyncio
from .client import call_mcp

def stock_tool(symbols: str, period: str = "current", start: str = None, end: str = None):
    params = {"symbols": symbols, "period": period}
    if start:
        params["start"] = start
    if end:
        params["end"] = end
    return asyncio.run(
        call_mcp("stock", params, server_url="http://localhost:8001/sse")
    )

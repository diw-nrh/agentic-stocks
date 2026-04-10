import asyncio
from .client import call_mcp

<<<<<<< HEAD
def stock_tool(symbols: str, period: str = "current", start: str = None, end: str = None):
    params = {"symbols": symbols, "period": period}
    if start:
        params["start"] = start
    if end:
        params["end"] = end
    return asyncio.run(
        call_mcp("stock", params, server_url="http://localhost:8001/sse")
    )
=======
def stock_tool(ticker: str, period: str = "current", dt: str = None):
    params = {"ticker": ticker, "period": period}
    if dt:
        params["dt"] = dt
    return asyncio.run(
        call_mcp("stock", params)
    )
>>>>>>> 3c00455ebbad5109b0f30cad101378c2dc65ef06

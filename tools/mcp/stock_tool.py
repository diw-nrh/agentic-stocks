import asyncio
from .client import call_mcp

def stock_tool(ticker: str, period: str = "current", dt: str = None):
    params = {"ticker": ticker, "period": period}
    if dt:
        params["dt"] = dt
    return asyncio.run(
        call_mcp("stock", params)
    )
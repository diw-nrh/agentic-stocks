import asyncio
from .client import call_mcp

def news_tool(news: str, location: str = "thailand", start: str = None, end: str = None, period: str = "current", day: str = "30") -> str:
    params = {"news": news, "location": location, "period": period, "day": day}
    if start:
        params["start"] = start
    if end:
        params["end"] = end
    return asyncio.run(
        call_mcp("news", params, server_url="http://localhost:8003/sse")
    )
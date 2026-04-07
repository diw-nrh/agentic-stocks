from mcp import ClientSession
from mcp.client.sse import sse_client

async def call_mcp(tool: str, args: dict):
    async with sse_client("http://localhost:8000/sse") as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool, args)
            return result
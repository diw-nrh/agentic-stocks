from mcp import ClientSession
from mcp.client.sse import sse_client

async def call_mcp(tool: str, args: dict, server_url: str = "http://localhost:8000/sse"):
    async with sse_client(server_url) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool, args)
            return result
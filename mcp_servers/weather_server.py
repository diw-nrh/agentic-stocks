from mcp.server.fastmcp import FastMCP
import httpx

mcp = FastMCP("weather-server")

@mcp.tool()
async def weather(location: str) -> str:
    async with httpx.AsyncClient() as client:
        r = await client.get(
            "http://localhost:9001/weather",
            params={"location": location}
        )

    data = r.json()
    return f"{data['location']} {data['temp']}°C {data['desc']}"

if __name__ == "__main__":
    mcp.run(transport="sse")
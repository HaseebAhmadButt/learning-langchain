from typing import List
from mcp.server.fastmcp import FastMCP

mcp = FastMCP(name="Weather")

@mcp.tool()
async def get_weather(location: str) -> str:
    """Get the weather in a given location."""
    return "Hot as hell"

if __name__ == "__main__":
    mcp.run(transport="sse")
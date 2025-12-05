from rnow.core.tool import tool


@tool
async def internet_search(query: str) -> dict:
    """Search the web and return up to 5 results (title, link, snippet)."""
    pass

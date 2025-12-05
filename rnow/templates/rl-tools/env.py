import os

import requests

from rnow.core.tool import tool


@tool
async def internet_search(query: str) -> dict:
    """Search the web using Tavily and return up to 5 results (title, link, snippet)."""
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return {"results": [], "error": "TAVILY_API_KEY environment variable not set"}

    try:
        resp = requests.post(
            "https://api.tavily.com/search",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "query": query,
                "max_results": 5,
            },
            timeout=10,
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        return {"results": [], "error": str(e)}

    data = resp.json()
    results = []

    for item in data.get("results", []):
        results.append(
            {
                "title": item.get("title", ""),
                "link": item.get("url", ""),
                "snippet": item.get("content", "")[:200],
            }
        )

    return results

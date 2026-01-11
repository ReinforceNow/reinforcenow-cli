import requests
from bs4 import BeautifulSoup

from rnow.core.tool import tool


@tool
def internet_search(query: str) -> dict:
    """Search the web and return up to 5 results (title, link, snippet)."""
    try:
        resp = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "format": "json",
                "srlimit": 5,
            },
            headers={"User-Agent": "ReinforceNow/1.0 (training platform)"},
            timeout=10,
        )
        resp.raise_for_status()
    except requests.RequestException:
        return []
    data = resp.json()
    results = []
    for item in data.get("query", {}).get("search", []):
        snippet = BeautifulSoup(item.get("snippet", ""), "html.parser").get_text()
        title = item.get("title", "")
        results.append(
            {
                "title": title,
                "link": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                "snippet": snippet[:200],
            }
        )
    return results

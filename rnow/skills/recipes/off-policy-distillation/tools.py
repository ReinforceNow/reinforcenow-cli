"""Web browsing tool using Crawl4AI."""

import asyncio

import nest_asyncio
from crawl4ai import AsyncWebCrawler

from rnow.core.tool import tool

nest_asyncio.apply()


@tool
def browse(url: str) -> str:
    """Browse a URL and return its content as markdown."""

    async def fetch():
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)
            return result.markdown[:8000] if result.markdown else ""

    return asyncio.run(fetch())

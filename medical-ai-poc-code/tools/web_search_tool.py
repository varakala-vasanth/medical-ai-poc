# tools/web_search_tool.py

import os
from typing import List
from langchain.tools import tool

try:
    from tavily import TavilyClient
except ImportError:
    TavilyClient = None


@tool("web_search")
def web_search(query: str) -> str:
    """
    Search the web for up-to-date medical / nephrology information.

    Uses Tavily if TAVILY_API_KEY is set; otherwise returns a helpful message.
    """
    api_key = os.getenv("TAVILY_API_KEY")

    if not TavilyClient or not api_key:
        return (
            "Web search is not configured (missing tavily or TAVILY_API_KEY). "
            f"User query was: '{query}'. You must rely on internal knowledge instead."
        )

    client = TavilyClient(api_key=api_key)

    try:
        resp = client.search(query=query, max_results=3)
    except Exception as e:
        return f"Web search failed with error: {e}. Query was: '{query}'."

    # Tavily returns a list of results with 'content' and 'url' typically.
    results: List[dict] = resp.get("results", []) if isinstance(resp, dict) else resp

    if not results:
        return f"No web results found for: '{query}'."

    chunks = []
    for i, r in enumerate(results, start=1):
        title = r.get("title", f"Result {i}")
        url = r.get("url", "no-url")
        content = r.get("content", "")[:400]
        chunks.append(f"[{i}] {title}\nURL: {url}\n{content}")

    return "Web search results:\n\n" + "\n\n".join(chunks)

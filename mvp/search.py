"""Web search tool: DuckDuckGo (free, no key) + Exa AI (optional key).

Fallbacks keep the tool working with zero configuration.
"""

import os
from pathlib import Path


def _load_env():
    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip().strip('"'))


def search_web(query: str, top_k: int = 5) -> list[dict]:
    """Return list of {title, url, snippet}. Tries DDG first, then Exa."""
    _load_env()
    results = _search_ddg(query, top_k)
    if not results:
        results = _search_exa(query, top_k)
    return results


def _search_ddg(query: str, top_k: int) -> list[dict]:
    try:
        from ddgs import DDGS
    except ImportError:
        return _search_ddg_instant(query, top_k)

    try:
        with DDGS() as ddgs:
            items = list(ddgs.text(query, max_results=top_k))
        return [
            {
                "title": it.get("title", ""),
                "url": it.get("href", ""),
                "snippet": it.get("body", ""),
            }
            for it in items
        ]
    except Exception:
        return []


def _search_ddg_instant(query: str, top_k: int) -> list[dict]:
    """Zero-dependency fallback: DuckDuckGo Instant Answer API."""
    import requests

    try:
        r = requests.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
    except Exception:
        return []
    results = []
    for topic in data.get("RelatedTopics", [])[:top_k]:
        if "Topics" in topic:
            for sub in topic["Topics"][:top_k]:
                results.append(
                    {
                        "title": sub.get("Text", "")[:80],
                        "url": sub.get("FirstURL", ""),
                        "snippet": sub.get("Text", ""),
                    }
                )
        elif topic.get("Text"):
            results.append(
                {
                    "title": topic.get("Text", "")[:80],
                    "url": topic.get("FirstURL", ""),
                    "snippet": topic.get("Text", ""),
                }
            )
    return results


def _search_exa(query: str, top_k: int) -> list[dict]:
    api_key = os.environ.get("EXA_API_KEY")
    if not api_key:
        return []
    import requests

    try:
        r = requests.post(
            "https://api.exa.ai/search",
            headers={"x-api-key": api_key, "Content-Type": "application/json"},
            json={"query": query, "numResults": top_k},
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
    except Exception:
        return []
    return [
        {
            "title": it.get("title", ""),
            "url": it.get("url", ""),
            "snippet": it.get("text", "")[:300],
        }
        for it in data.get("results", [])
    ]


def build_search_context(results: list[dict]) -> str:
    if not results:
        return ""
    parts = ["WEB SEARCH RESULTS (use these to answer; cite the URL):"]
    for i, res in enumerate(results, 1):
        parts.append(
            f"{i}. {res['title']} — {res['url']}\n   {res['snippet'][:250]}"
        )
    return "\n".join(parts)

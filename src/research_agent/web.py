"""Minimal injectable web search adapter for the web fallback path."""

from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser

import httpx

from research_agent.env import EnvSettings, load_env

try:
    from tavily import TavilyClient
except ImportError:  # pragma: no cover - dependency is installed in production
    TavilyClient = None


@dataclass(frozen=True)
class WebResult:
    title: str
    url: str
    snippet: str = ""


class TavilySearcher:
    """Tavily-backed search adapter with the same result contract as WebSearcher."""

    def __init__(self, api_key: str | None = None, client=None) -> None:
        values = load_env()
        key = api_key or values.get("TAVILY_API_KEY")
        if client is None and (not key or TavilyClient is None):
            raise ValueError("TAVILY_API_KEY is required for Tavily search")
        self.client = client or TavilyClient(api_key=key)

    def search(self, query: str, limit: int = 5) -> list[WebResult]:
        if not query.strip():
            raise ValueError("query must not be empty")
        response = self.client.search(query=query, max_results=limit)
        return [
            WebResult(
                title=str(item.get("title", "")),
                url=str(item.get("url", "")),
                snippet=str(item.get("content", "")),
            )
            for item in response.get("results", [])
        ]


def make_tavily_search(api_key: str | None = None):
    """Return the plan's callable ``(query, max_results)`` Tavily adapter."""

    searcher = TavilySearcher(api_key)
    return searcher.search


class WebSearcher:
    """HTTP search adapter; callers may inject a fake client in tests."""

    def __init__(
        self,
        cfg: EnvSettings | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self.cfg = cfg or load_env()
        self.client = client or httpx.Client(timeout=20.0, follow_redirects=True)

    def search(self, query: str, limit: int = 5) -> list[WebResult]:
        if not query.strip():
            raise ValueError("query must not be empty")
        if limit < 1:
            return []
        response = self.client.get(
            self.cfg.web_search_url,
            params={"q": query, "num": limit},
            headers={"User-Agent": "research-agent/1.0"},
        )
        response.raise_for_status()
        parser = _DuckParser(limit)
        parser.feed(response.text)
        return parser.results


class _DuckParser(HTMLParser):
    """Tiny dependency-free parser for DuckDuckGo's stable result classes."""

    def __init__(self, limit: int) -> None:
        super().__init__()
        self.limit = limit
        self.results: list[WebResult] = []
        self._link: str | None = None
        self._title: list[str] = []
        self._snippet: list[str] = []
        self._in_title = False
        self._in_snippet = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        classes = dict(attrs).get("class", "") or ""
        values = dict(attrs)
        if tag == "a" and "result__a" in classes:
            self._link = values.get("href")
            self._title = []
            self._in_title = True
        elif "result__snippet" in classes:
            self._snippet = []
            self._in_snippet = True

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self._title.append(data)
        if self._in_snippet:
            self._snippet.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._in_title:
            if self._link:
                self.results.append(WebResult(" ".join("".join(self._title).split()), self._link))
            self._in_title = False
            self._link = None
        if self._in_snippet:
            if self.results:
                previous = self.results[-1]
                self.results[-1] = WebResult(
                    previous.title, previous.url, " ".join("".join(self._snippet).split())
                )
            self._in_snippet = False
            if len(self.results) >= self.limit:
                self.close()


def search_web(query: str, limit: int = 5, searcher: WebSearcher | None = None) -> list[WebResult]:
    """Convenience function for one-shot searches."""

    return (searcher or WebSearcher()).search(query, limit)

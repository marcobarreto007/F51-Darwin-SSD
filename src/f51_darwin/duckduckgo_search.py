"""Small, defensive DuckDuckGo search adapter.

Search results are untrusted evidence.  This module only returns bounded
metadata/snippets; it never promotes web text into the training corpus.
"""

from __future__ import annotations

from html import unescape
from typing import Any, Callable, Iterable
from urllib.parse import parse_qs, unquote, urlparse


SearchBackend = Callable[[str, int], Iterable[dict[str, Any]]]


def _clean_text(value: object, *, limit: int) -> str:
    text = " ".join(unescape(str(value or "")).split())
    return text[:limit]


def _normalize_result_url(value: object) -> str:
    raw = unescape(str(value or "").strip())
    if raw.startswith("//"):
        raw = "https:" + raw
    parsed = urlparse(raw)
    if parsed.netloc.endswith("duckduckgo.com") and parsed.path.startswith("/l/"):
        target = parse_qs(parsed.query).get("uddg", [""])[0]
        raw = unquote(target)
        parsed = urlparse(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ""
    if parsed.username or parsed.password:
        return ""
    return raw[:2048]


def _default_backend(query: str, max_results: int) -> Iterable[dict[str, Any]]:
    """Use either maintained DDGS package name without making it mandatory."""
    try:
        from ddgs import DDGS  # type: ignore
    except ImportError:
        try:
            from duckduckgo_search import DDGS  # type: ignore
        except ImportError as exc:
            raise RuntimeError("install 'ddgs' or 'duckduckgo-search' for web search") from exc

    with DDGS() as client:
        # The automatic metasearch backend has been observably noisy for
        # technical queries.  Brave + Bing is deterministic enough for the
        # local runtime while still retaining two independent indexes.
        try:
            yield from client.text(
                query,
                max_results=max_results,
                backend="brave,bing",
            )
        except TypeError:
            # Compatibility with older ``duckduckgo-search`` releases.
            yield from client.text(query, max_results=max_results)


def search_web(
    query: str,
    max_results: int = 3,
    *,
    backend: SearchBackend | None = None,
) -> list[dict[str, str]]:
    """Return validated title/snippet/URL dictionaries.

    Backend failures are represented by an empty list.  In particular, an
    error sentinel is never considered a successful search result.
    """
    normalized_query = " ".join(query.split())[:500]
    if not normalized_query:
        return []
    limit = max(1, min(int(max_results), 10))
    provider = backend or _default_backend
    try:
        raw_results = provider(normalized_query, limit)
        results: list[dict[str, str]] = []
        seen: set[str] = set()
        for raw in raw_results:
            if not isinstance(raw, dict) or raw.get("error"):
                continue
            href = _normalize_result_url(raw.get("href") or raw.get("url"))
            title = _clean_text(raw.get("title"), limit=300)
            body = _clean_text(raw.get("body") or raw.get("snippet"), limit=1200)
            if not href or not (title or body) or href in seen:
                continue
            seen.add(href)
            results.append({"title": title, "body": body, "href": href})
            if len(results) >= limit:
                break
        return results
    except Exception:
        return []


def extract_text_from_results(results: list[dict], max_chars: int = 2000) -> str:
    """Build a bounded evidence block from validated search snippets."""
    limit = max(0, int(max_chars))
    if limit == 0:
        return ""
    chunks: list[str] = []
    used = 0
    for result in results:
        if not isinstance(result, dict) or result.get("error"):
            continue
        href = _normalize_result_url(result.get("href"))
        if not href:
            continue
        snippet = _clean_text(
            f"{result.get('title', '')}. {result.get('body', '')}", limit=1500
        ).strip(" .")
        if not snippet:
            continue
        separator = "\n\n" if chunks else ""
        remaining = limit - used - len(separator)
        if remaining <= 0:
            break
        chunks.append(separator + snippet[:remaining])
        used += len(separator) + min(len(snippet), remaining)
        if used >= limit:
            break
    return "".join(chunks)


__all__ = ["extract_text_from_results", "search_web"]

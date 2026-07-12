from dataclasses import dataclass
from typing import Literal, Optional

import anthropic
import trafilatura

MIN_TEXT_LENGTH = 250  # below this, it's almost always a cookie-wall/paywall stub, not an article


@dataclass
class ArticleFetchResult:
    url: str
    text: Optional[str]
    title: Optional[str]
    method: Literal["trafilatura", "claude_web_fetch", "failed"]
    error: Optional[str] = None


def _fetch_with_trafilatura(url: str) -> Optional[dict]:
    try:
        downloaded = trafilatura.fetch_url(url)
    except Exception:
        return None
    if not downloaded:
        return None
    try:
        return trafilatura.bare_extraction(downloaded, as_dict=True, with_metadata=True, url=url)
    except Exception:
        return None


def _fetch_with_claude(url: str, model: str = "claude-sonnet-5") -> ArticleFetchResult:
    client = anthropic.Anthropic()
    try:
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            tools=[{"type": "web_fetch_20260209", "name": "web_fetch"}],
            messages=[{"role": "user", "content": f"Fetch {url}"}],
        )
    except Exception as e:
        return ArticleFetchResult(url=url, text=None, title=None, method="failed", error=str(e))

    for block in response.content:
        if block.type != "web_fetch_tool_result":
            continue
        result = block.content
        if getattr(result, "type", None) == "web_fetch_tool_result_error":
            return ArticleFetchResult(
                url=url, text=None, title=None, method="failed",
                error=f"web_fetch error: {result.error_code}",
            )
        doc = result.content
        return ArticleFetchResult(url=url, text=doc.source.data, title=doc.title, method="claude_web_fetch")

    return ArticleFetchResult(url=url, text=None, title=None, method="failed", error="No web_fetch result in response")


def fetch_article(url: str) -> ArticleFetchResult:
    """
    Fetch a URL and extract clean article text. Tries trafilatura first (free,
    no LLM call); falls back to Claude's server-side web_fetch tool for
    paywalled/JS-heavy/bot-blocked pages trafilatura can't reach. Never raises --
    a "failed" result lets callers show a clear error instead of silently
    vectorizing empty text.
    """
    result = _fetch_with_trafilatura(url)
    text = (result or {}).get("text") or ""
    if len(text) >= MIN_TEXT_LENGTH:
        return ArticleFetchResult(
            url=url, text=text, title=(result or {}).get("title"), method="trafilatura",
        )

    return _fetch_with_claude(url)

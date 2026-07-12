import json
from dataclasses import dataclass, field
from typing import List, Literal, Optional

import anthropic

VERDICTS = ["corroborated", "contradicted", "unverifiable", "mixed"]
STANCES = ["supports", "contradicts", "unrelated"]

SYSTEM_PROMPT = (
    "You are an independent fact-corroboration assistant for a news article. "
    "A separate machine learning classifier already produces a real/fake prediction for this "
    "article -- that is not your job. Your ONLY job is to identify 2-4 central, checkable "
    "factual claims in the article, search the live web for independent sources on those claims, "
    "and report whether they are corroborated, contradicted, unverifiable, or give a mixed signal. "
    "Never state or imply whether the article overall is real or fake -- only report what "
    "independent sources say about specific claims. "
    "Never fabricate a citation: every source in your output must be a URL that actually appeared "
    "in a web_search result during this conversation, not a URL recalled from training data. "
    "Exclude the article's own domain and any outlet that is clearly syndicating/reprinting the "
    "same article from counting as independent corroboration."
)

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": VERDICTS},
        "rationale": {"type": "string"},
        "sources": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "title": {"type": "string"},
                    "stance": {"type": "string", "enum": STANCES},
                    "note": {"type": "string"},
                },
                "required": ["url", "title", "stance", "note"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["verdict", "rationale", "sources"],
    "additionalProperties": False,
}


@dataclass
class CorroborationSource:
    url: str
    title: str
    stance: Literal["supports", "contradicts", "unrelated"]
    note: str


@dataclass
class CorroborationResult:
    verdict: Literal["corroborated", "contradicted", "unverifiable", "mixed"]
    rationale: str
    sources: List[CorroborationSource]
    raw_search_queries: List[str] = field(default_factory=list)


def _collect_search_urls_and_queries(content_blocks):
    seen_urls = set()
    queries = []
    for block in content_blocks:
        if block.type == "server_tool_use" and block.name == "web_search":
            q = block.input.get("query")
            if q:
                queries.append(q)
        elif block.type == "web_search_tool_result" and isinstance(block.content, list):
            for item in block.content:
                url = getattr(item, "url", None)
                if url:
                    seen_urls.add(url)
    return seen_urls, queries


def corroborate_claims(
    article_text: str,
    article_url: Optional[str] = None,
    model: str = "claude-sonnet-5",
    max_search_uses: int = 6,
) -> CorroborationResult:
    """
    Independent web-search-based fact corroboration for a news article.
    Does NOT classify the article as real/fake -- that is the sklearn model's
    job, exclusively. This only reports whether independent sources on the
    live web corroborate, contradict, are silent on, or give a mixed signal
    on the article's central claims, with citations.
    """
    client = anthropic.Anthropic()

    user_parts = [f"Article text (truncated to 3000 chars):\n{article_text[:3000]}"]
    if article_url:
        user_parts.append(f"\nSource URL: {article_url}")
    user_content = "\n".join(user_parts)

    tools = [{"type": "web_search_20260209", "name": "web_search", "max_uses": max_search_uses}]
    messages = [{"role": "user", "content": user_content}]
    output_config = {"format": {"type": "json_schema", "schema": RESPONSE_SCHEMA}}

    response = client.messages.create(
        model=model,
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        tools=tools,
        messages=messages,
        output_config=output_config,
    )

    # Server-tool loop can pause if it hits the iteration cap; resume once, unchanged.
    if response.stop_reason == "pause_turn":
        messages = messages + [{"role": "assistant", "content": response.content}]
        response = client.messages.create(
            model=model,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            tools=tools,
            messages=messages,
            output_config=output_config,
        )

    seen_urls, queries = _collect_search_urls_and_queries(response.content)

    text_block = next((b for b in response.content if b.type == "text"), None)
    if text_block is None:
        return CorroborationResult(
            verdict="unverifiable",
            rationale="The corroboration agent did not return a structured result.",
            sources=[],
            raw_search_queries=queries,
        )

    data = json.loads(text_block.text)

    # Defensive anti-hallucination check: only keep sources that actually
    # appeared in a web_search_tool_result during this conversation -- the
    # same failure mode narrate_shap hit once already, guarded against here too.
    sources = [
        CorroborationSource(
            url=s.get("url", ""),
            title=s.get("title", ""),
            stance=s.get("stance", "unrelated"),
            note=s.get("note", ""),
        )
        for s in data.get("sources", [])
        if s.get("url", "") in seen_urls
    ]

    return CorroborationResult(
        verdict=data.get("verdict", "unverifiable"),
        rationale=data.get("rationale", ""),
        sources=sources,
        raw_search_queries=queries,
    )

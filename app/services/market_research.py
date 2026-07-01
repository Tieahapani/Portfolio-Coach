import json
import asyncio
from cachetools import TTLCache
from openai import AsyncOpenAI
from app.config import get_settings
from app.schemas import MarketData, JobPosting

# Cache market research per role for 6 hours (max 30 roles)
_market_cache: TTLCache = TTLCache(maxsize=30, ttl=6 * 60 * 60)


MARKET_PROMPT = """Search for current "{role}" job postings. Focus on internship and entry-level roles in 2025-2026.

Respond with ONLY raw JSON (no markdown, no backticks, no explanation):
{{
  "market_skills": ["12-15 in-demand technical skills"],
  "trending_tools": ["6-8 trending tools/frameworks"],
  "sample_jobs": [{{"title":"...","company":"...","key_skills":["...","..."]}}],
  "industry_trends": "2 sentence summary of current hiring trends"
}}"""


def _parse_json(text: str) -> dict | None:
    """Extract JSON from LLM response text."""
    cleaned = text.replace("```json", "").replace("```", "").strip()
    start = cleaned.find("{")
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(cleaned[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(cleaned[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


async def search_openai(role: str) -> dict | None:
    """Use GPT-4o-mini for market research."""
    settings = get_settings()
    client = AsyncOpenAI(api_key=settings.openai_api_key)

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": MARKET_PROMPT.format(role=role)}],
        temperature=0.7,
        max_tokens=3000,
    )

    text = response.choices[0].message.content or ""
    return _parse_json(text)


async def research_market(role: str, mode: str = "fast") -> MarketData:
    """Market research using GPT-4o-mini (cached 6 hours per role)."""
    cache_key = role.lower().strip()
    if cache_key in _market_cache:
        return _market_cache[cache_key]

    try:
        data = await search_openai(role)
    except Exception as e:
        print(f"Market research failed: {e}")
        data = None

    if not data:
        return MarketData(sources=["fallback"])

    result = MarketData(
        market_skills=data.get("market_skills", []),
        trending_tools=data.get("trending_tools", []),
        industry_trends=data.get("industry_trends", ""),
        sample_jobs=[JobPosting(**j) for j in data.get("sample_jobs", [])],
        sources=["GPT-4o-mini"],
    )
    _market_cache[cache_key] = result
    return result

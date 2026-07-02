import json
import asyncio
from cachetools import TTLCache
from google import genai
from app.config import get_settings
from app.schemas import MarketData, JobPosting

# Cache market research per role for 6 hours (max 30 roles)
_market_cache: TTLCache = TTLCache(maxsize=30, ttl=6 * 60 * 60)


MARKET_PROMPT = """Search for current "{role}" job postings. Focus on internship and entry-level roles.

Find real, current job listings and extract the skills they require.

Respond with ONLY raw JSON (no markdown, no backticks, no explanation):
{{
  "market_skills": ["12-15 in-demand technical skills from real job postings"],
  "trending_tools": ["6-8 trending tools/frameworks you found mentioned"],
  "sample_jobs": [{{"title":"...","company":"...","key_skills":["...","..."]}}],
  "industry_trends": "2 sentence summary of current hiring trends based on what you found"
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


async def search_gemini(role: str) -> dict | None:
    """Use Gemini 2.5 Flash with Google Search grounding for market research."""
    settings = get_settings()
    client = genai.Client(api_key=settings.effective_gemini_key)

    response = await asyncio.to_thread(
        client.models.generate_content,
        model="gemini-2.5-flash",
        contents=MARKET_PROMPT.format(role=role),
        config={"tools": [{"google_search": {}}]},
    )

    text = response.text or ""
    return _parse_json(text)


async def research_market(role: str, mode: str = "fast") -> MarketData:
    """Market research using Gemini 2.5 Flash with Google Search (cached 6 hours)."""
    cache_key = role.lower().strip()
    if cache_key in _market_cache:
        return _market_cache[cache_key]

    try:
        data = await search_gemini(role)
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
        sources=["Gemini 2.5 Flash (Google Search)"],
    )
    _market_cache[cache_key] = result
    return result
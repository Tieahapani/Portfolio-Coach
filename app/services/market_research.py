import asyncio
from cachetools import TTLCache
from google import genai
from pydantic import ValidationError
from app.config import get_settings
from app.schemas import MarketData, MarketOutput
from app.services.llm_loop import _parse_json

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


MAX_ATTEMPTS = 2  # search-grounded calls are slower, so fewer retries


async def search_gemini(role: str, extra_instruction: str = "") -> MarketOutput | None:
    """One search-grounded Gemini call, schema-validated. None if invalid."""
    settings = get_settings()
    client = genai.Client(api_key=settings.effective_gemini_key)

    response = await asyncio.to_thread(
        client.models.generate_content,
        model="gemini-2.5-flash",
        contents=MARKET_PROMPT.format(role=role) + extra_instruction,
        config={"tools": [{"google_search": {}}]},
    )

    raw = _parse_json(response.text or "")
    if raw is None:
        print("[loop:market] unparseable JSON")
        return None
    try:
        return MarketOutput.model_validate(raw)
    except ValidationError as e:
        errors = "; ".join(
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}"
            for err in e.errors()[:5]
        )
        print(f"[loop:market] schema failed: {errors}")
        return None


async def research_market(role: str, mode: str = "fast") -> MarketData:
    """Market research via a validated retry loop (cached 6 hours)."""
    cache_key = role.lower().strip()
    if cache_key in _market_cache:
        return _market_cache[cache_key]

    data = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        extra = "" if attempt == 1 else (
            "\n\nIMPORTANT: your previous answer was invalid or incomplete. "
            "Return ONLY raw JSON with at least 12 market_skills, 6 trending_tools, "
            "and a non-empty industry_trends summary."
        )
        try:
            data = await search_gemini(role, extra)
        except Exception as e:
            print(f"[loop:market] attempt {attempt} failed: {e}")
        if data:
            break

    if not data:
        # Bad market data poisons recommendations — don't cache the fallback
        return MarketData(sources=["fallback"])

    result = MarketData(
        market_skills=data.market_skills,
        trending_tools=data.trending_tools,
        industry_trends=data.industry_trends,
        sample_jobs=data.sample_jobs,
        sources=["Gemini 2.5 Flash (Google Search)"],
    )
    _market_cache[cache_key] = result
    return result
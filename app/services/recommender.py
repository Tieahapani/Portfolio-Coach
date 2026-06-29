import json
from openai import AsyncOpenAI
from app.config import get_settings
from app.schemas import GitHubProfile, MarketData


RECOMMEND_PROMPT = """You are an expert Portfolio Coach. Analyze this developer's GitHub profile against real market demands and recommend projects to build.

## Developer's GitHub Profile
Languages: {languages}
Frameworks/Tools: {frameworks}
Topics: {topics}
Repos ({total_repos} total):
{repo_summaries}

## Target Role: {target_role}

## Live Market Data (from job postings)
In-demand skills: {market_skills}
Trending tools: {trending_tools}
Industry trends: {industry_trends}
Sample jobs found:
{sample_jobs}

## Instructions
Identify skill gaps and recommend 4 concrete, buildable projects that bridge them.
For EACH project, include 3-4 learning resources (real courses, tutorials, YouTube channels, documentation) that will help the developer learn the skills needed to build it. Tailor resources to their current skill level based on their profile.

Respond with ONLY raw JSON (no markdown, no backticks):
{{
  "profile_summary": "2-3 sentence strengths assessment",
  "skill_gaps": ["gap1", "gap2"],
  "matched_skills": ["skill1", "skill2"],
  "projects": [
    {{
      "title": "Project Name",
      "description": "3-4 sentence description with specific features and architecture",
      "difficulty": "Beginner|Intermediate|Advanced",
      "skills_gained": ["skill1", "skill2"],
      "tech_stack": ["tech1", "tech2"],
      "why": "Why this bridges a specific gap",
      "build_time_estimate": "e.g. 2-3 weeks",
      "resources": [
        {{
          "title": "Resource name",
          "type": "course|video|docs|tutorial|book",
          "platform": "YouTube|Coursera|freeCodeCamp|Udemy|Official Docs|etc",
          "url": "https://...",
          "why": "Why this resource helps for this specific project"
        }}
      ]
    }}
  ],
  "overall_strategy": "2 sentence career positioning advice"
}}"""


def _build_prompt(
    profile: GitHubProfile, market: MarketData, target_role: str
) -> str:
    """Build the recommendation prompt with all context."""
    repo_lines = []
    for r in profile.repos[:15]:
        line = f"- {r.name} ({r.language or '?'}) – {(r.description or '')[:80]}"
        if r.readme_excerpt:
            line += f"\n  README: {r.readme_excerpt[:200]}"
        repo_lines.append(line)

    job_lines = [
        f"- {j.title} at {j.company}: {', '.join(j.key_skills)}"
        for j in market.sample_jobs
    ]

    return RECOMMEND_PROMPT.format(
        languages=", ".join(profile.languages),
        frameworks=", ".join(profile.frameworks),
        topics=", ".join(profile.topics),
        total_repos=profile.total_repos,
        repo_summaries="\n".join(repo_lines),
        target_role=target_role,
        market_skills=", ".join(market.market_skills),
        trending_tools=", ".join(market.trending_tools),
        industry_trends=market.industry_trends,
        sample_jobs="\n".join(job_lines) or "None found",
    )


def _parse_json(text: str) -> dict | None:
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


async def generate_recommendations(
    profile: GitHubProfile,
    market: MarketData,
    target_role: str,
) -> tuple[dict, str]:
    """Generate project recommendations using GPT-4o-mini."""
    settings = get_settings()
    prompt = _build_prompt(profile, market, target_role)

    try:
        client = AsyncOpenAI(api_key=settings.openai_api_key)
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=8000,
        )
        text = response.choices[0].message.content or ""
        result = _parse_json(text)
        if result:
            return result, "GPT-4o-mini"
    except Exception as e:
        print(f"OpenAI recommendation failed: {e}")

    return {}, "none"

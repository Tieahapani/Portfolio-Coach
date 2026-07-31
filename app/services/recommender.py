from app.schemas import GitHubProfile, MarketData, RecommendationOutput
from app.services.llm_loop import call_with_verification


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

IMPORTANT — do not default to new projects every time. First look at the developer's EXISTING repos: if extending one of them would teach a market-demanded skill they lack, recommend that enhancement instead of a brand-new project. For enhancements, name the exact repo (e.g. "Add Kubernetes deployment + observability to 'my-api'") and describe what to add and which gap it closes. Aim for a mix: typically 1-2 enhancements of existing repos (when they fit) and the rest new projects. Only recommend all-new projects if none of their repos are worth extending toward the target role.

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


VALID_DIFFICULTIES = {"Beginner", "Intermediate", "Advanced"}


def _make_verifier(profile: GitHubProfile, market: MarketData, target_role: str):
    """Encode judgment once: fact checks a recommendation must pass.

    Pure code, run on every attempt — the LLM can't ship an answer that
    contradicts the real GitHub or job market data we fetched.
    """
    repo_names = {r.name.lower() for r in profile.repos}
    # Everything we actually observed about the developer, lowercased
    profile_text = " ".join(
        profile.languages + profile.frameworks + profile.topics
        + [f"{r.name} {r.description or ''} {r.readme_excerpt}" for r in profile.repos]
    ).lower()
    # Role relevance ground truth: in-demand skills list + skills from the
    # actual job postings we found. A project must connect to BOTH.
    market_text = " ".join(market.market_skills + market.trending_tools).lower()
    jobs_text = " ".join(
        s for j in market.sample_jobs for s in j.key_skills
    ).lower()

    def _core_skills(p) -> list[str]:
        """The project's declared purpose: skills_gained, minus hedged
        entries like 'React (optional, if frontend needs building)' —
        if it's optional, it doesn't count as evidence of relevance."""
        return [
            s.lower() for s in p.skills_gained
            if s and "optional" not in s.lower() and "if needed" not in s.lower()
        ]

    def _in_text(term: str, text: str) -> bool:
        words = [w for w in term.split() if len(w) > 2]
        return any(w in text for w in words)

    def _grounded(skill: str) -> bool:
        """A claimed skill passes if any meaningful word of it has evidence
        in the profile ('Python scripting' → 'python' ✓, 'Kubernetes' ✗)."""
        words = [w for w in skill.lower().split() if len(w) > 2]
        return any(w in profile_text for w in words)

    def verify(out: RecommendationOutput) -> list[str]:
        errors = []
        for skill in out.matched_skills:
            if not _grounded(skill):
                errors.append(
                    f"matched_skills claims '{skill}' but it appears nowhere in their "
                    "repos, languages, frameworks, or READMEs — only list skills with real evidence"
                )
        # Claiming nothing is also wrong when we can see real skills
        if not out.matched_skills and (profile.languages or profile.frameworks):
            errors.append(
                "matched_skills is empty, but they demonstrably know: "
                f"{', '.join((profile.languages + profile.frameworks)[:10])} — "
                "list which of their real skills are relevant to the target role"
            )
        for p in out.projects:
            if p.difficulty not in VALID_DIFFICULTIES:
                errors.append(f"project '{p.title}': difficulty must be one of {sorted(VALID_DIFFICULTIES)}")
            if not p.tech_stack:
                errors.append(f"project '{p.title}': tech_stack is empty")
            if not p.skills_gained:
                errors.append(f"project '{p.title}': skills_gained is empty")
            # A NEW project whose title equals an existing repo is a duplicate.
            # (Enhancements are fine — they mention the repo in the description.)
            if p.title.lower() in repo_names and "add" not in p.title.lower():
                errors.append(
                    f"project '{p.title}' duplicates a repo they already have — "
                    "recommend an enhancement to it or a different project"
                )
            # Role relevance, hardened against gaming: judge the project by
            # its core skills_gained (hedged/"optional" entries don't count),
            # and require the MAJORITY of them to be in market demand — one
            # token relevant skill can't carry an off-role project.
            core = _core_skills(p)
            if core and market_text:
                matched = sum(1 for s in core if _in_text(s, market_text))
                if matched * 2 < len(core):
                    errors.append(
                        f"project '{p.title}': only {matched} of its {len(core)} core "
                        f"skills are in demand for '{target_role}' — at least half must be. "
                        "Redesign it around the in-demand skills, don't just tag them on as optional"
                    )
            if core and jobs_text and not any(_in_text(s, jobs_text) for s in core):
                errors.append(
                    f"project '{p.title}' matches no skill from the actual job postings "
                    f"for '{target_role}' — align it with skills real employers listed"
                )
        return errors[:6]  # cap feedback so the retry prompt stays focused

    return verify


async def generate_recommendations(
    profile: GitHubProfile,
    market: MarketData,
    target_role: str,
) -> tuple[dict, str]:
    """Generate project recommendations via the verified LLM loop."""
    prompt = _build_prompt(profile, market, target_role)

    result = await call_with_verification(
        [{"role": "user", "content": prompt}],
        schema=RecommendationOutput,
        verify=_make_verifier(profile, market, target_role),
        label="recommend",
    )
    if result:
        return result.model_dump(), "Gemini-2.5-Flash"
    return {}, "none"

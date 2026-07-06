"""Progress Coach agent: Gemini 2.5 Flash with function calling.

The agent inspects a tracked project's repo (file tree, README, commits),
compares what's built against the recommended project's scope, and returns
a coaching assessment. Capped tool loop + fallback so the UI never breaks.
"""

import httpx
from cachetools import TTLCache
from openai import AsyncOpenAI

from app.config import get_settings
from app.services.github_service import _build_headers, fetch_readme
from app.services.project_tracking import _fetch_repo_commits
from app.services.recommender import GEMINI_BASE_URL, _parse_json

MAX_TOOL_ROUNDS = 6

# Coach results keyed by (project_id, commit_count) — new commits bust the cache.
_coach_cache: TTLCache = TTLCache(maxsize=200, ttl=6 * 60 * 60)


async def _get_file_tree(username: str, repo: str) -> str:
    """Repo file paths (up to 150) via the git trees API."""
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(
            f"https://api.github.com/repos/{username}/{repo}/git/trees/HEAD",
            params={"recursive": "1"},
            headers=_build_headers(),
        )
        if resp.status_code != 200:
            return "Could not fetch file tree."
        paths = [t["path"] for t in resp.json().get("tree", []) if t["type"] == "blob"]
        return "\n".join(paths[:150]) or "Repo is empty."


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_file_tree",
            "description": "List all file paths in the project repo. Use to see what has actually been built.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_readme",
            "description": "Get the repo's README content.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_recent_commits",
            "description": "Get commit messages and dates since the project was accepted.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

AGENT_PROMPT = """You are a Progress Coach for a developer building a recommended portfolio project.

## The recommended project (the scope they committed to)
Title: {title}
Description: {description}
Tech stack: {tech_stack}
Skills to gain: {skills_gained}

## Their repo
{username}/{repo} — {commit_count} commits since accepting, status: {status}

Use your tools to inspect the repo (file tree, README, commits) and judge how much
of the recommended scope is actually built. Be honest and specific — reference real
files and commits you saw. Then respond with ONLY raw JSON (no markdown):
{{
  "progress_pct": 0-100,
  "built_so_far": "2-3 sentences on what exists, referencing actual files/commits",
  "missing": ["scope item not yet built", "..."],
  "next_commits": ["specific commit-sized task 1", "task 2", "task 3"],
  "verdict": "1-2 sentence honest coaching assessment"
}}"""


def _fallback(project: dict) -> dict:
    """Non-agent summary if the tool loop fails."""
    return {
        "progress_pct": None,
        "built_so_far": f"{project.get('commit_count', 0)} commits so far on {project.get('linked_repo')}.",
        "missing": [],
        "next_commits": [],
        "verdict": "Coach analysis unavailable right now — keep committing and try again shortly.",
        "fallback": True,
    }


async def coach_project(project: dict) -> dict:
    """Run the agent loop for one tracked project (must have linked_repo)."""
    settings = get_settings()
    username = project["github_username"]
    repo = project["linked_repo"]

    cache_key = (project["id"], project.get("commit_count", 0))
    cached = _coach_cache.get(cache_key)
    if cached is not None:
        return cached

    async def run_tool(name: str) -> str:
        if name == "get_file_tree":
            return await _get_file_tree(username, repo)
        if name == "get_readme":
            return (await fetch_readme(username, repo)) or "No README."
        if name == "get_recent_commits":
            commits = await _fetch_repo_commits(username, repo, project["accepted_at"])
            return "\n".join(f"{c['date']}: {c['message']}" for c in commits) or "No commits."
        return f"Unknown tool: {name}"

    messages = [
        {
            "role": "user",
            "content": AGENT_PROMPT.format(
                title=project["title"],
                description=project["description"],
                tech_stack=", ".join(project.get("tech_stack", [])),
                skills_gained=", ".join(project.get("skills_gained", [])),
                username=username,
                repo=repo,
                commit_count=project.get("commit_count", 0),
                status=project.get("status", "unknown"),
            ),
        }
    ]

    try:
        client = AsyncOpenAI(
            api_key=settings.effective_gemini_key,
            base_url=GEMINI_BASE_URL,
        )
        for round_num in range(MAX_TOOL_ROUNDS + 1):
            # Last round: force a final answer, no more tools
            use_tools = round_num < MAX_TOOL_ROUNDS
            response = await client.chat.completions.create(
                model="gemini-2.5-flash",
                messages=messages,
                tools=TOOLS if use_tools else None,
                temperature=0.4,
                max_tokens=8000,
                reasoning_effort="low",
            )
            msg = response.choices[0].message

            if msg.tool_calls:
                messages.append(msg)
                for tc in msg.tool_calls:
                    result = await run_tool(tc.function.name)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })
                continue

            result = _parse_json(msg.content or "")
            if result:
                result["fallback"] = False
                _coach_cache[cache_key] = result
                return result
            break
    except Exception as e:
        print(f"Progress agent failed for {username}/{repo}: {e}")

    return _fallback(project)

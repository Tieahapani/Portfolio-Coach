"""Background commit tracker — monitors registered users' GitHub activity
and checks alignment with their target role's market demands."""

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import httpx
from google import genai

from app.config import get_settings

logger = logging.getLogger(__name__)

DATA_FILE = Path(__file__).parent.parent.parent / "data" / "tracked_users.json"

ALIGNMENT_PROMPT = """You are a career coach analyzing a developer's recent GitHub activity against their target role.

## Target Role: {target_role}

## Recent Commits (last 7 days)
{commit_summary}

## Their Current Skills
Languages: {languages}

## Market Demands for {target_role}
{market_skills}

Analyze whether their recent work aligns with market demands for their target role.

Respond with ONLY raw JSON (no markdown, no backticks):
{{
  "alignment_score": 0.0 to 1.0,
  "aligned_areas": ["what they're doing that matches market demand"],
  "gaps": ["what the market wants that they're not working on"],
  "nudge": "1-2 sentence actionable suggestion for what to work on next",
  "recent_focus": "1 sentence summary of what they've been building"
}}"""


def _load_data() -> dict:
    """Load tracked users from JSON file."""
    if DATA_FILE.exists():
        return json.loads(DATA_FILE.read_text())
    return {"users": {}}


def _save_data(data: dict) -> None:
    """Save tracked users to JSON file."""
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    DATA_FILE.write_text(json.dumps(data, indent=2, default=str))


def register_user(username: str, target_role: str) -> dict:
    """Register a user for commit tracking."""
    data = _load_data()
    data["users"][username.lower()] = {
        "username": username,
        "target_role": target_role,
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "last_checked": None,
        "insights": [],
    }
    _save_data(data)
    return data["users"][username.lower()]


def unregister_user(username: str) -> None:
    """Remove a user from tracking."""
    data = _load_data()
    data["users"].pop(username.lower(), None)
    _save_data(data)


def get_tracked_users() -> list[dict]:
    """Get all tracked users."""
    data = _load_data()
    return list(data["users"].values())


def get_user_insights(username: str) -> dict | None:
    """Get a specific user's tracking data and insights."""
    data = _load_data()
    return data["users"].get(username.lower())


async def fetch_recent_commits(username: str) -> list[dict]:
    """Fetch a user's recent commit activity using two strategies:

    1. Events API — catches PushEvents (direct pushes) and PullRequestEvents
       (merged PRs). Fast but only covers last 90 days of *public* activity.
    2. Commits API — falls back to scanning the user's most recently pushed
       repos and fetching their actual commit history. Catches everything
       including commits that didn't trigger a public event.

    Both are combined and deduplicated so we get the fullest picture.
    """
    settings = get_settings()
    headers = {"Accept": "application/vnd.github.v3+json"}
    if settings.has_github_token:
        headers["Authorization"] = f"token {settings.github_token}"

    commits = []
    seen_messages = set()

    async with httpx.AsyncClient(timeout=15) as client:
        # ── Strategy 1: Events API ──
        # PushEvent = direct pushes to a branch
        # PullRequestEvent = PRs opened/merged (shows project activity)
        try:
            resp = await client.get(
                f"https://api.github.com/users/{username}/events",
                headers=headers,
                params={"per_page": 50},
            )
            resp.raise_for_status()
            events = resp.json()

            for event in events:
                repo_name = event.get("repo", {}).get("name", "")
                short_repo = repo_name.split("/")[-1] if "/" in repo_name else repo_name
                created_at = event.get("created_at", "")

                if event.get("type") == "PushEvent":
                    # Each push can contain multiple commits
                    for commit in event.get("payload", {}).get("commits", []):
                        msg = commit.get("message", "")[:200]
                        if msg not in seen_messages:
                            seen_messages.add(msg)
                            commits.append({
                                "repo": short_repo,
                                "message": msg,
                                "date": created_at,
                            })

                elif event.get("type") == "PullRequestEvent":
                    # PR activity shows what they're working on
                    pr = event.get("payload", {}).get("pull_request", {})
                    action = event.get("payload", {}).get("action", "")
                    title = pr.get("title", "")
                    if title and action in ("opened", "closed") and title not in seen_messages:
                        seen_messages.add(title)
                        prefix = "PR merged" if pr.get("merged") else f"PR {action}"
                        commits.append({
                            "repo": short_repo,
                            "message": f"{prefix}: {title}",
                            "date": created_at,
                        })
        except Exception as e:
            logger.error(f"Events API failed for {username}: {e}")

        # ── Strategy 2: Commits API (fallback / supplement) ──
        # Directly fetch commits from their most active repos.
        # This catches commits the Events API might miss.
        if len(commits) < 10:
            try:
                resp = await client.get(
                    f"https://api.github.com/users/{username}/repos",
                    headers=headers,
                    params={"per_page": 6, "sort": "pushed", "direction": "desc"},
                )
                resp.raise_for_status()
                repos = resp.json()

                for repo in repos:
                    if repo.get("fork"):
                        continue
                    repo_name = repo.get("name", "")
                    owner = repo.get("owner", {}).get("login", username)
                    try:
                        resp = await client.get(
                            f"https://api.github.com/repos/{owner}/{repo_name}/commits",
                            headers=headers,
                            params={"author": username, "per_page": 5},
                        )
                        if resp.status_code != 200:
                            continue
                        for c in resp.json():
                            msg = c.get("commit", {}).get("message", "")[:200]
                            date = c.get("commit", {}).get("author", {}).get("date", "")
                            if msg not in seen_messages:
                                seen_messages.add(msg)
                                commits.append({
                                    "repo": repo_name,
                                    "message": msg,
                                    "date": date,
                                })
                    except Exception:
                        continue
            except Exception as e:
                logger.error(f"Commits API failed for {username}: {e}")

    commits.sort(key=lambda x: x.get("date", ""), reverse=True)
    return commits[:30]


async def analyze_alignment(
    username: str,
    commits: list[dict],
    target_role: str,
    languages: list[str],
    market_skills: list[str],
) -> dict | None:
    """Use Gemini 2.5 Flash to analyze commit alignment with market demands."""
    settings = get_settings()
    if not settings.has_gemini or not commits:
        return None

    commit_lines = [
        f"- [{c['repo']}] {c['message']}" for c in commits[:20]
    ]

    prompt = ALIGNMENT_PROMPT.format(
        target_role=target_role,
        commit_summary="\n".join(commit_lines),
        languages=", ".join(languages) if languages else "Unknown",
        market_skills=", ".join(market_skills) if market_skills else "Not available",
    )

    try:
        client = genai.Client(api_key=settings.effective_gemini_key)
        response = await asyncio.to_thread(
            client.models.generate_content,
            model="gemini-2.5-flash",
            contents=prompt,
        )
        text = response.text or ""
        cleaned = text.replace("```json", "").replace("```", "").strip()
        return json.loads(cleaned)
    except Exception as e:
        logger.error(f"Alignment analysis failed for {username}: {e}")
        return None


async def check_user(username: str) -> dict | None:
    """Run a full check for a single user: fetch commits, analyze alignment."""
    from app.services.github_service import analyze_github
    from app.services.market_research import research_market

    data = _load_data()
    user = data["users"].get(username.lower())
    if not user:
        return None

    target_role = user["target_role"]

    # Fetch recent commits + profile + market data
    commits = await fetch_recent_commits(username)
    if not commits:
        return {"status": "no_commits", "message": "No recent commits found"}

    profile = await analyze_github(username, fetch_readmes=False)
    market = await research_market(target_role)

    # Analyze alignment
    insight = await analyze_alignment(
        username,
        commits,
        target_role,
        profile.languages,
        market.market_skills,
    )

    if not insight:
        return {"status": "analysis_failed"}

    # Save insight
    insight["checked_at"] = datetime.now(timezone.utc).isoformat()
    insight["commit_count"] = len(commits)

    user["last_checked"] = insight["checked_at"]
    # Keep last 10 insights
    user["insights"] = [insight] + user.get("insights", [])[:9]
    data["users"][username.lower()] = user
    _save_data(data)

    return insight


async def run_tracker_cycle() -> None:
    """Check all registered users. Called by background loop."""
    data = _load_data()
    users = list(data["users"].values())

    if not users:
        return

    logger.info(f"Tracker cycle: checking {len(users)} users")
    for user in users:
        try:
            await check_user(user["username"])
            logger.info(f"Tracked: {user['username']}")
        except Exception as e:
            logger.error(f"Tracker failed for {user['username']}: {e}")

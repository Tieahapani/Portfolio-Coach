"""Background commit tracker — monitors registered users' GitHub activity
and checks alignment with their target role's market demands."""

import asyncio
import hashlib
import json
import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
from google import genai

from app.config import get_settings

logger = logging.getLogger(__name__)

# Legacy JSON store — migrated into SQLite on first DB access, then renamed .bak
LEGACY_JSON = Path(__file__).parent.parent.parent / "data" / "tracked_users.json"
DB_PATH = Path(__file__).parent.parent.parent / "data" / "peers.db"

MAX_INSIGHTS = 10

ALIGNMENT_PROMPT = """You are a career coach analyzing a developer's recent GitHub activity against their target role.

## Target Role: {target_role}

## Recent Commits (last 90 days)
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


def _get_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS tracked_users (
            username TEXT PRIMARY KEY,
            display_username TEXT NOT NULL,
            target_role TEXT NOT NULL,
            registered_at TEXT NOT NULL,
            last_checked TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS tracker_insights (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            checked_at TEXT NOT NULL,
            payload TEXT NOT NULL
        )
    """)
    conn.commit()
    _migrate_legacy_json(conn)
    return conn


def _migrate_legacy_json(conn: sqlite3.Connection) -> None:
    """One-time import of the old tracked_users.json, then rename it to .bak."""
    if not LEGACY_JSON.exists():
        return
    try:
        data = json.loads(LEGACY_JSON.read_text())
        for key, user in data.get("users", {}).items():
            conn.execute(
                """INSERT OR IGNORE INTO tracked_users
                   (username, display_username, target_role, registered_at, last_checked)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    key,
                    user.get("username", key),
                    user.get("target_role", ""),
                    user.get("registered_at", datetime.now(timezone.utc).isoformat()),
                    user.get("last_checked"),
                ),
            )
            # insights are stored newest-first in the JSON; keep that order
            for insight in reversed(user.get("insights", [])[:MAX_INSIGHTS]):
                conn.execute(
                    "INSERT INTO tracker_insights (username, checked_at, payload) VALUES (?, ?, ?)",
                    (key, insight.get("checked_at", ""), json.dumps(insight)),
                )
        conn.commit()
        LEGACY_JSON.rename(LEGACY_JSON.with_suffix(".json.bak"))
        logger.info(f"Migrated {len(data.get('users', {}))} tracked users from JSON to SQLite")
    except Exception as e:
        logger.error(f"Tracker JSON migration failed: {e}")


def _user_row_to_dict(row: sqlite3.Row) -> dict:
    return {
        "username": row["display_username"],
        "target_role": row["target_role"],
        "registered_at": row["registered_at"],
        "last_checked": row["last_checked"],
    }


def _fetch_insights(conn: sqlite3.Connection, username_key: str) -> list[dict]:
    rows = conn.execute(
        "SELECT payload FROM tracker_insights WHERE username = ? ORDER BY id DESC LIMIT ?",
        (username_key, MAX_INSIGHTS),
    ).fetchall()
    return [json.loads(r["payload"]) for r in rows]


def register_user(username: str, target_role: str) -> dict:
    """Register a user for commit tracking (re-registering updates the role)."""
    now = datetime.now(timezone.utc).isoformat()
    conn = _get_db()
    conn.execute(
        """INSERT INTO tracked_users (username, display_username, target_role, registered_at)
           VALUES (?, ?, ?, ?)
           ON CONFLICT(username) DO UPDATE SET target_role = excluded.target_role""",
        (username.lower(), username, target_role, now),
    )
    conn.commit()
    conn.close()
    user = get_user_insights(username)
    return user


def unregister_user(username: str) -> None:
    """Remove a user from tracking."""
    conn = _get_db()
    conn.execute("DELETE FROM tracked_users WHERE username = ?", (username.lower(),))
    conn.execute("DELETE FROM tracker_insights WHERE username = ?", (username.lower(),))
    conn.commit()
    conn.close()


def get_tracked_users() -> list[dict]:
    """Get all tracked users (without insight history)."""
    conn = _get_db()
    rows = conn.execute("SELECT * FROM tracked_users").fetchall()
    conn.close()
    return [_user_row_to_dict(r) for r in rows]


def get_user_insights(username: str) -> dict | None:
    """Get a user's tracking data with their insight history (newest first)."""
    conn = _get_db()
    row = conn.execute(
        "SELECT * FROM tracked_users WHERE username = ?", (username.lower(),)
    ).fetchone()
    if not row:
        conn.close()
        return None
    user = _user_row_to_dict(row)
    user["insights"] = _fetch_insights(conn, username.lower())
    conn.close()
    return user


def _touch_last_checked(username_key: str, checked_at: str) -> None:
    conn = _get_db()
    conn.execute(
        "UPDATE tracked_users SET last_checked = ? WHERE username = ?",
        (checked_at, username_key),
    )
    conn.commit()
    conn.close()


def _save_insight(username_key: str, insight: dict) -> None:
    """Store an insight, trim history to MAX_INSIGHTS, update last_checked."""
    conn = _get_db()
    conn.execute(
        "INSERT INTO tracker_insights (username, checked_at, payload) VALUES (?, ?, ?)",
        (username_key, insight.get("checked_at", ""), json.dumps(insight)),
    )
    conn.execute(
        """DELETE FROM tracker_insights WHERE username = ? AND id NOT IN (
               SELECT id FROM tracker_insights WHERE username = ? ORDER BY id DESC LIMIT ?
           )""",
        (username_key, username_key, MAX_INSIGHTS),
    )
    conn.execute(
        "UPDATE tracked_users SET last_checked = ? WHERE username = ?",
        (insight.get("checked_at"), username_key),
    )
    conn.commit()
    conn.close()


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

    cutoff = datetime.now(timezone.utc) - timedelta(days=90)
    cutoff_iso = cutoff.strftime("%Y-%m-%dT%H:%M:%SZ")

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
                if created_at and created_at < cutoff_iso:
                    continue

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
                    repo_name = repo.get("name", "")
                    owner = repo.get("owner", {}).get("login", username)
                    # Skip repos with no pushes inside the 90-day window
                    pushed_at = repo.get("pushed_at", "")
                    if pushed_at and pushed_at < cutoff_iso:
                        continue
                    try:
                        resp = await client.get(
                            f"https://api.github.com/repos/{owner}/{repo_name}/commits",
                            headers=headers,
                            params={"author": username, "per_page": 5, "since": cutoff_iso},
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

    user = get_user_insights(username)
    if not user:
        return None

    target_role = user["target_role"]

    # Fetch recent commits + profile + market data
    commits = await fetch_recent_commits(username)
    if not commits:
        return {"status": "no_commits", "message": "No recent commits found"}

    # Content-based cache: if commits haven't changed since the last insight
    # for the same target role, reuse it instead of a fresh (nondeterministic) LLM call.
    commit_hash = hashlib.sha256(
        "\n".join(f"{c['repo']}:{c['message']}" for c in commits).encode()
    ).hexdigest()
    last_insight = (user.get("insights") or [None])[0]
    if (
        last_insight
        and last_insight.get("commit_hash") == commit_hash
        and last_insight.get("target_role") == target_role
    ):
        _touch_last_checked(username.lower(), datetime.now(timezone.utc).isoformat())
        return last_insight

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
    insight["commit_hash"] = commit_hash
    insight["target_role"] = target_role

    _save_insight(username.lower(), insight)

    return insight


async def run_tracker_cycle() -> None:
    """Check all registered users. Called by background loop."""
    users = get_tracked_users()

    if not users:
        return

    logger.info(f"Tracker cycle: checking {len(users)} users")
    for user in users:
        try:
            await check_user(user["username"])
            logger.info(f"Tracked: {user['username']}")
        except Exception as e:
            logger.error(f"Tracker failed for {user['username']}: {e}")

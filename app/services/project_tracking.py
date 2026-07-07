"""Tracked-project lifecycle: suggest repo name, auto-detect repo, check activity.

Status flow: accepted (no repo) → active (repo linked, recent commits)
             → stalled (no commits in 14 days) → completed (manual)
"""

import re
from datetime import datetime, timezone, timedelta

import httpx

from app.services.github_service import _build_headers
from app.services.project_db import get_projects, get_tracked_usernames, update_project

STALL_DAYS = 14


def slugify_title(title: str) -> str:
    """'Flaky Test Predictor CLI' → 'flaky-test-predictor-cli'"""
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return slug[:100] or "recommended-project"


async def _repo_exists(username: str, repo_name: str) -> bool:
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(
            f"https://api.github.com/repos/{username}/{repo_name}",
            headers=_build_headers(),
        )
        return resp.status_code == 200


async def _fetch_repo_commits(username: str, repo_name: str, since_iso: str) -> list[dict]:
    """Commits on one repo since a date: [{message, date}]."""
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(
            f"https://api.github.com/repos/{username}/{repo_name}/commits",
            params={"since": since_iso, "per_page": 100, "author": username},
            headers=_build_headers(),
        )
        if resp.status_code != 200:
            return []
        return [
            {
                "message": c.get("commit", {}).get("message", "").split("\n")[0],
                "date": c.get("commit", {}).get("author", {}).get("date", ""),
            }
            for c in resp.json()
        ]


async def _fetch_contributors(username: str, repo_name: str) -> list[str]:
    """Contributor logins on the repo, excluding the owner and bots."""
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(
            f"https://api.github.com/repos/{username}/{repo_name}/contributors",
            params={"per_page": 20},
            headers=_build_headers(),
        )
        if resp.status_code != 200:
            return []
        return [
            c["login"]
            for c in resp.json()
            if c.get("login")
            and c["login"].lower() != username.lower()
            and not c["login"].endswith("[bot]")
        ]


def _activity_status(commit_count: int, last_commit_at: str | None) -> str:
    if commit_count == 0:
        return "not_started"
    cutoff = datetime.now(timezone.utc) - timedelta(days=STALL_DAYS)
    if last_commit_at and datetime.fromisoformat(last_commit_at.replace("Z", "+00:00")) < cutoff:
        return "stalled"
    return "active"


async def refresh_projects(github_username: str) -> list[dict]:
    """Refresh every tracked project for a user (called on Projects page load).

    - accepted + no repo yet  → check if suggested repo now exists → link it
    - linked                  → fetch commits since accept, update count/status
    - completed               → left alone
    """
    username = github_username.lower()
    now = datetime.now(timezone.utc).isoformat()
    refreshed = []

    for project in get_projects(username):
        if project["status"] == "completed":
            refreshed.append(project)
            continue

        repo = project.get("linked_repo")

        # Auto-detect: suggested repo appeared on GitHub?
        if not repo:
            try:
                if await _repo_exists(username, project["suggested_repo_name"]):
                    repo = project["suggested_repo_name"]
                    project = update_project(project["id"], linked_repo=repo)
            except Exception:
                pass

        if not repo:
            refreshed.append(update_project(project["id"], last_checked=now))
            continue

        # Activity check on the linked repo
        try:
            commits = await _fetch_repo_commits(username, repo, project["accepted_at"])
            contributors = await _fetch_contributors(username, repo)
            last_commit_at = commits[0]["date"] if commits else None
            status = _activity_status(len(commits), last_commit_at)
            project = update_project(
                project["id"],
                commit_count=len(commits),
                last_commit_at=last_commit_at,
                status=status,
                last_checked=now,
                contributors=contributors,
            )
        except Exception:
            project = update_project(project["id"], last_checked=now)

        refreshed.append(project)

    return refreshed


async def refresh_all_projects() -> None:
    """Refresh every user's tracked projects (background loop)."""
    for username in get_tracked_usernames():
        try:
            await refresh_projects(username)
        except Exception as e:
            print(f"Background refresh failed for {username}: {e}")

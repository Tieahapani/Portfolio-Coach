"""SQLite storage for tracked projects (accepted AI recommendations)."""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).parent.parent.parent / "data" / "peers.db"


def _get_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS tracked_projects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            github_username TEXT NOT NULL,
            title TEXT NOT NULL,
            description TEXT DEFAULT '',
            tech_stack TEXT DEFAULT '[]',
            skills_gained TEXT DEFAULT '[]',
            difficulty TEXT DEFAULT 'Intermediate',
            suggested_repo_name TEXT NOT NULL,
            linked_repo TEXT,
            status TEXT DEFAULT 'accepted',
            accepted_at TEXT NOT NULL,
            last_checked TEXT,
            commit_count INTEGER DEFAULT 0,
            last_commit_at TEXT,
            contributors TEXT DEFAULT '[]',
            UNIQUE(github_username, suggested_repo_name)
        )
    """)
    # Migration for tables created before the contributors column existed
    cols = [r[1] for r in conn.execute("PRAGMA table_info(tracked_projects)")]
    if "contributors" not in cols:
        conn.execute("ALTER TABLE tracked_projects ADD COLUMN contributors TEXT DEFAULT '[]'")
    conn.commit()
    return conn


def _row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    for field in ("tech_stack", "skills_gained", "contributors"):
        d[field] = json.loads(d.get(field) or "[]")
    return d


def add_project(
    github_username: str,
    title: str,
    description: str,
    tech_stack: list[str],
    skills_gained: list[str],
    difficulty: str,
    suggested_repo_name: str,
) -> dict:
    """Insert a tracked project. Re-accepting the same project is a no-op."""
    now = datetime.now(timezone.utc).isoformat()
    conn = _get_db()
    conn.execute(
        """
        INSERT INTO tracked_projects (github_username, title, description,
            tech_stack, skills_gained, difficulty, suggested_repo_name, accepted_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(github_username, suggested_repo_name) DO NOTHING
        """,
        (
            github_username.lower(),
            title,
            description,
            json.dumps(tech_stack),
            json.dumps(skills_gained),
            difficulty,
            suggested_repo_name,
            now,
        ),
    )
    conn.commit()
    row = conn.execute(
        "SELECT * FROM tracked_projects WHERE github_username = ? AND suggested_repo_name = ?",
        (github_username.lower(), suggested_repo_name),
    ).fetchone()
    conn.close()
    return _row_to_dict(row)


def get_projects(github_username: str) -> list[dict]:
    conn = _get_db()
    rows = conn.execute(
        "SELECT * FROM tracked_projects WHERE github_username = ? ORDER BY accepted_at DESC",
        (github_username.lower(),),
    ).fetchall()
    conn.close()
    return [_row_to_dict(r) for r in rows]


def get_tracked_usernames() -> list[str]:
    """Distinct usernames with at least one non-completed tracked project."""
    conn = _get_db()
    rows = conn.execute(
        "SELECT DISTINCT github_username FROM tracked_projects WHERE status != 'completed'"
    ).fetchall()
    conn.close()
    return [r["github_username"] for r in rows]


def get_project(project_id: int) -> dict | None:
    conn = _get_db()
    row = conn.execute(
        "SELECT * FROM tracked_projects WHERE id = ?", (project_id,)
    ).fetchone()
    conn.close()
    return _row_to_dict(row) if row else None


def update_project(project_id: int, **fields) -> dict | None:
    """Update arbitrary columns (linked_repo, status, commit_count, ...)."""
    allowed = {
        "linked_repo", "status", "last_checked", "commit_count", "last_commit_at",
        "contributors",
    }
    updates = {k: v for k, v in fields.items() if k in allowed}
    if isinstance(updates.get("contributors"), list):
        updates["contributors"] = json.dumps(updates["contributors"])
    if not updates:
        return get_project(project_id)
    conn = _get_db()
    sets = ", ".join(f"{k} = ?" for k in updates)
    conn.execute(
        f"UPDATE tracked_projects SET {sets} WHERE id = ?",
        (*updates.values(), project_id),
    )
    conn.commit()
    conn.close()
    return get_project(project_id)


def delete_project(project_id: int) -> bool:
    conn = _get_db()
    cur = conn.execute("DELETE FROM tracked_projects WHERE id = ?", (project_id,))
    conn.commit()
    conn.close()
    return cur.rowcount > 0

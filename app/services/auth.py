"""GitHub OAuth + JWT session + SQLite user storage."""

import json
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path

import httpx
import jwt
from app.config import get_settings

DB_PATH = Path(__file__).parent.parent.parent / "data" / "users.db"

GITHUB_AUTHORIZE_URL = "https://github.com/login/oauth/authorize"
GITHUB_TOKEN_URL = "https://github.com/login/oauth/access_token"
GITHUB_USER_URL = "https://api.github.com/user"

JWT_ALGORITHM = "HS256"
JWT_EXPIRY_DAYS = 30


# ── Database ──

def _get_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            github_username TEXT PRIMARY KEY,
            name TEXT DEFAULT '',
            avatar_url TEXT DEFAULT '',
            contact TEXT DEFAULT '',
            target_role TEXT DEFAULT '',
            current_projects TEXT DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn


def upsert_user(
    github_username: str,
    name: str = "",
    avatar_url: str = "",
    contact: str = "",
    target_role: str = "",
    current_projects: str = "",
) -> dict:
    now = datetime.now(timezone.utc).isoformat()
    conn = _get_db()
    conn.execute(
        """
        INSERT INTO users (github_username, name, avatar_url, contact,
                          target_role, current_projects, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(github_username) DO UPDATE SET
            name=excluded.name,
            avatar_url=excluded.avatar_url,
            contact=CASE WHEN excluded.contact != '' THEN excluded.contact ELSE users.contact END,
            target_role=CASE WHEN excluded.target_role != '' THEN excluded.target_role ELSE users.target_role END,
            current_projects=CASE WHEN excluded.current_projects != '' THEN excluded.current_projects ELSE users.current_projects END,
            updated_at=excluded.updated_at
        """,
        (github_username.lower(), name, avatar_url, contact,
         target_role, current_projects, now, now),
    )
    conn.commit()
    conn.close()
    return get_user(github_username)


def get_user(github_username: str) -> dict | None:
    conn = _get_db()
    row = conn.execute(
        "SELECT * FROM users WHERE github_username = ?",
        (github_username.lower(),),
    ).fetchone()
    conn.close()
    if not row:
        return None
    return dict(row)


def update_user_profile(github_username: str, contact: str = "", target_role: str = "", current_projects: str = "") -> dict | None:
    now = datetime.now(timezone.utc).isoformat()
    conn = _get_db()
    conn.execute(
        """
        UPDATE users SET
            contact = CASE WHEN ? != '' THEN ? ELSE contact END,
            target_role = CASE WHEN ? != '' THEN ? ELSE target_role END,
            current_projects = ?,
            updated_at = ?
        WHERE github_username = ?
        """,
        (contact, contact, target_role, target_role, current_projects, now, github_username.lower()),
    )
    conn.commit()
    conn.close()
    return get_user(github_username)


# ── GitHub OAuth ──

async def exchange_code_for_user(code: str) -> dict:
    """Exchange GitHub OAuth code for access token, then fetch user info."""
    settings = get_settings()
    async with httpx.AsyncClient() as client:
        token_resp = await client.post(
            GITHUB_TOKEN_URL,
            data={
                "client_id": settings.github_client_id,
                "client_secret": settings.github_client_secret,
                "code": code,
            },
            headers={"Accept": "application/json"},
        )
        token_data = token_resp.json()
        access_token = token_data.get("access_token")
        if not access_token:
            raise ValueError(f"OAuth failed: {token_data.get('error_description', 'unknown error')}")

        user_resp = await client.get(
            GITHUB_USER_URL,
            headers={"Authorization": f"Bearer {access_token}"},
        )
        user_data = user_resp.json()

    return {
        "github_username": user_data["login"],
        "name": user_data.get("name") or user_data["login"],
        "avatar_url": user_data.get("avatar_url", ""),
    }


# ── JWT ──

def create_token(github_username: str) -> str:
    settings = get_settings()
    payload = {
        "sub": github_username.lower(),
        "exp": datetime.now(timezone.utc) + timedelta(days=JWT_EXPIRY_DAYS),
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=JWT_ALGORITHM)


def verify_token(token: str) -> str | None:
    """Returns github_username if valid, None otherwise."""
    settings = get_settings()
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[JWT_ALGORITHM])
        return payload.get("sub")
    except jwt.PyJWTError:
        return None

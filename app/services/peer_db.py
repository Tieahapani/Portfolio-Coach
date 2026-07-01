"""SQLite storage for peer profiles + ChromaDB for embeddings."""

import sqlite3
from pathlib import Path

import chromadb
from chromadb.config import Settings

DB_PATH = Path(__file__).parent.parent.parent / "data" / "peers.db"
CHROMA_PATH = Path(__file__).parent.parent.parent / "data" / "chroma"


def _get_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS peers (
            github_username TEXT PRIMARY KEY,
            target_role TEXT NOT NULL,
            contact TEXT NOT NULL,
            current_projects TEXT DEFAULT '',
            matched_skills TEXT DEFAULT '[]',
            skill_gaps TEXT DEFAULT '[]',
            languages TEXT DEFAULT '[]',
            frameworks TEXT DEFAULT '[]',
            topics TEXT DEFAULT '[]',
            registered_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    # Migration for existing DBs
    try:
        conn.execute("SELECT topics FROM peers LIMIT 1")
    except sqlite3.OperationalError:
        conn.execute("ALTER TABLE peers ADD COLUMN topics TEXT DEFAULT '[]'")
    conn.commit()
    return conn


def _get_chroma():
    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_or_create_collection(
        name="peer_profiles",
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def upsert_peer(
    github_username: str,
    target_role: str,
    contact: str,
    current_projects: str,
    matched_skills: list[str],
    skill_gaps: list[str],
    languages: list[str],
    frameworks: list[str],
    topics: list[str] | None = None,
) -> dict:
    """Insert or update a peer profile in SQLite."""
    import json
    from datetime import datetime, timezone

    topics = topics or []
    now = datetime.now(timezone.utc).isoformat()
    conn = _get_db()
    conn.execute(
        """
        INSERT INTO peers (github_username, target_role, contact, current_projects,
                          matched_skills, skill_gaps, languages, frameworks, topics,
                          registered_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(github_username) DO UPDATE SET
            target_role=excluded.target_role,
            contact=excluded.contact,
            current_projects=excluded.current_projects,
            matched_skills=excluded.matched_skills,
            skill_gaps=excluded.skill_gaps,
            languages=excluded.languages,
            frameworks=excluded.frameworks,
            topics=excluded.topics,
            updated_at=excluded.updated_at
        """,
        (
            github_username.lower(),
            target_role,
            contact,
            current_projects,
            json.dumps(matched_skills),
            json.dumps(skill_gaps),
            json.dumps(languages),
            json.dumps(frameworks),
            json.dumps(topics),
            now,
            now,
        ),
    )
    conn.commit()
    conn.close()
    return get_peer(github_username)


def get_peer(github_username: str) -> dict | None:
    import json
    conn = _get_db()
    row = conn.execute(
        "SELECT * FROM peers WHERE github_username = ?",
        (github_username.lower(),),
    ).fetchone()
    conn.close()
    if not row:
        return None
    d = dict(row)
    for field in ("matched_skills", "skill_gaps", "languages", "frameworks", "topics"):
        d[field] = json.loads(d.get(field, "[]"))
    return d


def get_all_peers() -> list[dict]:
    import json
    conn = _get_db()
    rows = conn.execute("SELECT * FROM peers ORDER BY updated_at DESC").fetchall()
    conn.close()
    results = []
    for row in rows:
        d = dict(row)
        for field in ("matched_skills", "skill_gaps", "languages", "frameworks", "topics"):
            d[field] = json.loads(d.get(field, "[]"))
        results.append(d)
    return results


def get_peer_count() -> int:
    conn = _get_db()
    count = conn.execute("SELECT COUNT(*) FROM peers").fetchone()[0]
    conn.close()
    return count


def store_embedding(github_username: str, embedding: list[float], metadata: dict):
    """Store or update a peer's embedding in ChromaDB."""
    collection = _get_chroma()
    collection.upsert(
        ids=[github_username.lower()],
        embeddings=[embedding],
        metadatas=[metadata],
    )


def query_similar(embedding: list[float], n: int = 10, exclude_user: str = "") -> list[dict]:
    """Find the n most similar peers by cosine similarity."""
    collection = _get_chroma()
    total = collection.count()
    if total == 0:
        return []

    # Query more than needed so we can filter out the requesting user
    results = collection.query(
        query_embeddings=[embedding],
        n_results=min(n + 1, total),
        include=["metadatas", "distances"],
    )

    matches = []
    for i, uid in enumerate(results["ids"][0]):
        if uid == exclude_user.lower():
            continue
        matches.append({
            "github_username": uid,
            "similarity": 1 - results["distances"][0][i],  # cosine distance → similarity
            "metadata": results["metadatas"][0][i],
        })

    return matches[:n]

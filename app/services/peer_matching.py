"""Multi-step peer matching pipeline:
1. Build profile text → embed with text-embedding-3-small
2. Store in ChromaDB
3. Query ChromaDB for similar peers
4. GPT-4o-mini ranks matches, explains reasoning, suggests collab projects
"""

import json
from openai import AsyncOpenAI
from app.config import get_settings
from app.services.peer_db import (
    upsert_peer, get_peer, store_embedding, query_similar, get_all_peers,
)


def _build_profile_text(peer: dict) -> str:
    """Combine peer fields into a single string for embedding."""
    parts = [
        f"Target role: {peer['target_role']}",
        f"Languages: {', '.join(peer.get('languages', []))}",
        f"Frameworks: {', '.join(peer.get('frameworks', []))}",
        f"Skills: {', '.join(peer.get('matched_skills', []))}",
        f"Skill gaps: {', '.join(peer.get('skill_gaps', []))}",
    ]
    if peer.get("current_projects"):
        parts.append(f"Current projects: {peer['current_projects']}")
    return "\n".join(parts)


async def _get_embedding(text: str) -> list[float]:
    """Get embedding from OpenAI text-embedding-3-small."""
    settings = get_settings()
    client = AsyncOpenAI(api_key=settings.openai_api_key)
    response = await client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return response.data[0].embedding


async def register_peer(
    github_username: str,
    target_role: str,
    contact: str,
    current_projects: str = "",
    matched_skills: list[str] | None = None,
    skill_gaps: list[str] | None = None,
    languages: list[str] | None = None,
    frameworks: list[str] | None = None,
) -> dict:
    """Step 1 & 2: Save peer profile to SQLite, embed and store in ChromaDB."""
    matched_skills = matched_skills or []
    skill_gaps = skill_gaps or []
    languages = languages or []
    frameworks = frameworks or []

    # Save to SQLite
    peer = upsert_peer(
        github_username=github_username,
        target_role=target_role,
        contact=contact,
        current_projects=current_projects,
        matched_skills=matched_skills,
        skill_gaps=skill_gaps,
        languages=languages,
        frameworks=frameworks,
    )

    # Build text and embed
    profile_text = _build_profile_text(peer)
    try:
        embedding = await _get_embedding(profile_text)
        store_embedding(
            github_username=github_username,
            embedding=embedding,
            metadata={
                "target_role": target_role,
                "languages": ", ".join(languages),
                "contact": contact,
            },
        )
    except Exception as e:
        print(f"Embedding failed for {github_username}: {e}")

    return peer


MATCH_PROMPT = """You are a peer matching agent for developers. Analyze the requesting user's profile against potential peer matches and provide detailed match analysis.

## Requesting User
Username: {username}
Target Role: {target_role}
Languages: {languages}
Skills: {matched_skills}
Skill Gaps: {skill_gaps}
Current Projects: {current_projects}

## Potential Peers (ranked by profile similarity)
{peer_profiles}

## Instructions
For each peer, analyze:
1. Why they're a good match (shared goals, complementary skills, similar gaps)
2. What collaboration could look like
3. A specific project they could build together

Respond with ONLY raw JSON (no markdown, no backticks):
{{
  "matches": [
    {{
      "github_username": "peer_username",
      "match_reason": "2-3 sentence explanation of why this is a good match",
      "collaboration_type": "co-builder|mentor|study-partner|complementary",
      "suggested_project": {{
        "title": "Project they could build together",
        "description": "2-3 sentences about what and how",
        "tech_stack": ["tech1", "tech2"]
      }},
      "shared_interests": ["interest1", "interest2"],
      "complementary_skills": ["what one brings that the other needs"]
    }}
  ]
}}"""


async def find_peers(github_username: str, n: int = 5) -> dict:
    """Full multi-step matching pipeline:
    1. Get user's profile and embedding
    2. Query ChromaDB for similar peers
    3. Enrich with SQLite profile data
    4. GPT-4o-mini analyzes and ranks matches
    """
    settings = get_settings()
    user = get_peer(github_username)
    if not user:
        return {"error": "User not registered. Run an analysis first."}

    # Step 1: Get user's embedding
    profile_text = _build_profile_text(user)
    try:
        embedding = await _get_embedding(profile_text)
    except Exception as e:
        return {"error": f"Embedding failed: {e}"}

    # Step 2: Query ChromaDB for similar peers
    candidates = query_similar(embedding, n=n, exclude_user=github_username)
    if not candidates:
        return {"matches": [], "message": "No peers in the pool yet. Be the first!"}

    # Step 3: Enrich candidates with full profile from SQLite
    enriched = []
    for c in candidates:
        peer = get_peer(c["github_username"])
        if peer:
            peer["similarity_score"] = round(c["similarity"], 3)
            enriched.append(peer)

    if not enriched:
        return {"matches": [], "message": "No matching peers found."}

    # Step 4: GPT-4o-mini analyzes matches
    peer_lines = []
    for p in enriched:
        peer_lines.append(
            f"- {p['github_username']} (similarity: {p['similarity_score']})\n"
            f"  Role: {p['target_role']}\n"
            f"  Languages: {', '.join(p.get('languages', []))}\n"
            f"  Skills: {', '.join(p.get('matched_skills', []))}\n"
            f"  Gaps: {', '.join(p.get('skill_gaps', []))}\n"
            f"  Projects: {p.get('current_projects', 'None listed')}\n"
            f"  Contact: {p.get('contact', '')}"
        )

    prompt = MATCH_PROMPT.format(
        username=github_username,
        target_role=user["target_role"],
        languages=", ".join(user.get("languages", [])),
        matched_skills=", ".join(user.get("matched_skills", [])),
        skill_gaps=", ".join(user.get("skill_gaps", [])),
        current_projects=user.get("current_projects", "None listed"),
        peer_profiles="\n\n".join(peer_lines),
    )

    try:
        client = AsyncOpenAI(api_key=settings.openai_api_key)
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=4000,
        )
        text = response.choices[0].message.content or ""
        cleaned = text.replace("```json", "").replace("```", "").strip()
        llm_result = json.loads(cleaned)
    except Exception as e:
        print(f"LLM match analysis failed: {e}")
        # Fallback: return raw similarity matches without LLM analysis
        llm_result = {"matches": []}

    # Merge LLM analysis with profile data
    llm_matches = {m["github_username"]: m for m in llm_result.get("matches", [])}
    final_matches = []

    for p in enriched:
        username = p["github_username"]
        llm_data = llm_matches.get(username, {})
        final_matches.append({
            "github_username": username,
            "target_role": p["target_role"],
            "contact": p.get("contact", ""),
            "current_projects": p.get("current_projects", ""),
            "languages": p.get("languages", []),
            "matched_skills": p.get("matched_skills", []),
            "skill_gaps": p.get("skill_gaps", []),
            "similarity_score": p.get("similarity_score", 0),
            "match_reason": llm_data.get("match_reason", "Similar profile and goals."),
            "collaboration_type": llm_data.get("collaboration_type", "co-builder"),
            "suggested_project": llm_data.get("suggested_project", {}),
            "shared_interests": llm_data.get("shared_interests", []),
            "complementary_skills": llm_data.get("complementary_skills", []),
        })

    return {
        "user": github_username,
        "target_role": user["target_role"],
        "matches": final_matches,
        "pool_size": len(get_all_peers()),
    }

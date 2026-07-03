"""Peer matching pipeline:
1. Build profile text (full) + skills text (what they HAVE) → embed with text-embedding-3-small
2. Store both in ChromaDB (collections: peer_profiles, peer_skills)
3. Retrieval modes:
   - similar:        user's profile embedding vs peer_profiles (same journey)
   - complementary:  user's skill GAPS embedding vs peer_skills (they know what you lack)
   - both:           mix of the two
4. GPT-4o-mini analyzes matches, explains reasoning, suggests collab projects
"""

import json
from cachetools import TTLCache
from openai import AsyncOpenAI
from app.config import get_settings
from app.services.peer_db import (
    upsert_peer, get_peer, store_embedding, query_similar, get_all_peers,
    get_embedded_ids,
)

# Skip re-embedding if peer was embedded within the last hour
_embedding_cache: TTLCache = TTLCache(maxsize=200, ttl=60 * 60)


def _build_profile_text(peer: dict) -> str:
    """Combine peer fields into a single string for embedding."""
    parts = [
        f"Target role: {peer['target_role']}",
        f"Languages: {', '.join(peer.get('languages', []))}",
        f"Frameworks: {', '.join(peer.get('frameworks', []))}",
        f"Skills: {', '.join(peer.get('matched_skills', []))}",
        f"Skill gaps: {', '.join(peer.get('skill_gaps', []))}",
    ]
    if peer.get("topics"):
        parts.append(f"Domains & topics: {', '.join(peer['topics'])}")
    if peer.get("current_projects"):
        parts.append(f"Current projects: {peer['current_projects']}")
    return "\n".join(parts)


def _build_skills_text(peer: dict) -> str:
    """Only what the peer HAS (no gaps) — searched by others' gap queries."""
    parts = [
        f"Languages: {', '.join(peer.get('languages', []))}",
        f"Frameworks: {', '.join(peer.get('frameworks', []))}",
        f"Skills: {', '.join(peer.get('matched_skills', []))}",
    ]
    if peer.get("topics"):
        parts.append(f"Domains & topics: {', '.join(peer['topics'])}")
    return "\n".join(parts)


def _build_gaps_text(peer: dict) -> str:
    """What the peer WANTS to learn — used as the complementary query."""
    return (
        f"Looking for developers experienced in: "
        f"{', '.join(peer.get('skill_gaps', []))}"
    )


async def _get_embeddings(texts: list[str]) -> list[list[float]]:
    """Batch embeddings from OpenAI text-embedding-3-small."""
    settings = get_settings()
    client = AsyncOpenAI(api_key=settings.openai_api_key)
    response = await client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    return [d.embedding for d in response.data]


async def register_peer(
    github_username: str,
    target_role: str,
    contact: str,
    current_projects: str = "",
    matched_skills: list[str] | None = None,
    skill_gaps: list[str] | None = None,
    languages: list[str] | None = None,
    frameworks: list[str] | None = None,
    topics: list[str] | None = None,
) -> dict:
    """Step 1 & 2: Save peer profile to SQLite, embed and store in ChromaDB."""
    matched_skills = matched_skills or []
    skill_gaps = skill_gaps or []
    languages = languages or []
    frameworks = frameworks or []
    topics = topics or []

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
        topics=topics,
    )

    # Embed profile + skills (skip if embedded within TTL)
    cache_key = github_username.lower()
    if cache_key not in _embedding_cache:
        try:
            profile_emb, skills_emb = await _get_embeddings(
                [_build_profile_text(peer), _build_skills_text(peer)]
            )
            metadata = {
                "target_role": target_role,
                "languages": ", ".join(languages),
            }
            store_embedding(
                github_username=github_username,
                embedding=profile_emb,
                metadata=metadata,
                collection_name="peer_profiles",
            )
            store_embedding(
                github_username=github_username,
                embedding=skills_emb,
                metadata=metadata,
                collection_name="peer_skills",
            )
            _embedding_cache[cache_key] = True
        except Exception as e:
            print(f"Embedding failed for {github_username}: {e}")

    return peer


async def _backfill_skills_embeddings() -> None:
    """Embed skills for peers registered before the peer_skills collection existed."""
    embedded = get_embedded_ids("peer_skills")
    missing = [p for p in get_all_peers() if p["github_username"] not in embedded]
    if not missing:
        return
    embeddings = await _get_embeddings([_build_skills_text(p) for p in missing])
    for peer, emb in zip(missing, embeddings):
        store_embedding(
            github_username=peer["github_username"],
            embedding=emb,
            metadata={
                "target_role": peer["target_role"],
                "languages": ", ".join(peer.get("languages", [])),
            },
            collection_name="peer_skills",
        )


MATCH_PROMPT = """You are a peer matching agent for developers. Analyze the requesting user's profile against potential peer matches and provide detailed match analysis.

## Requesting User
Username: {username}
Target Role: {target_role}
Languages: {languages}
Skills: {matched_skills}
Skill Gaps: {skill_gaps}
Current Projects: {current_projects}

## Potential Peers
{peer_profiles}

Each peer has a match type:
- "similar": they are on a similar journey (same goals, overlapping skills and gaps) — good study partners and co-builders
- "complementary": their skills cover the requesting user's skill gaps — good mentors or teammates to learn from while building

## Instructions
For each peer, analyze:
1. Why they're a good match, consistent with their match type (shared goals for similar; which of the user's gaps they cover for complementary)
2. What collaboration could look like
3. A specific project they could build together. For complementary peers, the project should deliberately use the skills the user is missing so they learn them from the peer.

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


async def find_peers(github_username: str, n: int = 5, mode: str = "both") -> dict:
    """Full multi-step matching pipeline:
    1. Embed user's profile (similar) and/or skill gaps (complementary)
    2. Query ChromaDB: profile vs peer_profiles, gaps vs peer_skills
    3. Merge, tag match_type, enrich with SQLite profile data
    4. GPT-4o-mini analyzes and ranks matches

    Modes: "similar" | "complementary" | "both"
    """
    settings = get_settings()
    user = get_peer(github_username)
    if not user:
        return {"error": "User not registered. Run an analysis first."}
    if mode not in ("similar", "complementary", "both"):
        mode = "both"

    want_similar = mode in ("similar", "both")
    want_complementary = mode in ("complementary", "both") and user.get("skill_gaps")

    # Step 1: Embed the query texts (one batched call)
    texts, kinds = [], []
    if want_similar:
        texts.append(_build_profile_text(user))
        kinds.append("similar")
    if want_complementary:
        texts.append(_build_gaps_text(user))
        kinds.append("complementary")
    if not texts:
        return {"matches": [], "message": "No skill gaps on record — run an analysis first."}

    try:
        if want_complementary:
            await _backfill_skills_embeddings()
        embeddings = await _get_embeddings(texts)
    except Exception as e:
        return {"error": f"Embedding failed: {e}"}

    # Step 2: Query the right collection per kind.
    # NOTE: scores from the two collections are not comparable (profile-vs-profile
    # runs higher than gaps-vs-skills), so we merge by per-list rank, not raw score.
    result_lists: dict[str, list[dict]] = {}
    for kind, emb in zip(kinds, embeddings):
        collection = "peer_profiles" if kind == "similar" else "peer_skills"
        results = query_similar(
            emb, n=n, exclude_user=github_username, collection_name=collection
        )
        for c in results:
            c["match_type"] = kind
        result_lists[kind] = results

    # Dedupe peers appearing in both lists: keep the kind where they rank better
    # (tie → complementary, since covering a gap is the more actionable signal).
    if len(result_lists) == 2:
        sim_rank = {c["github_username"]: i for i, c in enumerate(result_lists["similar"])}
        comp_rank = {c["github_username"]: i for i, c in enumerate(result_lists["complementary"])}
        dupes = set(sim_rank) & set(comp_rank)
        for uname in dupes:
            drop_kind = "complementary" if sim_rank[uname] < comp_rank[uname] else "similar"
            result_lists[drop_kind] = [
                c for c in result_lists[drop_kind] if c["github_username"] != uname
            ]

    # Interleave (complementary first) so "both" always shows a mix
    ranked = []
    lists = [result_lists.get("complementary", []), result_lists.get("similar", [])]
    i = 0
    while len(ranked) < n and any(lists):
        for lst in lists:
            if i < len(lst) and len(ranked) < n:
                ranked.append(lst[i])
        if i >= max(len(l) for l in lists):
            break
        i += 1

    if not ranked:
        return {"matches": [], "message": "No peers in the pool yet. Be the first!"}

    # Step 3: Enrich candidates with full profile data
    enriched = []
    for c in ranked:
        peer = get_peer(c["github_username"])
        if peer:
            peer["similarity_score"] = round(c["similarity"], 3)
            peer["match_type"] = c["match_type"]
            enriched.append(peer)

    if not enriched:
        return {"matches": [], "message": "No matching peers found."}

    # Step 4: GPT-4o-mini analyzes matches
    peer_lines = []
    for p in enriched:
        peer_lines.append(
            f"- {p['github_username']} (match type: {p['match_type']}, similarity: {p['similarity_score']})\n"
            f"  Role: {p['target_role']}\n"
            f"  Languages: {', '.join(p.get('languages', []))}\n"
            f"  Topics: {', '.join(p.get('topics', []))}\n"
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
            "match_type": p.get("match_type", "similar"),
            "match_reason": llm_data.get("match_reason", "Similar profile and goals."),
            "collaboration_type": llm_data.get("collaboration_type", "co-builder"),
            "suggested_project": llm_data.get("suggested_project", {}),
            "shared_interests": llm_data.get("shared_interests", []),
            "complementary_skills": llm_data.get("complementary_skills", []),
        })

    return {
        "user": github_username,
        "target_role": user["target_role"],
        "mode": mode,
        "matches": final_matches,
        "pool_size": len(get_all_peers()),
    }

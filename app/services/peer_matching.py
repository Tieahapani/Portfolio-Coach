"""Peer matching pipeline:
1. Build profile text (full) + skills text (what they HAVE) → embed with text-embedding-3-small
2. Store both in ChromaDB (collections: peer_profiles, peer_skills)
3. Retrieval modes:
   - similar:        user's profile embedding vs peer_profiles (same journey)
   - complementary:  user's skill GAPS embedding vs peer_skills (they know what you lack)
4. Gemini 2.5 Flash analyzes matches, explains reasoning, suggests collab projects
"""

from cachetools import TTLCache
from openai import AsyncOpenAI
from app.config import get_settings
from app.services.recommender import GEMINI_BASE_URL, _parse_json
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
1. Why they're a good match, consistent with their match type (shared goals for similar; which of the user's gaps they cover for complementary). Name the SPECIFIC overlapping or complementary skills — never say "similar interests" without naming them.
2. A specific project they could build together, following the project rules below.

## Project rules (STRICT)
- The project must be grounded in BOTH people's actual data: reference their target roles, their listed current projects/topics, and the exact skill names involved. If the user's gap is "Kubernetes", the project must have a concrete Kubernetes component — name it.
- For complementary peers: the project must deliberately force the requesting user to practice their missing skills, with the peer's strengths covering the parts the user can't do yet. State the division of work: who builds what.
- For similar peers: pick a project slightly ABOVE both of their current levels, targeting a gap they SHARE, so they struggle through it together.
- The project must be scoped to something two people can ship a working v1 of in 2-4 weeks. Not a platform, not a startup — a focused tool.
- BANNED: todo apps, generic dashboards, portfolio websites, generic chat apps, "e-commerce platform", "social media app", vague "ML model" projects with no dataset/domain, and any idea that doesn't name a specific domain and dataset/API.
- Every project needs a specific domain hook: a real dataset, a real API, or a real problem from one of their current projects.

BAD example: "Build a machine learning dashboard to visualize data and improve skills."
GOOD example: "Build a CLI that ingests a GitHub repo's Actions logs and predicts flaky tests. Peer sets up the k8s-based runner (their strength, your gap: Kubernetes); you build the log parser in Python. Ship v1 against the peer's existing 'ci-tools' repo."

Respond with ONLY raw JSON (no markdown, no backticks):
{{
  "matches": [
    {{
      "github_username": "peer_username",
      "match_reason": "2-3 sentences naming the specific skills/goals that connect them",
      "collaboration_type": "co-builder|mentor|study-partner|complementary",
      "suggested_project": {{
        "title": "Specific, concrete project name",
        "description": "3-4 sentences: what it does, the specific domain/dataset/API it uses, who builds which part, and which of the user's gap skills it forces them to learn",
        "tech_stack": ["tech1", "tech2"]
      }},
      "shared_interests": ["interest1", "interest2"],
      "complementary_skills": ["what one brings that the other needs"]
    }}
  ]
}}"""


async def find_peers(github_username: str, n: int = 2, mode: str = "similar") -> dict:
    """Full multi-step matching pipeline:
    1. Embed user's profile (similar) or skill gaps (complementary)
    2. Query ChromaDB: profile vs peer_profiles, or gaps vs peer_skills
    3. Tag match_type, enrich with SQLite profile data
    4. Gemini 2.5 Flash analyzes and ranks matches

    Modes: "similar" | "complementary"
    """
    settings = get_settings()
    user = get_peer(github_username)
    if not user:
        return {"error": "User not registered. Run an analysis first."}
    if mode not in ("similar", "complementary"):
        mode = "similar"

    # Step 1: Embed the query text
    if mode == "similar":
        query_text = _build_profile_text(user)
        collection = "peer_profiles"
    else:
        if not user.get("skill_gaps"):
            return {"matches": [], "message": "No skill gaps on record — run an analysis first."}
        query_text = _build_gaps_text(user)
        collection = "peer_skills"

    try:
        if mode == "complementary":
            await _backfill_skills_embeddings()
        embeddings = await _get_embeddings([query_text])
    except Exception as e:
        return {"error": f"Embedding failed: {e}"}

    # Step 2: Query the matching collection
    ranked = query_similar(
        embeddings[0], n=n, exclude_user=github_username, collection_name=collection
    )
    for c in ranked:
        c["match_type"] = mode

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

    # Step 4: Gemini 2.5 Flash analyzes matches
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
        client = AsyncOpenAI(
            api_key=settings.effective_gemini_key,
            base_url=GEMINI_BASE_URL,
        )
        response = await client.chat.completions.create(
            model="gemini-2.5-flash",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=8000,
            reasoning_effort="low",  # small thinking budget; tokens count against max_tokens
        )
        text = response.choices[0].message.content or ""
        llm_result = _parse_json(text)
        if llm_result is None:
            raise ValueError(f"Unparseable LLM output: {text[:200]}")
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

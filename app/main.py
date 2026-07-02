import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request, Cookie
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from openinference.instrumentation.openai import OpenAIInstrumentor
from openinference.instrumentation.google_genai import GoogleGenAIInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from app.config import get_settings

# ── Phoenix LLM Observability ──
PHOENIX_ENDPOINT = os.getenv("PHOENIX_ENDPOINT", "http://127.0.0.1:6006/v1/traces")
tracer_provider = TracerProvider()
tracer_provider.add_span_processor(
    SimpleSpanProcessor(OTLPSpanExporter(endpoint=PHOENIX_ENDPOINT))
)
OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
GoogleGenAIInstrumentor().instrument(tracer_provider=tracer_provider)
from app.schemas import (
    AnalyzeRequest,
    AnalysisResult,
    GitHubProfile,
    LearningResource,
    MarketData,
    ProjectRecommendation,
)
from app.services.github_service import analyze_github
from app.services.market_research import research_market
from app.services.recommender import generate_recommendations
from app.services.tracker import (
    register_user, unregister_user, get_tracked_users,
    get_user_insights, check_user, run_tracker_cycle,
)
from app.services.peer_matching import register_peer, find_peers
from app.services.peer_db import get_peer, get_peer_count
from app.services.auth import (
    exchange_code_for_user, upsert_user, get_user,
    update_user_profile, create_token, verify_token,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    print("=" * 50)
    print("Portfolio Coach API")
    print(f"  OpenAI:  {'✓ configured' if settings.has_openai else '✗ not set'}")
    print(f"  GitHub:  {'✓ token set' if settings.has_github_token else '○ public API'}")
    print(f"  Phoenix: → {PHOENIX_ENDPOINT}")
    print("=" * 50)

    if not settings.has_openai:
        print("⚠️  WARNING: No OPENAI_API_KEY configured in .env")

    # Start background tracker (every 6 hours)
    tracker_task = asyncio.create_task(_tracker_loop())
    print("  Tracker: ✓ background loop started (6h interval)")

    yield

    tracker_task.cancel()
    try:
        await tracker_task
    except asyncio.CancelledError:
        pass


TRACKER_INTERVAL = 6 * 60 * 60  # 6 hours


async def _tracker_loop():
    """Background loop that checks all tracked users periodically."""
    while True:
        await asyncio.sleep(TRACKER_INTERVAL)
        try:
            await run_tracker_cycle()
        except Exception as e:
            print(f"Tracker cycle error: {e}")


app = FastAPI(
    title="Portfolio Coach API",
    description="Analyzes GitHub profiles against job market demands and recommends projects to build.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


STATIC_DIR = Path(__file__).parent / "static"


@app.get("/", response_class=FileResponse)
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/analyze", response_class=FileResponse)
async def analyze_page():
    return FileResponse(STATIC_DIR / "analyze.html")


@app.get("/tracker", response_class=FileResponse)
async def tracker_page():
    return FileResponse(STATIC_DIR / "tracker.html")


@app.get("/peers", response_class=FileResponse)
async def peers_page():
    return FileResponse(STATIC_DIR / "peers.html")


# ── Auth ──

@app.get("/auth/login")
async def auth_login():
    """Redirect to GitHub OAuth."""
    settings = get_settings()
    return RedirectResponse(
        f"https://github.com/login/oauth/authorize"
        f"?client_id={settings.github_client_id}"
        f"&redirect_uri=http://localhost:{settings.port}/auth/callback"
    )


@app.get("/auth/callback")
async def auth_callback(code: str):
    """Handle GitHub OAuth callback."""
    try:
        github_user = await exchange_code_for_user(code)
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"OAuth failed: {e}")

    # Create or update user in DB
    user = upsert_user(
        github_username=github_user["github_username"],
        name=github_user["name"],
        avatar_url=github_user["avatar_url"],
    )

    # Set JWT cookie and redirect to analyze page
    token = create_token(github_user["github_username"])
    response = RedirectResponse(url="/analyze", status_code=302)
    response.set_cookie(
        key="session",
        value=token,
        httponly=True,
        max_age=30 * 24 * 60 * 60,  # 30 days
        samesite="lax",
    )
    return response


@app.get("/auth/me")
async def auth_me(session: str = Cookie(default="")):
    """Get current logged-in user profile."""
    if not session:
        return {"authenticated": False}
    username = verify_token(session)
    if not username:
        return {"authenticated": False}
    user = get_user(username)
    if not user:
        return {"authenticated": False}
    return {"authenticated": True, "user": user}


@app.put("/auth/profile")
async def auth_update_profile(req: dict, session: str = Cookie(default="")):
    """Update user's contact, target role, or current projects."""
    if not session:
        raise HTTPException(status_code=401, detail="Not authenticated")
    username = verify_token(session)
    if not username:
        raise HTTPException(status_code=401, detail="Invalid session")
    user = update_user_profile(
        github_username=username,
        contact=req.get("contact", ""),
        target_role=req.get("target_role", ""),
        current_projects=req.get("current_projects", ""),
    )
    return {"user": user}


@app.get("/auth/logout")
async def auth_logout():
    """Clear session cookie."""
    response = RedirectResponse(url="/", status_code=302)
    response.delete_cookie("session")
    return response


# ── Health ──

@app.get("/health")
async def health():
    settings = get_settings()
    return {
        "status": "ok",
        "gemini": settings.has_gemini,
        "anthropic": settings.has_anthropic,
        "github_token": settings.has_github_token,
    }


# ── GitHub Profile ──

@app.get("/api/github/{username}", response_model=GitHubProfile)
async def get_github_profile(username: str, readmes: bool = False):
    """Fetch and analyze a GitHub user's profile."""
    try:
        return await analyze_github(username, fetch_readmes=readmes)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"GitHub error: {e}")


# ── Market Research ──

@app.get("/api/market", response_model=MarketData)
async def get_market_data(role: str, mode: str = "fast"):
    """Search job market for in-demand skills."""
    try:
        return await research_market(role, mode=mode)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Market research error: {e}")


# ── Full Analysis (main endpoint) ──

@app.post("/api/analyze", response_model=AnalysisResult)
async def analyze(req: AnalyzeRequest):
    """
    Full pipeline: GitHub analysis → market research → project recommendations.

    Modes:
    - fast:     Single LLM call (Gemini preferred), no Indeed MCP, no READMEs
    - thorough: Parallel web + Indeed search, README analysis, separate recommendation call
    """
    settings = get_settings()

    if not settings.has_openai:
        raise HTTPException(
            status_code=500,
            detail="No LLM API key configured. Set OPENAI_API_KEY in .env",
        )

    # Step 1: GitHub profile
    try:
        profile = await analyze_github(
            req.github_username,
            fetch_readmes=(req.mode == "thorough"),
        )
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"GitHub user not found: {e}")

    if not profile.repos:
        raise HTTPException(status_code=404, detail="No public repos found for this user.")

    # Step 2: Market research
    try:
        market = await research_market(req.target_role, mode=req.mode)
    except Exception as e:
        # Non-fatal: continue with empty market data
        market = MarketData(sources=["fallback"])

    # Step 3: Recommendations
    try:
        rec_data, model_used = await generate_recommendations(
            profile, market, req.target_role
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation error: {e}")

    # Build response
    return AnalysisResult(
        github_profile=profile,
        market_data=market,
        profile_summary=rec_data.get("profile_summary", ""),
        skill_gaps=rec_data.get("skill_gaps", []),
        matched_skills=rec_data.get("matched_skills", []),
        projects=[
            ProjectRecommendation(
                **{**p, "resources": [LearningResource(**r) for r in p.get("resources", [])]}
            )
            for p in rec_data.get("projects", [])
        ],
        overall_strategy=rec_data.get("overall_strategy", ""),
        model_used=model_used,
        raw_text=None,
    )


# ── Tracker ──

@app.post("/api/tracker/register")
async def tracker_register(username: str, target_role: str):
    """Register a user for background commit tracking."""
    user = register_user(username, target_role)
    # Run first check immediately
    insight = await check_user(username)
    return {"user": user, "latest_insight": insight}


@app.delete("/api/tracker/{username}")
async def tracker_unregister(username: str):
    """Stop tracking a user."""
    unregister_user(username)
    return {"status": "removed"}


@app.get("/api/tracker")
async def tracker_list():
    """List all tracked users."""
    return {"users": get_tracked_users()}


@app.get("/api/tracker/{username}")
async def tracker_insights(username: str):
    """Get insights for a tracked user."""
    data = get_user_insights(username)
    if not data:
        raise HTTPException(status_code=404, detail="User not tracked")
    return data


@app.post("/api/tracker/{username}/check")
async def tracker_check_now(username: str):
    """Manually trigger a check for a user."""
    insight = await check_user(username)
    if not insight:
        raise HTTPException(status_code=404, detail="User not tracked or check failed")
    return insight


# ── Peers ──

@app.post("/api/peers/register")
async def peer_register(req: dict):
    """Register or update a peer profile (called automatically after analysis)."""
    try:
        peer = await register_peer(
            github_username=req["github_username"],
            target_role=req["target_role"],
            contact=req["contact"],
            current_projects=req.get("current_projects", ""),
            matched_skills=req.get("matched_skills", []),
            skill_gaps=req.get("skill_gaps", []),
            languages=req.get("languages", []),
            frameworks=req.get("frameworks", []),
            topics=req.get("topics", []),
        )
        return {"status": "registered", "peer": peer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Peer registration failed: {e}")


@app.get("/api/peers/match/{username}")
async def peer_match(username: str):
    """Find matching peers for a user."""
    result = await find_peers(username)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@app.get("/api/peers/{username}")
async def peer_profile(username: str):
    """Get a peer's profile."""
    peer = get_peer(username)
    if not peer:
        raise HTTPException(status_code=404, detail="Peer not found")
    return peer


@app.get("/api/peers")
async def peer_list():
    """Get peer pool stats."""
    return {"pool_size": get_peer_count()}


# ── Run ──

if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=True)

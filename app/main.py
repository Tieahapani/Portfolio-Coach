import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # make .env vars visible to os.getenv (e.g. PHOENIX_*)
from fastapi import FastAPI, HTTPException, Request, Cookie
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from openinference.instrumentation.openai import OpenAIInstrumentor
from openinference.instrumentation.google_genai import GoogleGenAIInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from app.config import get_settings

# ── Phoenix LLM Observability ──
# Local default: http://127.0.0.1:6006/v1/traces (run `phoenix serve` locally).
# Production: set PHOENIX_ENDPOINT + PHOENIX_API_KEY in .env for Phoenix Cloud.
PHOENIX_ENDPOINT = os.getenv("PHOENIX_ENDPOINT", "http://127.0.0.1:6006/v1/traces")
PHOENIX_API_KEY = os.getenv("PHOENIX_API_KEY", "")
_headers = {"api_key": PHOENIX_API_KEY, "authorization": f"Bearer {PHOENIX_API_KEY}"} if PHOENIX_API_KEY else None
tracer_provider = TracerProvider()
tracer_provider.add_span_processor(
    BatchSpanProcessor(OTLPSpanExporter(endpoint=PHOENIX_ENDPOINT, headers=_headers))
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
from app.services.peer_matching import register_peer, find_peers
from app.services.peer_db import get_peer, get_peer_count
from app.services.project_db import (
    add_project, get_project, update_project, delete_project,
)
from app.services.project_tracking import slugify_title, refresh_projects
from app.services.progress_agent import coach_project
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

    yield


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

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=FileResponse)
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/analyze", response_class=FileResponse)
async def analyze_page():
    return FileResponse(STATIC_DIR / "analyze.html")


@app.get("/tracker")
async def tracker_page():
    """Old tracker page — merged into the Progress page."""
    return RedirectResponse("/projects", status_code=301)


@app.get("/peers", response_class=FileResponse)
async def peers_page():
    return FileResponse(STATIC_DIR / "peers.html")


@app.get("/projects", response_class=FileResponse)
async def projects_page():
    return FileResponse(STATIC_DIR / "projects.html")


# ── Auth ──

@app.get("/auth/login")
async def auth_login(request: Request):
    """Redirect to GitHub OAuth."""
    settings = get_settings()
    redirect_uri = str(request.base_url).rstrip("/") + "/auth/callback"
    return RedirectResponse(
        f"https://github.com/login/oauth/authorize"
        f"?client_id={settings.github_client_id}"
        f"&redirect_uri={redirect_uri}"
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
async def peer_match(username: str, mode: str = "similar"):
    """Find matching peers for a user. mode: similar | complementary"""
    result = await find_peers(username, mode=mode)
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


# ── Tracked Projects ──

@app.post("/api/projects/accept")
async def project_accept(req: dict):
    """Accept a recommended project: store it and suggest a repo name."""
    try:
        title = req["title"]
        project = add_project(
            github_username=req["github_username"],
            title=title,
            description=req.get("description", ""),
            tech_stack=req.get("tech_stack", []),
            skills_gained=req.get("skills_gained", []),
            difficulty=req.get("difficulty", "Intermediate"),
            suggested_repo_name=slugify_title(title),
        )
        return {"status": "tracking", "project": project}
    except KeyError as e:
        raise HTTPException(status_code=422, detail=f"Missing field: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not track project: {e}")


@app.get("/api/projects/{username}")
async def project_list(username: str):
    """List a user's tracked projects, refreshing repo detection + activity."""
    return {"projects": await refresh_projects(username)}


@app.post("/api/projects/{project_id}/link")
async def project_link(project_id: int, req: dict):
    """Manually link a repo to a tracked project."""
    project = get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    repo = (req.get("repo_name") or "").strip().strip("/")
    if not repo:
        raise HTTPException(status_code=422, detail="repo_name required")
    return {"project": update_project(project_id, linked_repo=repo)}


@app.post("/api/projects/{project_id}/complete")
async def project_complete(project_id: int):
    """Mark a tracked project as completed."""
    project = get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return {"project": update_project(project_id, status="completed")}


@app.post("/api/projects/{project_id}/coach")
async def project_coach(project_id: int):
    """Run the Progress Coach agent on a tracked project."""
    project = get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    if not project.get("linked_repo"):
        raise HTTPException(status_code=400, detail="No repo linked yet — create the suggested repo first")
    return {"coaching": await coach_project(project), "project": project}


@app.delete("/api/projects/{project_id}")
async def project_delete(project_id: int):
    """Stop tracking a project."""
    if not delete_project(project_id):
        raise HTTPException(status_code=404, detail="Project not found")
    return {"status": "deleted"}


# ── Run ──

if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=True)

# Portfolio Coach API

Analyzes your GitHub profile against real job market demands and recommends high-impact projects to build next.

## Architecture

```
Client (React / curl)
  │
  ▼
FastAPI Backend (/api/analyze)
  │
  ├─ Step 1: GitHub API
  │   └─ Fetch repos, detect languages/frameworks, read READMEs
  │
  ├─ Step 2: Market Research (parallel in thorough mode)
  │   ├─ Gemini 2.5 Flash + Google Search grounding  ← primary (cheap + fast)
  │   └─ Claude Sonnet + Indeed MCP server            ← real Indeed job data
  │
  └─ Step 3: Project Recommendations
      └─ Gemini or Claude generates gap analysis + 4 project ideas
```

## Quick Start

```bash
# 1. Clone and enter
cd portfolio-coach

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure API keys
cp .env.example .env
# Edit .env with your keys:
#   GEMINI_API_KEY   → free at aistudio.google.com
#   ANTHROPIC_API_KEY → for Claude + Indeed MCP
#   GITHUB_TOKEN     → optional, for private repos

# 5. Run
python -m app.main
# or
uvicorn app.main:app --reload
```

Server starts at `http://localhost:8000`. API docs at `http://localhost:8000/docs`.

## API Endpoints

### `POST /api/analyze` — Full pipeline

```bash
curl -X POST http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "github_username": "Tieahapani",
    "target_role": "AI Engineer Intern",
    "mode": "fast"
  }'
```

**Modes:**
| Mode | Speed | Data Sources | Cost |
|------|-------|-------------|------|
| `fast` | ~5-10s | Gemini + Google Search | ~$0.001 |
| `thorough` | ~30-60s | Gemini + Google Search + Claude + Indeed MCP | ~$0.05 |

### `GET /api/github/{username}` — GitHub profile only

```bash
curl http://localhost:8000/api/github/Tieahapani?readmes=true
```

### `GET /api/market?role=...` — Market research only

```bash
curl "http://localhost:8000/api/market?role=AI+Engineer+Intern&mode=fast"
```

### `GET /health` — Check configured services

```bash
curl http://localhost:8000/health
```

## Response Schema

```json
{
  "github_profile": {
    "repos": [...],
    "languages": ["Python", "JavaScript"],
    "frameworks": ["FastAPI", "LangChain", "React"],
    "total_repos": 25
  },
  "market_data": {
    "market_skills": ["Python", "PyTorch", "RAG", ...],
    "trending_tools": ["LangGraph", "vLLM", ...],
    "sample_jobs": [{"title": "...", "company": "...", "key_skills": [...]}],
    "sources": ["Web Search", "Indeed"]
  },
  "profile_summary": "Strong Python developer with RAG experience...",
  "skill_gaps": ["MLOps", "Model fine-tuning", ...],
  "matched_skills": ["Python", "LangChain", "FastAPI", ...],
  "projects": [
    {
      "title": "Production RAG Pipeline with Evaluation",
      "description": "Build a full RAG system with...",
      "difficulty": "Intermediate",
      "skills_gained": ["MLOps", "RAGAS", "Docker"],
      "tech_stack": ["LangChain", "FastAPI", "ChromaDB", "Docker"],
      "why": "Bridges your RAG knowledge to production deployment...",
      "build_time_estimate": "2-3 weeks"
    }
  ],
  "overall_strategy": "Focus on production ML skills...",
  "model_used": "Gemini 2.5 Flash"
}
```

## Cost Comparison

| Provider | Input | Output | Per analysis (fast) |
|----------|-------|--------|-------------------|
| Gemini 2.5 Flash | $0.15/1M | $0.60/1M | ~$0.001 |
| Claude Sonnet 4.6 | $3.00/1M | $15.00/1M | ~$0.03 |

Gemini is ~20-30x cheaper per request.

## Project Structure

```
portfolio-coach/
├── app/
│   ├── main.py              # FastAPI app + endpoints
│   ├── config.py             # Settings from .env
│   ├── schemas.py            # Pydantic models
│   └── services/
│       ├── github_service.py  # GitHub API + skill detection
│       ├── market_research.py # Gemini/Claude web search + Indeed MCP
│       └── recommender.py     # Project recommendation generation
├── requirements.txt
├── .env.example
└── README.md
```

## Deployment

```bash
# Docker
docker build -t portfolio-coach .
docker run -p 8000:8000 --env-file .env portfolio-coach

# Render / Railway / Fly.io
# Set env vars in dashboard, deploy from GitHub
```

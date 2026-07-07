# Portfolio Coach

**Live demo:** [portfolio-coach.duckdns.org](https://portfolio-coach.duckdns.org)

Portfolio Coach analyzes your GitHub profile against real job market demand and tells you exactly what to build next — then tracks whether you actually build it.

## The Problem

Most developers build portfolio projects blindly. They follow tutorials, clone popular ideas, and end up with repos that look like everyone else's — while job postings ask for skills their portfolio never demonstrates.

There's a gap between **what you build** and **what gets you hired**. Portfolio Coach closes that gap in four steps:

1. **Analyze** — reads your public GitHub repos and compares them against live job postings for your target role
2. **Recommend** — suggests specific projects (with tech stacks, difficulty, and time estimates) that fill your skill gaps, or ways to level up repos you already have
3. **Track** — once you accept a project, it watches your commits and an AI coach reviews your actual code progress against the plan
4. **Connect** — matches you with peers on a similar journey (or with complementary skills) and suggests a project to build together

## Features

| Feature | What it does |
|---------|-------------|
| **Portfolio Analysis** | Maps your repos, languages, and frameworks against skills employers are hiring for right now |
| **Market Signals** | Pulls live job posting data to surface in-demand skills and trending tools for your target role |
| **Project Tracker** | Accept a recommended project, and the app auto-detects when you create the repo, tracks your commit activity (active / stalled / completed), and shows who you built it with — collaborators are detected automatically |
| **Progress Coach** | An AI agent that inspects your repo — file tree, README, commit history — and gives an honest assessment: what's built, what's missing, and your next three commits. It remembers its last assessment and tells you what changed since ("you did 2 of the 3 commits I suggested") |
| **Peer Matching** | Finds the top 2 developers most similar to you (accountability partners) or complementary to you (they're strong where you're weak), with a concrete collaboration project idea |

## How It Works

```
You enter: GitHub username + target role (e.g. "AI Engineer Intern")
     │
     ▼
┌─ FastAPI Backend ─────────────────────────────────────────────┐
│                                                               │
│  1. GitHub API → your repos, languages, frameworks, READMEs   │
│  2. Market research → live job postings + web search          │
│  3. Gemini 2.5 Flash → gap analysis + project recommendations │
│                                                               │
│  Accept a project → tracked in SQLite                         │
│  Create the repo → auto-detected, commits + collaborators     │
│  refreshed by a background loop every 30 minutes              │
│  Click "Coach me" → AI agent reads your repo, compares it     │
│  against its last assessment, and reports what changed        │
│                                                               │
│  Peer matching → your profile is embedded as a vector and     │
│  searched against other users (ChromaDB vector database)      │
└───────────────────────────────────────────────────────────────┘
```

## Key Architecture Decisions

**One cheap, fast model for everything.** All LLM work runs on Gemini 2.5 Flash (~$0.001 per analysis) through its OpenAI-compatible API. We originally mixed GPT-4o-mini and Gemini, benchmarked them side by side, and standardized on Gemini — same quality for our use case at a fraction of the cost. OpenAI is only used for text embeddings (peer matching).

**The Progress Coach is a real agent, not a single prompt.** It uses function calling: the model decides which tools to use (fetch file tree, read README, list commits), inspects the repo in a loop, and only then writes its assessment. The loop is capped at 6 rounds, and if anything fails, a graceful fallback keeps the UI working.

**Peer matching is vector search, not keyword matching.** Each user's profile (skills, gaps, target role) is converted into embeddings and stored in ChromaDB. "Similar" mode searches for profiles near yours; "complementary" mode searches for people whose *skills* match your *gaps*.

**No webhooks — a background job instead.** Every 30 minutes, the app quietly checks each tracked project for new commits, newly created repos, and collaborators. That means the Progress page loads instantly from the local database instead of waiting on GitHub. The only time we check GitHub live is right after you link a repo — when freshness actually matters.

**The coach has memory.** Every assessment is saved. The next time you ask for coaching, the AI sees what it told you last time and reports what actually changed — did you do the commits it suggested, or not? That turns it from a one-time reviewer into an accountability partner.

**Boring, reliable persistence.** SQLite for users, peers, tracked projects, and coach history. Expensive AI results are cached and only recomputed when something changes (a new commit automatically triggers a fresh coach analysis). No database server to run or break.

**You can only change your own stuff.** Any action that modifies data — tracking a project, running the coach, registering as a peer — requires signing in with GitHub, and only works on your own data. The AI features are also rate-limited, so a stranger can't run up the API bill. Login sessions are cryptographically signed so they can't be faked.

**Plain HTML/JS frontend.** No React, no build step. Five static pages served directly by FastAPI. Easier to deploy, nothing to compile, and fast on a $6/month server.

## Failures & How We Handled Them

**1. Gemini kept returning truncated, broken JSON.**
Analysis and peer matching randomly failed with "Unterminated string" errors. Root cause: Gemini's internal "thinking" tokens count against the output token limit on the OpenAI-compatible endpoint — the model spent its budget thinking and got cut off mid-answer. Fix: explicitly control the reasoning effort per feature (off for structured extraction, low for judgment tasks), raise the token ceiling, and add a robust JSON parser that extracts the answer even from messy output.

**2. Peer collaboration ideas were embarrassingly generic.**
Early versions suggested "build a full-stack task manager together" to everyone. Instead of switching to a bigger, pricier model, we fixed the prompt: banned a list of clichéd project ideas, added bad-vs-good examples, and required a clear division of work based on each person's actual skills. Prompt engineering solved what looked like a model-quality problem.

**3. Deploys caused 502 errors on our 1GB server.**
After each deploy the site appeared down. The vector database (ChromaDB) takes several seconds to load on a small machine, so health checks fired before the app finished booting. Fix: understand the startup sequence instead of panicking — wait for boot, then verify. Also hit an OS-level snag: the server's SQLite was too old for ChromaDB, solved by swapping in a modern build at import time.

**4. A "both" mode in peer matching that tried to do too much.**
Merging similar + complementary results in one request produced worse output than either mode alone. We removed it. Two clear modes beat one confusing one — cutting a feature was the right call.

**5. Collaborator detection worked — but GitHub showed no collaborators.**
We added "built with @user" badges by reading GitHub's contributors API, then tested on a real two-person repo and got nothing. Root cause: both developers were committing with unconfigured git emails (`user@machine.local`), so GitHub couldn't attribute the commits to any account — which also silently broke commit counting. The code was right; the data was wrong. Lesson: features that depend on external data need a test against real-world messy data, not just a clean happy path.

## Tech Stack

- **Backend:** Python, FastAPI
- **AI:** Gemini 2.5 Flash (analysis, recommendations, coaching agent, peer matching), OpenAI text-embedding-3-small (embeddings)
- **Storage:** SQLite + ChromaDB (vector search)
- **Frontend:** Vanilla HTML/CSS/JS
- **Auth:** GitHub OAuth + JWT sessions
- **Deployment:** DigitalOcean droplet, systemd, HTTPS via DuckDNS

## Quick Start

```bash
git clone https://github.com/Tieahapani/Portfolio-Coach.git
cd portfolio-coach

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# Fill in:
#   GEMINI_API_KEY  → free at aistudio.google.com
#   OPENAI_API_KEY  → for peer-matching embeddings
#   GITHUB_TOKEN    → raises GitHub API rate limits
#   JWT_SECRET      → python -c "import secrets; print(secrets.token_hex(32))"

uvicorn app.main:app --reload
```

Open `http://localhost:8000`. Interactive API docs at `/docs`.

## Project Structure

```
portfolio-coach/
├── app/
│   ├── main.py                    # FastAPI app + all endpoints
│   ├── config.py                  # Settings from .env
│   ├── schemas.py                 # Pydantic models
│   ├── services/
│   │   ├── github_service.py      # GitHub API + skill detection
│   │   ├── market_research.py     # Live job market research
│   │   ├── recommender.py         # Gap analysis + project recommendations
│   │   ├── project_db.py          # Tracked projects (SQLite)
│   │   ├── project_tracking.py    # Repo auto-detect + commit activity
│   │   ├── progress_agent.py      # Function-calling Progress Coach agent
│   │   ├── peer_db.py             # Peer profiles (SQLite + ChromaDB)
│   │   ├── peer_matching.py       # Vector search + match analysis
│   │   └── auth.py                # GitHub OAuth + JWT
│   └── static/                    # index, analyze, peers, projects
├── requirements.txt
└── README.md
```

---

Built to bridge the gap between what you build and what gets you hired.

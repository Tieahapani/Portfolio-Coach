import httpx
import asyncio
from app.config import get_settings
from app.schemas import RepoInfo, GitHubProfile

FRAMEWORK_PATTERNS: dict[str, str] = {
    "react": "React", "nextjs": "Next.js", "vue": "Vue", "angular": "Angular",
    "svelte": "Svelte", "django": "Django", "flask": "Flask", "fastapi": "FastAPI",
    "express": "Express", "nestjs": "NestJS", "spring": "Spring Boot",
    "tensorflow": "TensorFlow", "pytorch": "PyTorch", "langchain": "LangChain",
    "llamaindex": "LlamaIndex", "streamlit": "Streamlit", "gradio": "Gradio",
    "flutter": "Flutter", "react-native": "React Native",
    "docker": "Docker", "kubernetes": "Kubernetes", "terraform": "Terraform",
    "aws": "AWS", "gcp": "GCP", "azure": "Azure",
    "mongodb": "MongoDB", "postgres": "PostgreSQL", "redis": "Redis",
    "graphql": "GraphQL", "prisma": "Prisma", "supabase": "Supabase",
    "firebase": "Firebase", "openai": "OpenAI API", "anthropic": "Anthropic API",
    "huggingface": "HuggingFace", "chromadb": "ChromaDB", "pinecone": "Pinecone",
    "langgraph": "LangGraph", "crewai": "CrewAI", "autogen": "AutoGen",
    "rag": "RAG", "whisper": "Whisper", "bedrock": "AWS Bedrock",
    "vercel": "Vercel", "render": "Render",
    "tailwind": "Tailwind CSS", "material-ui": "Material UI",
}


def _build_headers() -> dict:
    settings = get_settings()
    headers = {"Accept": "application/vnd.github.mercy-preview+json"}
    if settings.has_github_token:
        headers["Authorization"] = f"token {settings.github_token}"
    return headers


async def fetch_repos(username: str) -> list[dict]:
    """Fetch all public repos for a GitHub user."""
    headers = _build_headers()
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(
            f"https://api.github.com/users/{username}/repos",
            params={"per_page": 100, "sort": "updated"},
            headers=headers,
        )
        resp.raise_for_status()
        return resp.json()


async def fetch_readme(username: str, repo_name: str) -> str:
    """Fetch README content for a single repo."""
    headers = _build_headers()
    headers["Accept"] = "application/vnd.github.raw+json"
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            resp = await client.get(
                f"https://api.github.com/repos/{username}/{repo_name}/readme",
                headers=headers,
            )
            if resp.status_code == 200:
                return resp.text[:800]
        except Exception:
            pass
    return ""


def detect_skills(repos: list[dict]) -> tuple[list[str], list[str], list[str]]:
    """Detect languages, frameworks, and topics from repo metadata."""
    lang_count: dict[str, int] = {}
    topics: set[str] = set()
    frameworks: set[str] = set()

    for repo in repos:
        lang = repo.get("language")
        if lang:
            lang_count[lang] = lang_count.get(lang, 0) + 1

        for topic in repo.get("topics", []):
            topics.add(topic)
            lower = topic.lower()
            if lower in FRAMEWORK_PATTERNS:
                frameworks.add(FRAMEWORK_PATTERNS[lower])

        desc = f"{repo.get('description', '')} {repo.get('name', '')}".lower()
        for key, label in FRAMEWORK_PATTERNS.items():
            if key in desc:
                frameworks.add(label)

    languages = [l for l, _ in sorted(lang_count.items(), key=lambda x: -x[1])]
    return languages, sorted(frameworks), sorted(topics)


async def analyze_github(username: str, fetch_readmes: bool = False) -> GitHubProfile:
    """Full GitHub profile analysis."""
    raw_repos = await fetch_repos(username)
    non_fork = [r for r in raw_repos if not r.get("fork")]

    languages, frameworks, topics = detect_skills(non_fork)

    # Optionally fetch READMEs for top repos
    readmes: dict[str, str] = {}
    if fetch_readmes:
        top_repos = non_fork[:6]
        tasks = [fetch_readme(username, r["name"]) for r in top_repos]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for repo, readme in zip(top_repos, results):
            if isinstance(readme, str) and readme:
                readmes[repo["name"]] = readme

    repos = [
        RepoInfo(
            name=r["name"],
            description=r.get("description"),
            language=r.get("language"),
            topics=r.get("topics", []),
            stars=r.get("stargazers_count", 0),
            readme_excerpt=readmes.get(r["name"], ""),
        )
        for r in non_fork
    ]

    return GitHubProfile(
        repos=repos,
        languages=languages,
        frameworks=frameworks,
        topics=topics,
        total_repos=len(non_fork),
    )

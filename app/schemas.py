from pydantic import BaseModel
from typing import Optional


# ── Requests ──

class AnalyzeRequest(BaseModel):
    github_username: str
    target_role: str
    mode: str = "fast"  # "fast" or "thorough"


# ── GitHub ──

class RepoInfo(BaseModel):
    name: str
    description: str | None = None
    language: str | None = None
    topics: list[str] = []
    stars: int = 0
    readme_excerpt: str = ""


class GitHubProfile(BaseModel):
    repos: list[RepoInfo]
    languages: list[str] = []
    frameworks: list[str] = []
    topics: list[str] = []
    total_repos: int = 0


# ── Market ──

class JobPosting(BaseModel):
    title: str = ""
    company: str = ""
    key_skills: list[str] = []


class MarketData(BaseModel):
    market_skills: list[str] = []
    trending_tools: list[str] = []
    sample_jobs: list[JobPosting] = []
    industry_trends: str = ""
    sources: list[str] = []


# ── Recommendations ──

class LearningResource(BaseModel):
    title: str
    type: str = ""  # "course", "video", "docs", "tutorial", "book"
    platform: str = ""  # "YouTube", "Coursera", "freeCodeCamp", etc.
    url: str = ""
    why: str = ""  # why this resource is relevant


class ProjectRecommendation(BaseModel):
    title: str
    description: str
    difficulty: str = "Intermediate"
    skills_gained: list[str] = []
    tech_stack: list[str] = []
    why: str = ""
    build_time_estimate: str = ""
    resources: list[LearningResource] = []


class AnalysisResult(BaseModel):
    github_profile: GitHubProfile
    market_data: MarketData
    profile_summary: str = ""
    skill_gaps: list[str] = []
    matched_skills: list[str] = []
    projects: list[ProjectRecommendation] = []
    overall_strategy: str = ""
    model_used: str = ""
    raw_text: str | None = None

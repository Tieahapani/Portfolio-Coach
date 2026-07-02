from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    gemini_api_key: str = ""
    google_api_key: str = ""
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    groq_api_key: str = ""
    github_token: str = ""
    github_client_id: str = ""
    github_client_secret: str = ""
    jwt_secret: str = "change-me-in-production"
    host: str = "0.0.0.0"
    port: int = 8000

    @property
    def has_gemini(self) -> bool:
        return len(self.gemini_api_key) > 0 or len(self.google_api_key) > 0

    @property
    def effective_gemini_key(self) -> str:
        return self.gemini_api_key or self.google_api_key

    @property
    def has_anthropic(self) -> bool:
        return len(self.anthropic_api_key) > 0

    @property
    def has_openai(self) -> bool:
        return len(self.openai_api_key) > 0

    @property
    def has_groq(self) -> bool:
        return len(self.groq_api_key) > 0

    @property
    def has_github_token(self) -> bool:
        return len(self.github_token) > 0

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


@lru_cache
def get_settings() -> Settings:
    return Settings()

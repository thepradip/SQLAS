from __future__ import annotations
"""Application configuration from environment variables.

Author: Pradip Tivhale
"""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Azure OpenAI
    azure_openai_endpoint: str
    azure_openai_api_key: str
    azure_openai_deployment_name: str = "gpt-4o"
    azure_openai_api_version: str = "2024-12-01-preview"

    # Database — any SQLAlchemy-compatible async URL
    # SQLite:      sqlite+aiosqlite:///./health.db
    # PostgreSQL:  postgresql+asyncpg://user:pass@host:5432/dbname
    # MySQL:       mysql+aiomysql://user:pass@host:3306/dbname
    database_url: str = "sqlite+aiosqlite:///./health.db"

    # Query safety
    max_result_rows: int = 500
    query_timeout_seconds: int = 30

    # PII columns for SQLAS safety scoring (comma-separated in env)
    pii_columns: list[str] = []

    # Optional: domain hint injected into the system prompt
    domain_hint: str = ""

    # CORS
    frontend_url: str = "http://localhost:5173"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    return Settings()

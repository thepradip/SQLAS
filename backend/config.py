"""Application configuration from environment variables."""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Azure OpenAI
    azure_openai_endpoint: str
    azure_openai_api_key: str
    azure_openai_deployment_name: str = "gpt-5.2-chat"
    azure_openai_api_version: str = "2024-12-01-preview"

    # Database — any SQLAlchemy-compatible async URL
    # SQLite:      sqlite+aiosqlite:///./health.db
    # PostgreSQL:  postgresql+asyncpg://user:pass@host:5432/dbname
    # MySQL:       mysql+aiomysql://user:pass@host:3306/dbname
    database_url: str = "sqlite+aiosqlite:///./health.db"

    # Query safety
    max_result_rows: int = 500
    query_timeout_seconds: int = 30

    # Optional: domain hint injected into the system prompt (e.g., "This is a health analytics database.")
    domain_hint: str = ""

    # Large-schema mode — semantic table retrieval
    # Set AZURE_OPENAI_EMBEDDING_DEPLOYMENT to an ada-002 / text-embedding-3-small deployment
    # to upgrade from BM25 to BM25+dense hybrid (RRF). Leave blank for BM25-only (no extra cost).
    azure_openai_embedding_deployment: str = ""

    # Tables above this count switch to semantic index + focused context instead of full-schema dump.
    large_schema_threshold: int = 20

    # How many tables to retrieve per query (FK expansion adds up to +4 neighbors).
    max_context_tables: int = 8

    # Max columns injected per table into the LLM prompt.
    # For wide tables (100+ cols) only the most query-relevant columns are shown.
    # PKs and FK columns are always included regardless of this limit.
    # Set 0 to disable (inject all columns — not recommended for 50+ col tables).
    max_columns_per_table: int = 30

    # Token budget for injected schema context (1 token ≈ 4 chars).
    # Prevents context window overflow on very large focused schemas.
    schema_token_budget: int = 12_000

    # ── Agentic mode ───────────────────────────────────────────────────────────
    # When True, complex queries (correlation, comparison, explanation) are routed
    # to the ReAct agent which uses multi-step tool calling instead of single-shot SQL.
    # Requires a provider that supports tool calling (azure, openai:*, anthropic:*).
    # Set False to always use the fast pipeline.
    agentic_mode: bool = True

    # ── Alternative LLM providers (for eval comparison) ───────────────────────
    # Leave blank for providers you don't use — they'll only fail if you try to use them.
    openai_api_key: str = ""                          # for "openai:gpt-4o" etc.
    anthropic_api_key: str = ""                       # for "anthropic:claude-opus-4-7" etc.
    ollama_base_url: str = "http://localhost:11434"   # for "ollama:sqlcoder" etc.
    compat_api_key: str = "not-needed"                # for "compat:model@url" endpoints

    # ── Query caching ──────────────────────────────────────────────────────────
    cache_enabled: bool = True

    # Cosine similarity threshold for semantic cache hits (0.0–1.0).
    # 0.92 is conservative — only near-identical queries hit. Lower to 0.88 for more aggressive caching.
    semantic_cache_threshold: float = 0.92

    # TTL in seconds for SQL result cache. 0 = disable result caching.
    # 300 = 5 min (good for dashboards/reports). Set lower for real-time data.
    result_cache_ttl: int = 300

    # CORS
    frontend_url: str = "http://localhost:5173"

    class Config:
        env_file = "../.env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    return Settings()

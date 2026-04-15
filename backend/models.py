from __future__ import annotations
"""
Pydantic models for API request/response.

Author: Pradip Tivhale
"""

from pydantic import BaseModel


class QueryRequest(BaseModel):
    query: str
    conversation_id: str | None = None


class QueryData(BaseModel):
    columns: list[str]
    rows: list[list]
    row_count: int
    truncated: bool
    execution_time_ms: float


class SQLASScoresResponse(BaseModel):
    overall_score: float | None = None
    execution_accuracy: float | None = None
    semantic_equivalence: float | None = None
    faithfulness: float | None = None
    answer_relevance: float | None = None
    safety_score: float | None = None
    read_only_compliance: float | None = None
    error: str | None = None


class QueryResponse(BaseModel):
    sql: str
    data: QueryData | None
    response: str
    success: bool
    trace_id: str | None = None
    metrics: dict | None = None
    sqlas_scores: SQLASScoresResponse | None = None


class FeedbackRequest(BaseModel):
    trace_id: str
    value: bool
    comment: str | None = None
    user_id: str = "anonymous"


class DetailedFeedbackRequest(BaseModel):
    trace_id: str
    accuracy: int
    relevance: int
    sql_quality: int
    comment: str | None = None
    user_id: str = "anonymous"


class HealthCheck(BaseModel):
    status: str
    database: str
    tables: list[str]
    agent_type: str


class SchemaResponse(BaseModel):
    schema_text: str

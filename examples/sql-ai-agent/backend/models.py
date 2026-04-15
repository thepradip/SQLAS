from __future__ import annotations
"""
Pydantic models for API request/response.

Author: Pradip Tivhale
"""

from pydantic import BaseModel


class QueryRequest(BaseModel):
    query: str
    conversation_id: str | None = None


class QueryMetrics(BaseModel):
    total_latency_ms: float
    generation_latency_ms: float
    sql_execution_ms: float
    narration_latency_ms: float
    retry_count: int
    result_rows: int | None = None
    result_columns: int | None = None
    success: bool
    error_type: str | None = None
    # SQL complexity
    sql_length: int | None = None
    join_count: int | None = None
    cte_count: int | None = None
    subquery_count: int | None = None
    has_group_by: bool | None = None
    has_order_by: bool | None = None
    has_limit: bool | None = None
    has_having: bool | None = None
    has_case_when: bool | None = None
    has_distinct: bool | None = None
    has_aggregation: bool | None = None
    has_window_function: bool | None = None
    has_null_handling: bool | None = None
    table_count: int | None = None
    where_conditions: int | None = None
    query_type: str | None = None


class QueryData(BaseModel):
    columns: list[str]
    rows: list[list]
    row_count: int
    truncated: bool
    execution_time_ms: float


class QueryResponse(BaseModel):
    sql: str
    data: QueryData | None
    response: str
    success: bool
    trace_id: str | None = None
    metrics: QueryMetrics | None = None


class FeedbackRequest(BaseModel):
    trace_id: str
    value: bool  # True = thumbs up, False = thumbs down
    comment: str | None = None
    user_id: str = "anonymous"


class DetailedFeedbackRequest(BaseModel):
    trace_id: str
    accuracy: int  # 1-5
    relevance: int  # 1-5
    sql_quality: int  # 1-5
    comment: str | None = None
    user_id: str = "anonymous"


class HealthCheck(BaseModel):
    status: str
    database: str
    tables: list[str]
    mlflow_experiment: str


class SchemaResponse(BaseModel):
    schema_text: str

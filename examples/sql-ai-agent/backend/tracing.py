from __future__ import annotations
"""
MLflow Tracing & Metrics for SQL AI Agent.
Provides comprehensive observability: traces, spans, metrics, and feedback.

Author: Pradip Tivhale
"""

import re
import time
from functools import wraps

import mlflow
from mlflow.entities import SpanType, AssessmentSource, AssessmentSourceType

from config import get_settings

settings = get_settings()

EXPERIMENT_NAME = "sql-ai-agent"


def init_tracing():
    """Initialize MLflow experiment and OpenAI autologging."""
    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.openai.autolog()
    print(f"  MLflow tracing initialized — experiment: {EXPERIMENT_NAME}")


# ─── SQL Complexity Metrics ────────────────────────────────────────────────────

def compute_sql_metrics(sql: str) -> dict:
    """Analyze SQL query complexity and characteristics."""
    upper = sql.upper()
    return {
        "sql_length": len(sql),
        "join_count": len(re.findall(r"\bJOIN\b", upper)),
        "cte_count": len(re.findall(r"\bWITH\b", upper)),
        "subquery_count": upper.count("SELECT") - 1,  # nested SELECTs beyond the main one
        "has_group_by": "GROUP BY" in upper,
        "has_order_by": "ORDER BY" in upper,
        "has_limit": "LIMIT" in upper,
        "has_having": "HAVING" in upper,
        "has_case_when": "CASE WHEN" in upper or "CASE\n" in upper,
        "has_distinct": "DISTINCT" in upper,
        "has_aggregation": any(fn in upper for fn in ["COUNT(", "SUM(", "AVG(", "MIN(", "MAX("]),
        "has_window_function": "OVER(" in upper,
        "has_null_handling": "IS NULL" in upper or "IS NOT NULL" in upper or "COALESCE" in upper,
        "table_count": len(re.findall(r"\bFROM\b", upper)) + len(re.findall(r"\bJOIN\b", upper)),
        "where_conditions": len(re.findall(r"\bAND\b|\bOR\b", upper)) + (1 if "WHERE" in upper else 0),
        "query_type": _classify_query(upper),
    }


def _classify_query(upper_sql: str) -> str:
    """Classify query type for analytics."""
    if "JOIN" in upper_sql and "GROUP BY" in upper_sql:
        return "join_aggregation"
    if "JOIN" in upper_sql:
        return "join"
    if "GROUP BY" in upper_sql:
        return "aggregation"
    if "WHERE" in upper_sql:
        return "filter"
    return "simple_select"


# ─── Traced Agent Pipeline ─────────────────────────────────────────────────────

async def traced_run_query(
    user_query: str,
    conversation_history: list[dict],
    generate_sql_fn,
    execute_sql_fn,
    narrate_fn,
    retry_generate_fn,
) -> dict:
    """
    Full agent pipeline wrapped in MLflow tracing.
    Each step is a separate span with metrics.
    """
    with mlflow.start_span("sql_agent_pipeline", span_type=SpanType.AGENT) as root_span:
        root_span.set_inputs({"user_query": user_query, "history_length": len(conversation_history)})

        pipeline_start = time.perf_counter()
        retry_count = 0
        error_type = None

        # ── Step 1: SQL Generation ──────────────────────────────────────────
        with mlflow.start_span("generate_sql", span_type=SpanType.LLM) as gen_span:
            gen_start = time.perf_counter()
            gen_span.set_inputs({"user_query": user_query})

            sql = await generate_sql_fn(user_query, conversation_history)

            gen_latency_ms = (time.perf_counter() - gen_start) * 1000
            sql_metrics = compute_sql_metrics(sql)

            gen_span.set_outputs({"sql": sql})
            gen_span.set_attributes({
                "generation_latency_ms": round(gen_latency_ms, 2),
                **{f"sql.{k}": v for k, v in sql_metrics.items()},
            })

        # ── Step 2: SQL Execution ───────────────────────────────────────────
        query_result = None
        with mlflow.start_span("execute_sql", span_type=SpanType.TOOL) as exec_span:
            exec_span.set_inputs({"sql": sql})
            try:
                query_result = await execute_sql_fn(sql)
                exec_span.set_outputs({
                    "row_count": query_result["row_count"],
                    "execution_time_ms": query_result["execution_time_ms"],
                    "truncated": query_result["truncated"],
                    "columns": query_result["columns"],
                })
                exec_span.set_attributes({
                    "sql_execution_ms": query_result["execution_time_ms"],
                    "result_row_count": query_result["row_count"],
                    "result_column_count": len(query_result["columns"]),
                    "result_truncated": query_result["truncated"],
                })
            except Exception as e:
                error_type = type(e).__name__
                exec_span.set_attributes({"error": str(e), "error_type": error_type})
                exec_span.set_status("ERROR")

                # ── Step 2b: Retry ──────────────────────────────────────────
                retry_count = 1
                with mlflow.start_span("retry_generate_sql", span_type=SpanType.LLM) as retry_span:
                    retry_span.set_inputs({"original_sql": sql, "error": str(e)})
                    sql = await retry_generate_fn(user_query, sql, str(e), conversation_history)
                    retry_span.set_outputs({"corrected_sql": sql})
                    retry_span.set_attributes({"retry_sql_metrics": compute_sql_metrics(sql)})

                with mlflow.start_span("retry_execute_sql", span_type=SpanType.TOOL) as retry_exec_span:
                    retry_exec_span.set_inputs({"sql": sql})
                    try:
                        query_result = await execute_sql_fn(sql)
                        retry_exec_span.set_outputs({
                            "row_count": query_result["row_count"],
                            "execution_time_ms": query_result["execution_time_ms"],
                        })
                    except Exception as e2:
                        error_type = type(e2).__name__
                        retry_exec_span.set_attributes({"error": str(e2), "error_type": error_type})
                        retry_exec_span.set_status("ERROR")

                        total_latency_ms = (time.perf_counter() - pipeline_start) * 1000
                        root_span.set_outputs({"success": False, "error": str(e2)})
                        root_span.set_attributes({
                            "pipeline.success": False,
                            "pipeline.total_latency_ms": round(total_latency_ms, 2),
                            "pipeline.retry_count": retry_count,
                            "pipeline.error_type": error_type,
                            "pipeline.query_type": sql_metrics["query_type"],
                        })
                        root_span.set_status("ERROR")

                        trace_id = root_span.trace_id
                        return {
                            "sql": sql,
                            "data": None,
                            "response": f"Unable to process query after retry.\n\n**Error:** `{str(e2)}`\n\nTry rephrasing your question.",
                            "success": False,
                            "trace_id": trace_id,
                            "metrics": {
                                "total_latency_ms": round(total_latency_ms, 2),
                                "generation_latency_ms": round(gen_latency_ms, 2),
                                "sql_execution_ms": 0,
                                "narration_latency_ms": 0,
                                "retry_count": retry_count,
                                "success": False,
                                "error_type": error_type,
                                **sql_metrics,
                            },
                        }

        # ── Step 3: Narration ───────────────────────────────────────────────
        with mlflow.start_span("narrate_result", span_type=SpanType.LLM) as narr_span:
            narr_start = time.perf_counter()
            narr_span.set_inputs({
                "user_query": user_query,
                "row_count": query_result["row_count"],
            })

            response_text = await narrate_fn(user_query, sql, query_result)

            narr_latency_ms = (time.perf_counter() - narr_start) * 1000
            narr_span.set_outputs({"response_length": len(response_text)})
            narr_span.set_attributes({
                "narration_latency_ms": round(narr_latency_ms, 2),
                "response_length": len(response_text),
            })

        # ── Pipeline summary ────────────────────────────────────────────────
        total_latency_ms = (time.perf_counter() - pipeline_start) * 1000

        root_span.set_outputs({
            "success": True,
            "sql": sql,
            "row_count": query_result["row_count"],
            "response_preview": response_text[:200],
        })
        root_span.set_attributes({
            "pipeline.success": True,
            "pipeline.total_latency_ms": round(total_latency_ms, 2),
            "pipeline.generation_latency_ms": round(gen_latency_ms, 2),
            "pipeline.sql_execution_ms": query_result["execution_time_ms"],
            "pipeline.narration_latency_ms": round(narr_latency_ms, 2),
            "pipeline.retry_count": retry_count,
            "pipeline.result_rows": query_result["row_count"],
            "pipeline.query_type": sql_metrics["query_type"],
            "pipeline.join_count": sql_metrics["join_count"],
            "pipeline.table_count": sql_metrics["table_count"],
        })

        trace_id = root_span.trace_id

    metrics = {
        "total_latency_ms": round(total_latency_ms, 2),
        "generation_latency_ms": round(gen_latency_ms, 2),
        "sql_execution_ms": query_result["execution_time_ms"],
        "narration_latency_ms": round(narr_latency_ms, 2),
        "retry_count": retry_count,
        "result_rows": query_result["row_count"],
        "result_columns": len(query_result["columns"]),
        "success": True,
        "error_type": None,
        **sql_metrics,
    }

    return {
        "sql": sql,
        "data": {
            "columns": query_result["columns"],
            "rows": query_result["rows"],
            "row_count": query_result["row_count"],
            "truncated": query_result["truncated"],
            "execution_time_ms": query_result["execution_time_ms"],
        },
        "response": response_text,
        "success": True,
        "trace_id": trace_id,
        "metrics": metrics,
    }


# ─── Feedback API ──────────────────────────────────────────────────────────────

def log_user_feedback(
    trace_id: str,
    feedback_value: bool,
    user_id: str = "anonymous",
    comment: str | None = None,
):
    """Log thumbs up/down feedback for a trace."""
    mlflow.log_feedback(
        trace_id=trace_id,
        name="user_thumbs",
        value=feedback_value,
        source=AssessmentSource(
            source_type=AssessmentSourceType.HUMAN,
            source_id=user_id,
        ),
        rationale=comment or ("Positive feedback" if feedback_value else "Negative feedback"),
    )

    # Also set as trace tag for easy searching
    mlflow.set_trace_tag(trace_id, "user_feedback", "thumbs_up" if feedback_value else "thumbs_down")
    if comment:
        mlflow.set_trace_tag(trace_id, "feedback_comment", comment)


def log_detailed_feedback(
    trace_id: str,
    accuracy: int,
    relevance: int,
    sql_quality: int,
    user_id: str = "anonymous",
    comment: str | None = None,
):
    """Log detailed multi-dimension feedback (1-5 scale)."""
    mlflow.log_feedback(
        trace_id=trace_id,
        name="detailed_rating",
        value={
            "accuracy": accuracy,
            "relevance": relevance,
            "sql_quality": sql_quality,
            "overall": round((accuracy + relevance + sql_quality) / 3, 2),
        },
        source=AssessmentSource(
            source_type=AssessmentSourceType.HUMAN,
            source_id=user_id,
        ),
        rationale=comment or "Detailed rating provided",
    )

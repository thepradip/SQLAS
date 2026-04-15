"""
LangGraph StateGraph definition for the SQL AI Agent.

Pipeline:
    retrieve_schema → generate_sql → validate_sql → [safe?]
        → execute_sql → [success?]
            → narrate_result → evaluate_quality → END
            → handle_error → [retry?]
                → generate_sql (loop)
                → fail_after_retries → END
        → reject_unsafe → END

Author: Pradip Tivhale
"""

from __future__ import annotations

import time
import logging

from langgraph.graph import StateGraph, END

from agent.state import AgentState
from agent.nodes import (
    retrieve_schema,
    generate_sql,
    validate_sql,
    execute_sql,
    handle_error,
    narrate_result,
    evaluate_quality,
    reject_unsafe,
    fail_after_retries,
    init_schema,
)

logger = logging.getLogger(__name__)


# ─── Conditional edge functions ────────────────────────────────────────────────

def route_after_validation(state: dict) -> str:
    """Route based on SQLAS safety gate result."""
    if state.get("is_safe", False):
        return "execute_sql"
    return "reject_unsafe"


def route_after_execution(state: dict) -> str:
    """Route based on SQL execution success/failure."""
    if state.get("execution_error") is None and state.get("execution_result") is not None:
        return "narrate_result"
    return "handle_error"


def route_after_error(state: dict) -> str:
    """Route based on retry count."""
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)
    if retry_count < max_retries:
        return "generate_sql"
    return "fail_after_retries"


# ─── Graph builder ─────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """Build and compile the LangGraph SQL Agent pipeline."""

    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("retrieve_schema", retrieve_schema)
    graph.add_node("generate_sql", generate_sql)
    graph.add_node("validate_sql", validate_sql)
    graph.add_node("execute_sql", execute_sql)
    graph.add_node("handle_error", handle_error)
    graph.add_node("narrate_result", narrate_result)
    graph.add_node("evaluate_quality", evaluate_quality)
    graph.add_node("reject_unsafe", reject_unsafe)
    graph.add_node("fail_after_retries", fail_after_retries)

    # Entry point
    graph.set_entry_point("retrieve_schema")

    # Edges: linear flow
    graph.add_edge("retrieve_schema", "generate_sql")
    graph.add_edge("generate_sql", "validate_sql")

    # Conditional: safety gate
    graph.add_conditional_edges(
        "validate_sql",
        route_after_validation,
        {"execute_sql": "execute_sql", "reject_unsafe": "reject_unsafe"},
    )

    # Conditional: execution success/failure
    graph.add_conditional_edges(
        "execute_sql",
        route_after_execution,
        {"narrate_result": "narrate_result", "handle_error": "handle_error"},
    )

    # Conditional: retry or fail
    graph.add_conditional_edges(
        "handle_error",
        route_after_error,
        {"generate_sql": "generate_sql", "fail_after_retries": "fail_after_retries"},
    )

    # Linear: narrate → evaluate → END
    graph.add_edge("narrate_result", "evaluate_quality")
    graph.add_edge("evaluate_quality", END)

    # Terminal nodes → END
    graph.add_edge("reject_unsafe", END)
    graph.add_edge("fail_after_retries", END)

    return graph.compile()


# ─── Public API ────────────────────────────────────────────────────────────────

# Compiled graph (initialized on import)
_compiled_graph = None


def get_graph():
    """Get or create the compiled graph."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


async def run_query(user_query: str, conversation_history: list[dict]) -> dict:
    """Execute the full LangGraph pipeline: NL → SQL → Execute → Narrate → Evaluate."""
    graph = get_graph()

    initial_state: AgentState = {
        "question": user_query,
        "conversation_history": conversation_history,
        "schema_context": "",
        "generated_sql": "",
        "is_safe": False,
        "safety_details": {},
        "execution_result": None,
        "execution_error": None,
        "response": "",
        "sqlas_scores": None,
        "retry_count": 0,
        "max_retries": 2,
        "metrics": {},
        "success": False,
        "trace_id": None,
    }

    start = time.perf_counter()
    final_state = await graph.ainvoke(initial_state)
    total_ms = (time.perf_counter() - start) * 1000

    metrics = final_state.get("metrics", {})
    metrics["total_latency_ms"] = round(total_ms, 2)

    result = {
        "sql": final_state.get("generated_sql", ""),
        "data": None,
        "response": final_state.get("response", ""),
        "success": final_state.get("success", False),
        "trace_id": final_state.get("trace_id"),
        "metrics": metrics,
        "sqlas_scores": final_state.get("sqlas_scores"),
    }

    execution_result = final_state.get("execution_result")
    if execution_result:
        result["data"] = {
            "columns": execution_result["columns"],
            "rows": execution_result["rows"],
            "row_count": execution_result["row_count"],
            "truncated": execution_result["truncated"],
            "execution_time_ms": execution_result["execution_time_ms"],
        }

    logger.info(
        "Pipeline complete: success=%s, total_ms=%.1f, sqlas=%.2f",
        result["success"],
        total_ms,
        (final_state.get("sqlas_scores") or {}).get("overall_score", 0),
    )

    return result

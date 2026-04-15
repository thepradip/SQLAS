"""
Agent state definition for the LangGraph SQL Agent.

Author: Pradip Tivhale
"""

from __future__ import annotations
from typing import TypedDict, Any


class AgentState(TypedDict, total=False):
    """State flowing through the LangGraph SQL Agent pipeline."""

    # ── Input ──────────────────────────────────────────────────────────────
    question: str
    conversation_history: list[dict]

    # ── Schema retrieval ───────────────────────────────────────────────────
    schema_context: str

    # ── SQL generation ─────────────────────────────────────────────────────
    generated_sql: str

    # ── Validation (SQLAS safety gate) ─────────────────────────────────────
    is_safe: bool
    safety_details: dict[str, Any]

    # ── Execution ──────────────────────────────────────────────────────────
    execution_result: dict[str, Any] | None
    execution_error: str | None

    # ── Narration ──────────────────────────────────────────────────────────
    response: str

    # ── SQLAS evaluation ───────────────────────────────────────────────────
    sqlas_scores: dict[str, Any] | None

    # ── Control flow ───────────────────────────────────────────────────────
    retry_count: int
    max_retries: int

    # ── Metrics & tracing ──────────────────────────────────────────────────
    metrics: dict[str, Any]
    success: bool
    trace_id: str | None

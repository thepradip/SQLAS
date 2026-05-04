"""
ReAct Agent — Reason + Act loop for the Agentic SQL Agent.

The agent is given 4 tools and can use them in any order, any number of times
(up to MAX_STEPS), before calling final_answer().

This is what makes the system genuinely "agentic":
  - The LLM decides WHICH tools to call, not the pipeline
  - The LLM can inspect the schema before writing SQL
  - The LLM can run multiple queries and combine results
  - The LLM self-corrects: if a query returns unexpected results, it can
    reason about why and try a different approach

ReAct loop trace (logged to MLflow):
  step 1: list_tables          → see what's available
  step 2: describe_table(...)  → understand schema before querying
  step 3: execute_sql(...)     → run the query
  step 4: final_answer(...)    → deliver the answer

Contrast with pipeline mode (agent.py):
  query → [one LLM call] → SQL → execute → narrate
  (no introspection, no multi-step reasoning, no self-correction)
"""

import time
from typing import Optional

import mlflow
from mlflow.entities import SpanType

from llm_providers import LLMProvider
from tools import TOOL_DEFINITIONS, execute_tool
from visualization import build_visualization

MAX_STEPS = 10  # slightly higher because planning adds 1-2 steps

AGENTIC_SYSTEM_PROMPT = """You are an expert SQL analyst. Your goal is to answer the user's question CORRECTLY on the FIRST attempt by planning before acting.

## MANDATORY EXECUTION ORDER — no exceptions

STEP 1 ▶ create_plan  (ALWAYS FIRST — no other tool before this)
  Explicitly state: which tables, which JOINs, SQL approach, potential issues.
  This is not optional. If you skip it, your SQL will likely be wrong.

STEP 2 ▶ describe_table  (for EVERY table in your plan)
  Column names must be exact — do not guess them from table names.
  Check nullable columns so you know where to add IS NOT NULL or COALESCE.
  Check categorical top values so you know exact filter strings.

STEP 3 ▶ execute_sql  (only after describing all tables)
  Write SQL using the exact column names from describe_table output.
  For JOINs: aggregate the many-side first to avoid row explosion.
  For aggregations: always add ORDER BY for deterministic results.
  For percentages: CAST to REAL to avoid integer division.

STEP 4 ▶ final_answer  (only when you have real results)
  State the direct answer first. Never invent numbers.

## Why this order matters
Skipping create_plan → you guess column names → SQL fails → retry needed.
Skipping describe_table → wrong JOIN condition → row explosion or empty result.
These are the two most common causes of first-attempt failure. The plan prevents both.

## SQL Rules
- Only SELECT or WITH (CTE) + SELECT.
- ROUND numeric output to 2 decimal places.
- Default LIMIT 100 for detail queries; no LIMIT for aggregations.
- Handle NULLs: check null counts in describe_table output.

## Answer Rules
- Direct answer first: "987 patients have high blood pressure."
- 1-3 sentences. No markdown tables. No invented numbers.

{schema_overview}
"""


async def run_react_query(
    user_query: str,
    provider: LLMProvider,
    schema_overview: str,
    conversation_history: list[dict],
) -> dict:
    """
    Execute a user query via the ReAct loop.

    Returns the same dict shape as the pipeline agent so the API layer
    doesn't need to know which mode was used.
    """
    if not provider.supports_tool_calling:
        return {
            "success": False,
            "sql": "",
            "data": None,
            "response": (
                f"{provider.name} does not support tool calling. "
                "Switch to azure, openai:*, or anthropic:* for agentic mode."
            ),
            "agent_mode": "react",
            "steps": [],
        }

    system_prompt = AGENTIC_SYSTEM_PROMPT.format(
        schema_overview=schema_overview or "No schema overview available."
    )

    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        *conversation_history[-4:],
        {"role": "user", "content": user_query},
    ]

    steps: list[dict] = []
    final_sql = ""
    final_answer_text = ""
    last_result_data: Optional[dict] = None
    pipeline_start = time.perf_counter()
    plan_created = False          # enforcement flag — blocks execute_sql until plan exists
    tables_described: set[str] = set()  # tracks which tables have been described

    with mlflow.start_span("react_agent_loop", span_type=SpanType.AGENT) as root_span:
        root_span.set_inputs({"user_query": user_query, "provider": provider.name})

        for step_num in range(1, MAX_STEPS + 1):
            with mlflow.start_span(f"react_step_{step_num}", span_type=SpanType.TOOL) as step_span:
                step_span.set_inputs({"step": step_num, "messages_len": len(messages)})

                assistant_msg, tool_calls = provider.complete_with_tools(
                    messages, TOOL_DEFINITIONS, max_tokens=2000
                )
                messages.append(assistant_msg)

                if not tool_calls:
                    # LLM gave a text response — treat as final answer
                    final_answer_text = assistant_msg.get("content", "")
                    step_span.set_outputs({"action": "text_response", "content_len": len(final_answer_text)})
                    break

                for tc in tool_calls:
                    tool_name = tc["name"]
                    tool_args = tc["arguments"]

                    step_span.set_attributes({
                        "tool": tool_name,
                        "args": str(tool_args)[:200],
                    })

                    # ── Plan enforcement ──────────────────────────────────────
                    # Block execute_sql until create_plan has been called.
                    # This is the key guard that prevents first-attempt failures
                    # caused by guessing column names before inspecting the schema.
                    if tool_name == "execute_sql" and not plan_created:
                        blocked_msg = (
                            "BLOCKED: You must call create_plan before execute_sql. "
                            "State your plan (tables, JOINs, approach) first, then "
                            "describe_table for each table, then execute_sql."
                        )
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.get("id", tool_name),
                            "content": blocked_msg,
                        })
                        steps.append({
                            "step": step_num, "tool": "BLOCKED_execute_sql",
                            "args": tool_args, "result_preview": blocked_msg,
                        })
                        step_span.set_attributes({"enforcement": "blocked_no_plan"})
                        continue   # force agent to plan first

                    # Track state for enforcement and SQLAS metrics
                    if tool_name == "create_plan":
                        plan_created = True
                        step_span.set_attributes({"plan_created": True})
                    elif tool_name == "describe_table":
                        tname = tool_args.get("table_name", "")
                        if tname:
                            tables_described.add(tname.lower())

                    if tool_name == "final_answer":
                        final_answer_text = tool_args.get("answer", "")
                        final_sql = tool_args.get("sql", "")
                        steps.append({"step": step_num, "tool": "final_answer", "args": tool_args})
                        step_span.set_outputs({"action": "final_answer"})

                        total_ms = (time.perf_counter() - pipeline_start) * 1000
                        root_span.set_outputs({
                            "success": True,
                            "steps_taken": len(steps),
                            "final_sql": final_sql[:200],
                        })
                        root_span.set_attributes({
                            "react.steps_taken": len(steps),
                            "react.provider": provider.name,
                            "react.total_latency_ms": round(total_ms, 2),
                        })

                        return _build_result(
                            final_sql, final_answer_text, steps,
                            last_result_data, user_query, total_ms, success=True,
                        )

                    # Execute the tool
                    tool_result_str = await execute_tool(tool_name, tool_args)

                    # Track last execute_sql result for visualization
                    if tool_name == "execute_sql" and tool_args.get("sql"):
                        try:
                            from database import execute_readonly_query
                            last_result_data = await execute_readonly_query(tool_args["sql"])
                            final_sql = tool_args["sql"]
                        except Exception:
                            pass

                    steps.append({
                        "step": step_num,
                        "tool": tool_name,
                        "args": tool_args,
                        "result_preview": tool_result_str[:300],
                    })

                    # Feed result back to the conversation
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.get("id", tool_name),
                        "content": tool_result_str,
                    })

                    step_span.set_outputs({
                        "tool": tool_name,
                        "result_preview": tool_result_str[:200],
                    })

        # Exited loop without final_answer
        total_ms = (time.perf_counter() - pipeline_start) * 1000
        root_span.set_outputs({"success": bool(final_answer_text), "steps_taken": len(steps)})

    # Surface planning quality to SQLAS metrics
    root_span.set_attributes({
        "react.plan_created":       plan_created,
        "react.tables_described":   len(tables_described),
        "react.planned_first":      plan_created,
    })

    if not final_answer_text:
        final_answer_text = (
            f"Could not complete the analysis within {MAX_STEPS} steps. "
            "Try asking a more focused question."
        )

    return _build_result(
        final_sql, final_answer_text, steps,
        last_result_data, user_query, total_ms,
        success=bool(final_sql or final_answer_text),
    )


def _build_result(
    sql: str,
    response: str,
    steps: list[dict],
    result_data: Optional[dict],
    user_query: str,
    total_ms: float,
    success: bool,
) -> dict:
    viz = build_visualization(user_query, result_data) if result_data else None
    return {
        "sql": sql,
        "data": result_data,
        "response": response,
        "success": success,
        "trace_id": None,
        "agent_mode": "react",
        "steps": steps,
        "metrics": {
            "total_latency_ms": round(total_ms, 2),
            "generation_latency_ms": 0,
            "sql_execution_ms": result_data.get("execution_time_ms", 0) if result_data else 0,
            "narration_latency_ms": 0,
            "retry_count": 0,
            "steps_taken": len(steps),
            "result_rows": result_data.get("row_count") if result_data else None,
            "result_columns": len(result_data.get("columns", [])) if result_data else None,
            "success": success,
        },
        "visualization": viz,
    }

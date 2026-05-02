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

MAX_STEPS = 8   # prevents infinite loops; 4-5 steps is typical for complex queries

AGENTIC_SYSTEM_PROMPT = """You are an expert SQL analyst with access to a database.
You have tools to explore the database and answer questions accurately.

## Your Approach
1. Use list_tables if you need to discover available tables.
2. Use describe_table to understand column names, types, and value formats BEFORE writing SQL.
   This is critical — column names must be exact.
3. Use execute_sql to run queries. You may run multiple queries if needed.
4. Use final_answer ONLY when you have actual query results to report.

## SQL Rules
- Only SELECT or WITH...SELECT queries.
- Use exact column names from describe_table output.
- For JOIN queries: aggregate the many-side first to avoid row explosion.
- ROUND numeric results to 2 decimal places.
- ORDER BY for deterministic results.
- Default LIMIT 100 for detail queries.

## Answer Rules
- State the direct answer first (the number, comparison, or fact).
- Keep it concise — 1-3 sentences for simple queries.
- Never invent numbers. Only report what the query returned.

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

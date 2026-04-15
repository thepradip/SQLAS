"""
LangGraph node functions for the SQL AI Agent pipeline.

Each node receives the full AgentState and returns a partial state update.
SQLAS is used for pre-execution validation and post-response evaluation.

Author: Pradip Tivhale
"""

from __future__ import annotations

import re
import time
import logging
from pathlib import Path

from openai import AzureOpenAI
from sqlas import (
    safety_score,
    read_only_compliance,
    schema_compliance,
    evaluate as sqlas_evaluate,
)

from config import get_settings
from database import build_full_context, execute_readonly_query, get_full_schema

logger = logging.getLogger(__name__)
settings = get_settings()

client = AzureOpenAI(
    azure_endpoint=settings.azure_openai_endpoint,
    api_key=settings.azure_openai_api_key,
    api_version=settings.azure_openai_api_version,
)

DB_PATH = str(Path(__file__).resolve().parent.parent / "health.db")

# ─── Cached schema metadata (populated on startup) ────────────────────────────

_schema_context: str = ""
_valid_tables: set[str] = set()
_valid_columns: dict[str, set[str]] = {}


async def init_schema():
    """Called once on app startup to introspect the database."""
    global _schema_context, _valid_tables, _valid_columns
    _schema_context = await build_full_context()
    schema = await get_full_schema()
    _valid_tables = set(schema.keys())
    _valid_columns = {
        table: {col["name"] for col in info["columns"]}
        for table, info in schema.items()
    }
    logger.info("Schema introspected: %d tables", len(_valid_tables))


# ─── Helpers ───────────────────────────────────────────────────────────────────

def _build_system_prompt() -> str:
    domain = ""
    if settings.domain_hint:
        domain = f"\n## Domain Context\n{settings.domain_hint}\n"

    return f"""You are an expert SQL analyst. You write efficient, read-only SQL queries against any database schema provided below.

## STRICT READ-ONLY POLICY
- You MUST only generate SELECT statements or WITH (CTE) + SELECT.
- NEVER generate INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, TRUNCATE, GRANT, REVOKE, or any DDL/DML.
- If the user asks to modify, insert, or delete data, refuse politely.

## Database Schema (auto-discovered)
{_schema_context}
{domain}
## SQL Generation Rules
1. Write standard SQL compatible with the connected database.
2. Only generate SELECT or WITH...SELECT queries.
3. Use the column statistics provided above to write accurate filters and understand value ranges.
4. Use CASE WHEN to map coded values to readable labels when the column stats show a small number of distinct integer values.
5. Use meaningful column aliases (AS keyword).
6. For multi-table queries involving 1:N relationships, aggregate the N-side first, then JOIN to avoid row explosion.
7. Use CTEs (WITH clauses) for complex multi-step logic.
8. Default LIMIT 100 for detail queries. Aggregated summaries need no limit.
9. Handle NULLs appropriately (COALESCE, IS NOT NULL, etc.) — check the null counts in the stats.
10. For percentages: CAST to REAL to avoid integer division.
11. ROUND numeric outputs to 2 decimal places.
12. Always ORDER BY for deterministic results.
13. Use indexes — prefer filtering on indexed columns for large tables.

## Output Format
Return ONLY the SQL inside a ```sql``` block. No explanations outside the block.
"""


def _extract_sql(content: str) -> str:
    match = re.search(r"```sql\s*(.*?)```", content, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"```\s*((?:SELECT|WITH).*?)```", content, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return content.strip()


def _llm_judge(prompt: str) -> str:
    """LLM judge function for SQLAS evaluation metrics."""
    response = client.chat.completions.create(
        model=settings.azure_openai_deployment_name,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=500,
    )
    return response.choices[0].message.content


# ═══════════════════════════════════════════════════════════════════════════════
# NODE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════


async def retrieve_schema(state: dict) -> dict:
    """Node 1: Retrieve and inject database schema context."""
    start = time.perf_counter()
    context = _schema_context
    elapsed = (time.perf_counter() - start) * 1000

    metrics = state.get("metrics", {})
    metrics["schema_retrieval_ms"] = round(elapsed, 2)
    metrics["table_count"] = len(_valid_tables)

    return {"schema_context": context, "metrics": metrics}


async def generate_sql(state: dict) -> dict:
    """Node 2: Generate SQL from natural language using LLM + schema context."""
    start = time.perf_counter()

    question = state["question"]
    history = state.get("conversation_history", [])
    retry_count = state.get("retry_count", 0)
    execution_error = state.get("execution_error")

    messages = [{"role": "system", "content": _build_system_prompt()}]

    # Add conversation history (last 6 turns)
    for msg in history[-6:]:
        messages.append(msg)

    if retry_count > 0 and execution_error:
        # Retry: include the failed SQL and error for self-healing
        failed_sql = state.get("generated_sql", "")
        messages.append({
            "role": "user",
            "content": (
                f'SQL for "{question}" failed:\nError: {execution_error}\n\n'
                f"Failed SQL:\n```sql\n{failed_sql}\n```\n\n"
                f"Fix the SQL. Return only corrected query in a ```sql``` block."
            ),
        })
    else:
        messages.append({
            "role": "user",
            "content": f'Write an efficient SQL query to answer: "{question}"\n\nReturn ONLY the SQL in a ```sql``` block.',
        })

    response = client.chat.completions.create(
        model=settings.azure_openai_deployment_name,
        messages=messages,
        max_completion_tokens=2000,
    )

    sql = _extract_sql(response.choices[0].message.content)
    elapsed = (time.perf_counter() - start) * 1000

    metrics = state.get("metrics", {})
    metrics["generation_latency_ms"] = round(elapsed, 2)
    metrics["retry_count"] = retry_count

    logger.info("Generated SQL (retry=%d): %s", retry_count, sql[:100])

    return {
        "generated_sql": sql,
        "execution_error": None,
        "metrics": metrics,
    }


async def validate_sql(state: dict) -> dict:
    """Node 3: SQLAS safety gate — validate SQL before execution.

    Uses SQLAS metrics:
    - read_only_compliance: blocks DDL/DML
    - safety_score: checks for PII exposure, injection patterns
    - schema_compliance: validates tables/columns exist
    """
    sql = state["generated_sql"]

    # SQLAS read-only compliance check
    ro_score, ro_details = read_only_compliance(sql)

    # SQLAS safety score (PII + injection detection)
    safe_score, safe_details = safety_score(sql, pii_columns=settings.pii_columns)

    # SQLAS schema compliance (valid tables/columns)
    sc_score, sc_details = schema_compliance(
        sql=sql,
        valid_tables=_valid_tables,
        valid_columns=_valid_columns,
    )

    is_safe = ro_score >= 1.0 and safe_score >= 0.5
    details = {
        "read_only_compliance": ro_score,
        "safety_score": safe_score,
        "schema_compliance": sc_score,
        "read_only_details": ro_details,
        "safety_details": safe_details,
        "schema_details": sc_details,
    }

    if not is_safe:
        logger.warning("SQL failed safety gate: %s", details)

    return {"is_safe": is_safe, "safety_details": details}


async def execute_sql(state: dict) -> dict:
    """Node 4: Execute the validated SQL query (read-only)."""
    sql = state["generated_sql"]

    try:
        result = await execute_readonly_query(sql)
        metrics = state.get("metrics", {})
        metrics["sql_execution_ms"] = result["execution_time_ms"]
        metrics["result_rows"] = result["row_count"]
        metrics["result_columns"] = len(result["columns"])

        return {
            "execution_result": result,
            "execution_error": None,
            "metrics": metrics,
        }
    except Exception as e:
        logger.warning("SQL execution failed: %s", str(e))
        return {
            "execution_result": None,
            "execution_error": str(e),
        }


async def handle_error(state: dict) -> dict:
    """Node 5: Increment retry counter for self-healing loop."""
    retry_count = state.get("retry_count", 0) + 1
    logger.info("Retry %d/%d", retry_count, state.get("max_retries", 2))
    return {"retry_count": retry_count}


async def narrate_result(state: dict) -> dict:
    """Node 6: LLM narrates the query results in natural language."""
    start = time.perf_counter()

    question = state["question"]
    sql = state["generated_sql"]
    result = state["execution_result"]

    columns = result["columns"]
    rows = result["rows"][:50]
    row_count = result["row_count"]
    exec_time = result["execution_time_ms"]

    data_preview = f"Columns: {columns}\n"
    for row in rows[:25]:
        data_preview += f"{row}\n"
    if row_count > 25:
        data_preview += f"... ({row_count} total rows)\n"

    messages = [
        {
            "role": "system",
            "content": (
                "You are a data analyst. Answer the user's question based on the SQL results.\n\n"
                "RULES:\n"
                "- Give the DIRECT answer first. If it's a single number, just state it plainly.\n"
                "- Do NOT wrap single values in markdown tables. Only use tables when comparing 3+ rows of data.\n"
                "- Do NOT add disclaimers, caveats, or health advice. Just answer the question.\n"
                "- ONLY state facts from the SQL result. No invented numbers.\n"
                "- Keep it short -- 1-3 sentences for simple queries, a brief table + 1 sentence for comparisons.\n"
                "- No emojis, no headings, no bullet points for simple answers."
            ),
        },
        {
            "role": "user",
            "content": (
                f"**Question:** {question}\n\n"
                f"**SQL:**\n```sql\n{sql}\n```\n\n"
                f"**Results** ({row_count} rows, {exec_time}ms):\n```\n{data_preview}```\n\n"
                f"Provide a clear answer."
            ),
        },
    ]

    response_obj = client.chat.completions.create(
        model=settings.azure_openai_deployment_name,
        messages=messages,
        max_completion_tokens=2000,
    )

    response_text = response_obj.choices[0].message.content
    elapsed = (time.perf_counter() - start) * 1000

    metrics = state.get("metrics", {})
    metrics["narration_latency_ms"] = round(elapsed, 2)

    return {"response": response_text, "metrics": metrics, "success": True}


async def evaluate_quality(state: dict) -> dict:
    """Node 7: SQLAS post-response evaluation — score the full pipeline output.

    Uses sqlas.evaluate() to compute production metrics:
    execution accuracy, semantic correctness, faithfulness, safety, etc.
    """
    result = state.get("execution_result")
    if not result:
        return {"sqlas_scores": None}

    try:
        scores = sqlas_evaluate(
            question=state["question"],
            generated_sql=state["generated_sql"],
            llm_judge=_llm_judge,
            response=state.get("response", ""),
            result_data={
                "columns": result["columns"],
                "rows": result["rows"],
                "row_count": result["row_count"],
                "execution_time_ms": result["execution_time_ms"],
            },
        )

        sqlas_dict = {
            "overall_score": scores.overall_score,
            "execution_accuracy": scores.metrics.get("execution_accuracy", 0),
            "semantic_equivalence": scores.metrics.get("semantic_equivalence", 0),
            "faithfulness": scores.metrics.get("faithfulness", 0),
            "answer_relevance": scores.metrics.get("answer_relevance", 0),
            "safety_score": scores.metrics.get("safety_score", 0),
            "read_only_compliance": scores.metrics.get("read_only_compliance", 0),
        }

        metrics = state.get("metrics", {})
        metrics["sqlas_overall"] = scores.overall_score

        return {"sqlas_scores": sqlas_dict, "metrics": metrics}

    except Exception as e:
        logger.warning("SQLAS evaluation failed: %s", str(e))
        return {"sqlas_scores": {"error": str(e)}}


async def reject_unsafe(state: dict) -> dict:
    """Terminal node: SQL failed safety validation."""
    details = state.get("safety_details", {})
    return {
        "response": (
            "I cannot execute this query — it failed safety validation.\n\n"
            f"**Read-only compliance:** {details.get('read_only_compliance', 'N/A')}\n"
            f"**Safety score:** {details.get('safety_score', 'N/A')}\n\n"
            "Please rephrase as a read-only SELECT query."
        ),
        "success": False,
        "sqlas_scores": details,
    }


async def fail_after_retries(state: dict) -> dict:
    """Terminal node: SQL failed after all retry attempts."""
    error = state.get("execution_error", "Unknown error")
    return {
        "response": (
            f"Unable to process query after {state.get('retry_count', 0)} retries.\n\n"
            f"**Error:** `{error}`\n\n"
            "Try rephrasing your question."
        ),
        "success": False,
    }

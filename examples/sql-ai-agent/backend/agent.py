from __future__ import annotations
"""
Advanced SQL AI Agent — fully dynamic, schema-aware, strictly read-only.
Instrumented with MLflow tracing for full observability.

Author: Pradip Tivhale
"""

import re
from openai import AzureOpenAI

from config import get_settings
from database import build_full_context, execute_readonly_query
from tracing import traced_run_query

settings = get_settings()

client = AzureOpenAI(
    azure_endpoint=settings.azure_openai_endpoint,
    api_key=settings.azure_openai_api_key,
    api_version=settings.azure_openai_api_version,
)

# ─── Dynamic schema context (populated on startup) ─────────────────────────────

_db_context: str = ""


async def init_agent():
    """Called once on app startup to introspect the database."""
    global _db_context
    _db_context = await build_full_context()


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
{_db_context}
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


# ─── Core functions (called inside traced spans) ───────────────────────────────

async def _generate_sql(user_query: str, conversation_history: list[dict]) -> str:
    messages = [{"role": "system", "content": _build_system_prompt()}]
    for msg in conversation_history[-6:]:
        messages.append(msg)
    messages.append({
        "role": "user",
        "content": f'Write an efficient SQL query to answer: "{user_query}"\n\nReturn ONLY the SQL in a ```sql``` block.',
    })

    response = client.chat.completions.create(
        model=settings.azure_openai_deployment_name,
        messages=messages,
        max_completion_tokens=2000,
    )
    return _extract_sql(response.choices[0].message.content)


async def _narrate_result(user_query: str, sql: str, query_result: dict) -> str:
    columns = query_result["columns"]
    rows = query_result["rows"][:50]
    row_count = query_result["row_count"]
    exec_time = query_result["execution_time_ms"]

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
                "- Give the DIRECT answer first. If it's a single number, just state it plainly (e.g., '987 patients have abnormal blood pressure.')\n"
                "- Do NOT wrap single values in markdown tables. Only use tables when comparing 3+ rows of data.\n"
                "- Do NOT add disclaimers, caveats, or health advice. Just answer the question.\n"
                "- ONLY state facts from the SQL result. No invented numbers.\n"
                "- Keep it short — 1-3 sentences for simple queries, a brief table + 1 sentence for comparisons.\n"
                "- No emojis, no headings, no bullet points for simple answers."
            ),
        },
        {
            "role": "user",
            "content": (
                f"**Question:** {user_query}\n\n"
                f"**SQL:**\n```sql\n{sql}\n```\n\n"
                f"**Results** ({row_count} rows, {exec_time}ms):\n```\n{data_preview}```\n\n"
                f"Provide a clear answer."
            ),
        },
    ]

    response = client.chat.completions.create(
        model=settings.azure_openai_deployment_name,
        messages=messages,
        max_completion_tokens=2000,
    )
    return response.choices[0].message.content


async def _retry_generate_sql(
    user_query: str, failed_sql: str, error: str, conversation_history: list[dict]
) -> str:
    retry_msg = (
        f'SQL for "{user_query}" failed:\nError: {error}\n\n'
        f"Failed SQL:\n```sql\n{failed_sql}\n```\n\n"
        f"Fix the SQL. Return only corrected query in a ```sql``` block."
    )
    retry_messages = conversation_history[-4:] + [{"role": "user", "content": retry_msg}]
    messages = [{"role": "system", "content": _build_system_prompt()}] + retry_messages

    response = client.chat.completions.create(
        model=settings.azure_openai_deployment_name,
        messages=messages,
        max_completion_tokens=2000,
    )
    return _extract_sql(response.choices[0].message.content)


# ─── Public API ─────────────────────────────────────────────────────────────────

async def run_query(user_query: str, conversation_history: list[dict]) -> dict:
    """Full traced pipeline: NL → SQL → execute → narrate."""
    return await traced_run_query(
        user_query=user_query,
        conversation_history=conversation_history,
        generate_sql_fn=_generate_sql,
        execute_sql_fn=execute_readonly_query,
        narrate_fn=_narrate_result,
        retry_generate_fn=_retry_generate_sql,
    )

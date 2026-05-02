"""
Tool registry for the Agentic SQL Agent.

Four tools are exposed to the LLM:
  list_tables     — discover what tables exist
  describe_table  — inspect a table's schema, stats, and sample rows
  execute_sql     — run a read-only SQL query
  final_answer    — terminate the ReAct loop with the final answer

The LLM uses these tools in a ReAct loop:
  Thought → call a tool → observe result → Thought → ... → final_answer

Keeping the tool set small forces the agent to be precise.
A large tool set leads to decision paralysis and wasted steps.
"""

import json
from typing import Any

from database import execute_readonly_query, get_full_schema, get_all_col_stats_cached


# ── OpenAI-format tool definitions ────────────────────────────────────────────

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "list_tables",
            "description": (
                "List all tables available in the database with their row counts. "
                "Call this first when you are unsure which tables to query."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "describe_table",
            "description": (
                "Get the full schema of a specific table: column names, types, "
                "statistics (min/max/avg for numeric, top values for categorical), "
                "foreign keys, and 3 sample rows. "
                "Call this before writing SQL to understand exact column names and value formats."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "table_name": {
                        "type": "string",
                        "description": "Exact name of the table to inspect.",
                    }
                },
                "required": ["table_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_sql",
            "description": (
                "Execute a read-only SQL query and return the results. "
                "Only SELECT or WITH...SELECT queries are allowed. "
                "Returns columns, rows (up to 500), row_count, and execution_time_ms. "
                "You may call this multiple times to refine your answer."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "A valid SELECT or WITH...SELECT SQL query.",
                    }
                },
                "required": ["sql"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "final_answer",
            "description": (
                "Provide the final answer to the user's question. "
                "Call this ONLY when you have executed the necessary queries "
                "and are ready to give a complete, accurate answer. "
                "Do NOT call this before you have actual query results."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "Clear, factual answer based on the SQL results.",
                    },
                    "sql": {
                        "type": "string",
                        "description": "The final SQL query that produced the answer (may be empty for multi-step answers).",
                    },
                },
                "required": ["answer"],
            },
        },
    },
]

# Index for quick lookup by name
TOOL_MAP = {t["function"]["name"]: t for t in TOOL_DEFINITIONS}


# ── Tool execution ─────────────────────────────────────────────────────────────

async def execute_tool(name: str, arguments: dict) -> str:
    """
    Execute a tool by name and return a string result for the LLM to observe.
    Returns a compact, LLM-readable string — not raw Python objects.
    """
    try:
        if name == "list_tables":
            return await _list_tables()
        if name == "describe_table":
            return await _describe_table(arguments.get("table_name", ""))
        if name == "execute_sql":
            return await _execute_sql(arguments.get("sql", ""))
        return f"Unknown tool: {name}"
    except Exception as e:
        return f"Tool error ({name}): {e}"


async def _list_tables() -> str:
    schema = await get_full_schema()
    lines = [f"Available tables ({len(schema)} total):\n"]
    for table, info in schema.items():
        rows = info.get("row_count", "?")
        cols = len(info.get("columns", []))
        fks = [fk["referred_table"] for fk in info.get("foreign_keys", [])]
        fk_str = f"  → joins: {', '.join(fks)}" if fks else ""
        lines.append(f"  {table}: {cols} columns, {rows} rows{fk_str}")
    return "\n".join(lines)


async def _describe_table(table_name: str) -> str:
    if not table_name:
        return "Error: table_name is required."
    schema = await get_full_schema()
    if table_name not in schema:
        available = ", ".join(schema.keys())
        return f"Table '{table_name}' not found. Available: {available}"

    info = schema[table_name]
    col_stats = await get_all_col_stats_cached(schema)
    stats = col_stats.get(table_name, {})

    lines = [f"Table: {table_name}  ({info.get('row_count', '?')} rows)\n"]

    lines.append("Columns:")
    for col in info["columns"]:
        st = stats.get(col["name"], {})
        stat_str = ""
        if st.get("type") == "numeric":
            stat_str = f"  [min={st['min']}, max={st['max']}, avg={st['avg']}]"
        elif st.get("type") == "categorical":
            top = ", ".join(str(v[0]) for v in st.get("top_values", [])[:4])
            stat_str = f"  [distinct={st['distinct']}, values: {top}]"
        nullable = "" if col.get("nullable", True) else " NOT NULL"
        lines.append(f"  {col['name']}  {col['type']}{nullable}{stat_str}")

    if info.get("primary_key"):
        lines.append(f"\nPrimary key: {', '.join(info['primary_key'])}")

    if info.get("foreign_keys"):
        lines.append("\nForeign keys:")
        for fk in info["foreign_keys"]:
            lines.append(f"  {', '.join(fk['columns'])} → {fk['referred_table']}({', '.join(fk['referred_columns'])})")

    # Sample rows
    try:
        sample = await execute_readonly_query(f'SELECT * FROM "{table_name}" LIMIT 3')
        lines.append(f"\nSample rows (columns: {sample['columns']}):")
        for row in sample["rows"]:
            lines.append(f"  {row}")
    except Exception:
        pass

    return "\n".join(lines)


async def _execute_sql(sql: str) -> str:
    if not sql.strip():
        return "Error: sql is required."
    result = await execute_readonly_query(sql)
    lines = [
        f"Executed in {result['execution_time_ms']}ms  |  "
        f"{result['row_count']} rows"
        + (" (truncated to 500)" if result.get("truncated") else ""),
        f"Columns: {result['columns']}",
    ]
    for row in result["rows"][:20]:
        lines.append(f"  {row}")
    if result["row_count"] > 20:
        lines.append(f"  ... ({result['row_count'] - 20} more rows)")
    return "\n".join(lines)

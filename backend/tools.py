"""
Tool registry for the Agentic SQL Agent.

Five tools — in mandatory execution order:

  1. create_plan    — ALWAYS FIRST. Forces the agent to state its full strategy
                      before touching the database. Prevents guessing column names.

  2. list_tables    — discover available tables (use if plan needs table list)

  3. describe_table — inspect exact column names, types, sample values
                      MUST be called for every table before execute_sql

  4. execute_sql    — run a read-only SQL query (only after plan + describe)

  5. final_answer   — terminate the loop with the answer

The forced planning step is the single biggest improvement for first-attempt
success: the agent that plans explicitly gets column names right, identifies
JOINs correctly, and avoids NULL traps before writing a single line of SQL.
"""

import json
from typing import Any

from database import execute_readonly_query, get_full_schema, get_all_col_stats_cached


# ── OpenAI-format tool definitions ────────────────────────────────────────────

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "create_plan",
            "description": (
                "MANDATORY FIRST STEP — call this before any other tool. "
                "Create a structured execution plan that states exactly what you will do. "
                "This forces you to think through tables, JOINs, aggregations, and edge cases "
                "before touching the database — the single most effective way to solve queries "
                "correctly on the first attempt."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query_understanding": {
                        "type": "string",
                        "description": "What exactly is the user asking for? Restate it precisely.",
                    },
                    "tables_needed": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Table names you expect to use. Must describe_table each before execute_sql.",
                    },
                    "joins_needed": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "JOINs required, e.g. ['orders JOIN customers ON orders.customer_id = customers.id']. Empty if single-table query.",
                    },
                    "sql_approach": {
                        "type": "string",
                        "description": "SQL strategy: which aggregations, filters, GROUP BY, ORDER BY, CTEs you plan to use.",
                    },
                    "potential_issues": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Issues to watch for: NULL values, type casting, row explosion from bad JOINs, cardinality surprises.",
                    },
                },
                "required": ["query_understanding", "tables_needed", "sql_approach"],
            },
        },
    },
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
        if name == "create_plan":
            return _execute_plan(arguments)
        if name == "list_tables":
            return await _list_tables()
        if name == "describe_table":
            return await _describe_table(arguments.get("table_name", ""))
        if name == "execute_sql":
            return await _execute_sql(arguments.get("sql", ""))
        return f"Unknown tool: {name}"
    except Exception as e:
        return f"Tool error ({name}): {e}"


def _execute_plan(args: dict) -> str:
    """
    Acknowledge the plan and return a confirmation that prompts the agent
    to now execute it step by step.
    """
    tables  = args.get("tables_needed", [])
    joins   = args.get("joins_needed", [])
    approach = args.get("sql_approach", "")
    issues  = args.get("potential_issues", [])

    lines = [
        "Plan acknowledged. Execute it now in this exact order:",
        "",
        f"  Tables: {', '.join(tables) if tables else 'none specified'}",
        f"  Approach: {approach}",
    ]
    if joins:
        lines.append(f"  JOINs: {'; '.join(joins)}")
    if issues:
        lines.append(f"  Watch for: {'; '.join(issues)}")
    lines += [
        "",
        "NEXT STEPS:",
        "  1. Call describe_table for EACH table in your plan.",
        "  2. Only then call execute_sql with the verified column names.",
        "  3. Call final_answer once you have real results.",
    ]
    return "\n".join(lines)


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

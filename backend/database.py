from __future__ import annotations
"""
Database layer — fully dynamic schema introspection for ANY SQL database.
Supports SQLite, PostgreSQL, MySQL, etc. via SQLAlchemy async engine.
Strictly read-only query execution.

Author: Pradip Tivhale
"""

import time
from contextlib import asynccontextmanager

from sqlalchemy import text, inspect
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from config import get_settings

settings = get_settings()

engine = create_async_engine(
    settings.database_url,
    echo=False,
    pool_pre_ping=True,
)

AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


@asynccontextmanager
async def get_session():
    async with AsyncSessionLocal() as session:
        yield session


# ─── Dynamic schema introspection ──────────────────────────────────────────────

async def get_table_list() -> list[str]:
    """Return all table names in the database."""
    def _inspect(conn):
        return inspect(conn).get_table_names()
    async with engine.connect() as conn:
        return await conn.run_sync(_inspect)


async def get_full_schema() -> dict:
    """
    Introspect every table: columns, types, PKs, FKs, indexes.
    Returns a dict keyed by table name.
    """
    def _inspect(conn):
        insp = inspect(conn)
        tables = {}
        for table_name in insp.get_table_names():
            columns = []
            for col in insp.get_columns(table_name):
                columns.append({
                    "name": col["name"],
                    "type": str(col["type"]),
                    "nullable": col.get("nullable", True),
                    "default": str(col.get("default", "")) if col.get("default") else None,
                })

            pk = insp.get_pk_constraint(table_name)
            fks = insp.get_foreign_keys(table_name)
            indexes = insp.get_indexes(table_name)

            tables[table_name] = {
                "columns": columns,
                "primary_key": pk.get("constrained_columns", []),
                "foreign_keys": [
                    {
                        "columns": fk["constrained_columns"],
                        "referred_table": fk["referred_table"],
                        "referred_columns": fk["referred_columns"],
                    }
                    for fk in fks
                ],
                "indexes": [
                    {"name": idx["name"], "columns": idx["column_names"], "unique": idx.get("unique", False)}
                    for idx in indexes
                ],
            }
        return tables

    async with engine.connect() as conn:
        tables = await conn.run_sync(_inspect)

        # Row counts
        for table_name in tables:
            try:
                result = await conn.execute(text(f'SELECT COUNT(*) FROM "{table_name}"'))
                tables[table_name]["row_count"] = result.scalar()
            except Exception:
                tables[table_name]["row_count"] = "unknown"

    return tables


async def get_column_stats(table_name: str, columns: list[dict]) -> dict:
    """
    Auto-compute column-level statistics: min/max/avg for numeric,
    distinct count + top values for others.
    """
    stats = {}
    numeric_types = {"INTEGER", "REAL", "FLOAT", "DOUBLE", "NUMERIC", "DECIMAL", "BIGINT", "SMALLINT", "INT"}

    async with get_session() as session:
        for col in columns:
            col_name = col["name"]
            col_type = col["type"].upper().split("(")[0].strip()

            try:
                if col_type in numeric_types:
                    result = await session.execute(text(
                        f'SELECT MIN("{col_name}") AS min_val, MAX("{col_name}") AS max_val, '
                        f'ROUND(AVG("{col_name}"), 2) AS avg_val, '
                        f'COUNT(DISTINCT "{col_name}") AS distinct_count, '
                        f'SUM(CASE WHEN "{col_name}" IS NULL THEN 1 ELSE 0 END) AS null_count '
                        f'FROM "{table_name}"'
                    ))
                    row = result.fetchone()
                    stats[col_name] = {
                        "type": "numeric",
                        "min": row[0], "max": row[1], "avg": row[2],
                        "distinct": row[3], "nulls": row[4],
                    }
                else:
                    result = await session.execute(text(
                        f'SELECT COUNT(DISTINCT "{col_name}") AS distinct_count, '
                        f'SUM(CASE WHEN "{col_name}" IS NULL THEN 1 ELSE 0 END) AS null_count '
                        f'FROM "{table_name}"'
                    ))
                    row = result.fetchone()

                    top_result = await session.execute(text(
                        f'SELECT "{col_name}", COUNT(*) AS cnt FROM "{table_name}" '
                        f'WHERE "{col_name}" IS NOT NULL '
                        f'GROUP BY "{col_name}" ORDER BY cnt DESC LIMIT 10'
                    ))
                    top_values = [(r[0], r[1]) for r in top_result.fetchall()]

                    stats[col_name] = {
                        "type": "categorical",
                        "distinct": row[0], "nulls": row[1],
                        "top_values": top_values,
                    }
            except Exception:
                stats[col_name] = {"type": "unknown", "error": "could not compute stats"}

    return stats


async def get_sample_rows(table_name: str, limit: int = 5) -> dict:
    """Fetch sample rows from a table."""
    async with get_session() as session:
        result = await session.execute(text(f'SELECT * FROM "{table_name}" LIMIT {limit}'))
        columns = list(result.keys())
        rows = [list(r) for r in result.fetchall()]
    return {"columns": columns, "rows": rows}


async def build_full_context() -> str:
    """
    Build a comprehensive, auto-generated database context string.
    Works for ANY database — no hardcoded table/column knowledge.
    """
    schema = await get_full_schema()
    sections = []

    for table_name, info in schema.items():
        section = [f"## Table: `{table_name}` ({info['row_count']:,} rows)" if isinstance(info['row_count'], int) else f"## Table: `{table_name}`"]

        if info["primary_key"]:
            section.append(f"**Primary Key:** {', '.join(info['primary_key'])}")

        section.append("| Column | Type | Nullable |")
        section.append("|--------|------|----------|")
        for col in info["columns"]:
            section.append(f"| `{col['name']}` | {col['type']} | {col['nullable']} |")

        if info["foreign_keys"]:
            section.append("\n**Foreign Keys:**")
            for fk in info["foreign_keys"]:
                section.append(f"- `{', '.join(fk['columns'])}` -> `{fk['referred_table']}({', '.join(fk['referred_columns'])})`")

        if info["indexes"]:
            section.append(f"\n**Indexes:** {', '.join(idx['name'] for idx in info['indexes'])}")

        col_stats = await get_column_stats(table_name, info["columns"])
        section.append("\n**Column Statistics:**")
        for col_name, st in col_stats.items():
            if st["type"] == "numeric":
                section.append(f"- `{col_name}`: min={st['min']}, max={st['max']}, avg={st['avg']}, distinct={st['distinct']}, nulls={st['nulls']}")
            elif st["type"] == "categorical":
                top = ", ".join(f"{v}({c})" for v, c in st.get("top_values", [])[:5])
                section.append(f"- `{col_name}`: distinct={st['distinct']}, nulls={st['nulls']}, top_values=[{top}]")

        sample = await get_sample_rows(table_name, 3)
        section.append(f"\n**Sample rows:** `{sample['columns']}`")
        for row in sample["rows"]:
            section.append(f"  {row}")

        sections.append("\n".join(section))

    fk_summary = []
    for table_name, info in schema.items():
        for fk in info["foreign_keys"]:
            fk_summary.append(f"- `{table_name}.{', '.join(fk['columns'])}` -> `{fk['referred_table']}.{', '.join(fk['referred_columns'])}`")

    if fk_summary:
        sections.append("## Detected Relationships\n" + "\n".join(fk_summary))

    return "\n\n".join(sections)


# ─── Read-only query execution ─────────────────────────────────────────────────

FORBIDDEN_KEYWORDS = [
    "INSERT ", "UPDATE ", "DELETE ", "DROP ", "ALTER ", "CREATE ",
    "TRUNCATE ", "GRANT ", "REVOKE ", "ATTACH ", "DETACH ",
    "PRAGMA ", "VACUUM", "REINDEX",
]


async def execute_readonly_query(sql: str) -> dict:
    """
    Execute a strictly read-only SQL query.
    Returns {columns, rows, row_count, truncated, execution_time_ms}.
    """
    stripped = sql.strip().upper()

    if not (stripped.startswith("SELECT") or stripped.startswith("WITH")):
        raise ValueError("Only SELECT queries are allowed. This is a read-only system.")

    for kw in FORBIDDEN_KEYWORDS:
        if kw in stripped:
            raise ValueError(f"Forbidden operation: {kw.strip()}. This is a read-only system.")

    start = time.perf_counter()

    async with get_session() as session:
        result = await session.execute(text(sql))
        columns = list(result.keys())
        all_rows = result.fetchall()
        elapsed = (time.perf_counter() - start) * 1000

        max_rows = settings.max_result_rows
        truncated = len(all_rows) > max_rows
        rows = [list(r) for r in all_rows[:max_rows]]

    return {
        "columns": columns,
        "rows": rows,
        "row_count": len(all_rows),
        "truncated": truncated,
        "execution_time_ms": round(elapsed, 2),
    }

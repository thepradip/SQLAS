"""
Database layer — fully dynamic schema introspection for ANY SQL database.
Supports SQLite, PostgreSQL, MySQL, etc. via SQLAlchemy async engine.
Strictly read-only query execution.
"""

import asyncio
import hashlib
import json
import os
import time
from contextlib import asynccontextmanager

import sqlglot
from sqlglot import exp as sqlexp
from sqlalchemy import text, inspect
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from config import get_settings

settings = get_settings()

# SQLite uses NullPool and doesn't support pool_size/max_overflow.
# PostgreSQL/MySQL benefit from a proper connection pool.
_is_sqlite = "sqlite" in settings.database_url.lower()
_pool_kwargs = {} if _is_sqlite else {
    "pool_size":    20,    # max persistent connections
    "max_overflow": 40,    # extra connections under burst load
    "pool_timeout": 30,    # seconds before "pool exhausted" error
    "pool_recycle": 1800,  # recycle every 30 min (avoids MySQL 8h timeout)
}

engine = create_async_engine(
    settings.database_url,
    echo=False,
    pool_pre_ping=True,
    **_pool_kwargs,
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
    Auto-compute column-level statistics: min/max/avg for numeric, distinct count + top values for others.
    Works for any table and any column set.
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

                    # Top 10 most frequent values
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


_STATS_CACHE_FILE = ".schema_stats_cache.json"


def _schema_col_hash(schema: dict) -> str:
    """Hash based on table names + column definitions for cache invalidation."""
    sig = {t: sorted(c["name"] + str(c["type"]) for c in info["columns"]) for t, info in schema.items()}
    return hashlib.md5(json.dumps(sig, sort_keys=True).encode()).hexdigest()


async def get_all_col_stats(schema: dict) -> dict:
    """Compute column stats for every table. No caching."""
    all_stats: dict = {}
    for table_name, info in schema.items():
        all_stats[table_name] = await get_column_stats(table_name, info["columns"])
    return all_stats


async def get_all_col_stats_cached(schema: dict) -> dict:
    """
    Compute column stats for all tables with a JSON disk cache.
    Cache is invalidated whenever table or column definitions change.
    On a 100-table database this turns a multi-minute startup into ~1 second.
    """
    cache_hash = _schema_col_hash(schema)

    if os.path.exists(_STATS_CACHE_FILE):
        try:
            with open(_STATS_CACHE_FILE) as f:
                cache = json.load(f)
            if cache.get("hash") == cache_hash:
                print(f"  Schema stats cache hit — {len(cache['stats'])} tables loaded instantly.")
                return cache["stats"]
        except Exception:
            pass

    print(f"  Computing schema stats for {len(schema)} tables (first run, will be cached)...")
    all_stats = await get_all_col_stats(schema)

    try:
        with open(_STATS_CACHE_FILE, "w") as f:
            json.dump({"hash": cache_hash, "stats": all_stats}, f)
    except Exception:
        pass

    return all_stats


async def build_full_context() -> str:
    """
    Build a comprehensive database context string for small databases.
    Uses the stats cache so repeated restarts are instant.
    For large databases (100+ tables) use SchemaIndex.focused_context() instead.
    """
    from schema_index import _format_table_section

    schema = await get_full_schema()
    col_stats = await get_all_col_stats_cached(schema)
    sections = []

    for table_name, info in schema.items():
        sections.append(_format_table_section(table_name, info, col_stats.get(table_name, {})))

    fk_summary = []
    for table_name, info in schema.items():
        for fk in info["foreign_keys"]:
            fk_summary.append(
                f"- `{table_name}.{', '.join(fk['columns'])}` → "
                f"`{fk['referred_table']}.{', '.join(fk['referred_columns'])}`"
            )
    if fk_summary:
        sections.append("## Detected Relationships\n" + "\n".join(fk_summary))

    return "\n\n".join(sections)


# ─── Read-only query execution ─────────────────────────────────────────────────

_WRITE_NODE_TYPES = (
    sqlexp.Insert, sqlexp.Update, sqlexp.Delete, sqlexp.Drop,
    sqlexp.Create, sqlexp.Alter, sqlexp.Command,
)


def _validate_readonly_sql(sql: str) -> None:
    """
    Parse SQL into an AST with sqlglot and confirm it is strictly read-only.

    Catches what keyword matching misses:
      - write ops buried inside CTE definitions
      - write ops hidden in comments then un-commented
      - syntax errors (caught in <1ms, before hitting the DB)

    Raises ValueError with a user-friendly message on any violation.
    """
    stripped = sql.strip()
    upper = stripped.upper().lstrip()

    # Fast pre-check — must begin with SELECT or WITH
    if not (upper.startswith("SELECT") or upper.startswith("WITH")):
        raise ValueError("Only SELECT queries are allowed. This is a read-only system.")

    # Full AST validation
    try:
        statements = sqlglot.parse(stripped)
    except sqlglot.errors.ParseError as e:
        raise ValueError(f"SQL syntax error: {e}")

    if not statements or all(s is None for s in statements):
        raise ValueError("Empty or unparseable SQL.")

    for stmt in statements:
        if stmt is None:
            continue
        if not isinstance(stmt, (sqlexp.Select, sqlexp.With)):
            raise ValueError(
                f"Only SELECT queries are allowed. "
                f"Found statement type: {type(stmt).__name__}."
            )
        for node in stmt.walk():
            if isinstance(node, _WRITE_NODE_TYPES):
                raise ValueError(
                    f"Write operation detected inside query: {type(node).__name__}. "
                    "This is a read-only system."
                )


async def execute_readonly_query(sql: str) -> dict:
    """
    Execute a strictly read-only SQL query.

    Three protections vs. the old implementation:
      1. AST-level read-only validation via sqlglot (not keyword matching)
      2. Query timeout enforced via asyncio.wait_for
      3. fetchmany(max+1) — bounded memory, no OOM on large result sets
    """
    _validate_readonly_sql(sql)

    max_rows = settings.max_result_rows
    start = time.perf_counter()

    async with get_session() as session:
        try:
            result = await asyncio.wait_for(
                session.execute(text(sql)),
                timeout=float(settings.query_timeout_seconds),
            )
        except asyncio.TimeoutError:
            raise ValueError(
                f"Query timed out after {settings.query_timeout_seconds}s. "
                "Add a more specific WHERE clause or an explicit LIMIT."
            )

        columns = list(result.keys())
        rows_raw = result.fetchmany(max_rows + 1)   # fetch one extra to detect truncation
        elapsed = (time.perf_counter() - start) * 1000

    truncated = len(rows_raw) > max_rows
    rows = [list(r) for r in rows_raw[:max_rows]]

    return {
        "columns": columns,
        "rows": rows,
        "row_count": len(rows),
        "truncated": truncated,
        "execution_time_ms": round(elapsed, 2),
    }

"""
AriaSQL Multi-Tenant Row-Level Security.

Supports multiple tenants/users on one AriaSQL deployment with:
  - Table-level access control (tenant can only query allowed tables)
  - Row-level filtering (auto-inject WHERE clauses per user)
  - Per-tenant domain hints and PII column lists

Usage:
    registry = TenantRegistry()

    # Register a tenant
    registry.register(TenantConfig(
        tenant_id    = "acme_corp",
        api_key      = "acme-secret-key",
        allowed_tables = {"orders", "customers", "products"},
        row_filters  = {
            "orders":    "tenant_id = 'acme'",
            "customers": "org_id = 'acme'",
        },
        pii_columns  = ["email", "phone", "ssn"],
        domain_hint  = "ACME Corp e-commerce database.",
    ))

    # In FastAPI — resolve tenant from API key header
    tenant = registry.get_by_api_key(request.headers.get("X-API-Key"))

    # In execute_readonly_query — inject row filters
    sql_with_rls = apply_row_filters(sql, tenant)
"""

import hashlib
import json
import re
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TenantConfig:
    tenant_id: str
    api_key: str                         # hashed in storage
    allowed_tables: set[str] | None = None   # None = all tables
    row_filters: dict[str, str] = field(default_factory=dict)
    # {table_name: "WHERE clause WITHOUT the WHERE keyword"}
    pii_columns: list[str] | None = None     # None = use system defaults
    domain_hint: str = ""
    max_result_rows: int = 500
    created_at: float = 0.0


class TenantRegistry:
    """
    Stores and resolves tenant configurations.
    API key is stored as SHA-256 hash — never plaintext.
    """

    def __init__(self, db_path: str = ".tenants.db"):
        self._db_path = db_path
        self._by_id: dict[str, TenantConfig] = {}
        self._key_to_id: dict[str, str] = {}   # hashed_key -> tenant_id
        self._init_db()
        self._load_cache()

    def _init_db(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS tenants (
                    tenant_id      TEXT PRIMARY KEY,
                    api_key_hash   TEXT NOT NULL UNIQUE,
                    config_json    TEXT NOT NULL,
                    created_at     REAL NOT NULL
                );
            """)

    def _load_cache(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(
                "SELECT tenant_id, api_key_hash, config_json FROM tenants"
            ).fetchall()
        for tid, key_hash, cfg_json in rows:
            cfg_dict = json.loads(cfg_json)
            tc = TenantConfig(
                tenant_id=tid,
                api_key=key_hash,
                allowed_tables=set(cfg_dict.get("allowed_tables") or []) or None,
                row_filters=cfg_dict.get("row_filters", {}),
                pii_columns=cfg_dict.get("pii_columns"),
                domain_hint=cfg_dict.get("domain_hint", ""),
                max_result_rows=cfg_dict.get("max_result_rows", 500),
                created_at=cfg_dict.get("created_at", 0.0),
            )
            self._by_id[tid] = tc
            self._key_to_id[key_hash] = tid

    @staticmethod
    def _hash_key(api_key: str) -> str:
        return hashlib.sha256(api_key.encode()).hexdigest()

    def register(self, config: TenantConfig) -> str:
        """Register or update a tenant. Returns the tenant_id."""
        key_hash = self._hash_key(config.api_key)
        config.api_key = key_hash     # replace plaintext with hash
        if not config.created_at:
            config.created_at = time.time()

        cfg_dict = {
            "allowed_tables": list(config.allowed_tables) if config.allowed_tables else None,
            "row_filters": config.row_filters,
            "pii_columns": config.pii_columns,
            "domain_hint": config.domain_hint,
            "max_result_rows": config.max_result_rows,
            "created_at": config.created_at,
        }
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO tenants (tenant_id, api_key_hash, config_json, created_at)
                VALUES (?, ?, ?, ?)
            """, (config.tenant_id, key_hash, json.dumps(cfg_dict), config.created_at))
            conn.commit()

        self._by_id[config.tenant_id] = config
        self._key_to_id[key_hash] = config.tenant_id
        return config.tenant_id

    def get_by_api_key(self, api_key: str) -> Optional[TenantConfig]:
        """Resolve a TenantConfig from an API key. Returns None if not found."""
        if not api_key:
            return None
        key_hash = self._hash_key(api_key)
        tenant_id = self._key_to_id.get(key_hash)
        return self._by_id.get(tenant_id) if tenant_id else None

    def get(self, tenant_id: str) -> Optional[TenantConfig]:
        return self._by_id.get(tenant_id)

    def list_tenants(self) -> list[dict]:
        return [{"tenant_id": t.tenant_id, "domain_hint": t.domain_hint[:60],
                 "allowed_tables": len(t.allowed_tables) if t.allowed_tables else "all"}
                for t in self._by_id.values()]


def check_table_access(sql: str, tenant: Optional[TenantConfig]) -> tuple[bool, str]:
    """
    Verify the SQL only references tables the tenant is allowed to access.

    Returns:
        (allowed: bool, blocked_table: str)
    """
    if tenant is None or tenant.allowed_tables is None:
        return True, ""

    try:
        import sqlglot
        parsed = sqlglot.parse_one(sql)
        referenced = {t.name.lower() for t in parsed.find_all(sqlglot.exp.Table) if t.name}
    except Exception:
        # Fallback: regex
        referenced = set(re.findall(r'\bFROM\s+(\w+)|\bJOIN\s+(\w+)', sql, re.IGNORECASE))
        referenced = {t for pair in referenced for t in pair if t}

    allowed_lower = {t.lower() for t in tenant.allowed_tables}
    blocked = referenced - allowed_lower
    if blocked:
        return False, next(iter(blocked))
    return True, ""


def apply_row_filters(sql: str, tenant: Optional[TenantConfig]) -> str:
    """
    Inject tenant row filters into a SQL query.

    For each table in tenant.row_filters, adds a WHERE / AND clause so
    the tenant can only see their own rows.

    This is a best-effort implementation using sqlglot AST rewriting.
    Falls back gracefully if parsing fails.

    Example:
        tenant.row_filters = {"orders": "tenant_id = 'acme'"}
        apply_row_filters("SELECT * FROM orders", tenant)
        → "SELECT * FROM orders WHERE tenant_id = 'acme'"
    """
    if not tenant or not tenant.row_filters:
        return sql

    try:
        import sqlglot
        from sqlglot import exp

        tree = sqlglot.parse_one(sql)

        for select in tree.find_all(exp.Select):
            for table in select.find_all(exp.Table):
                table_lower = (table.name or "").lower()
                filter_clause = tenant.row_filters.get(table_lower)
                if not filter_clause:
                    continue

                # Build the filter condition
                filter_expr = sqlglot.parse_one(filter_clause)

                # Find or create WHERE clause for this SELECT
                where = select.find(exp.Where)
                if where:
                    # AND the new filter into the existing WHERE
                    where.set("this", exp.And(this=where.this, expression=filter_expr))
                else:
                    select.set("where", exp.Where(this=filter_expr))

        return tree.sql()

    except Exception:
        # Fallback: simple string injection for single-table queries
        for table_name, filter_clause in tenant.row_filters.items():
            pattern = rf'\bFROM\s+{re.escape(table_name)}\b'
            if re.search(pattern, sql, re.IGNORECASE):
                if "WHERE" in sql.upper():
                    sql = re.sub(r'\bWHERE\b', f'WHERE {filter_clause} AND', sql, count=1, flags=re.IGNORECASE)
                else:
                    sql = re.sub(pattern, f'FROM {table_name} WHERE {filter_clause}', sql, flags=re.IGNORECASE)
        return sql

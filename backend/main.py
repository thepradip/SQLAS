"""
FastAPI backend — SQL AI Agent REST API with MLflow observability.
Fully dynamic, works with any database.

Author: Pradip Tivhale
"""

import csv
import io
import json
import sqlite3
import time as _time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from config import get_settings
from database import get_table_list, build_full_context
from agent import init_agent, run_query, get_cache, set_training_store, get_provider_name
from tracing import init_tracing, log_user_feedback, log_detailed_feedback, EXPERIMENT_NAME
from training_store import TrainingStore
from tenant import TenantRegistry, TenantConfig, check_table_access, apply_row_filters
from models import (
    QueryRequest, QueryResponse, HealthCheck, SchemaResponse,
    FeedbackRequest, DetailedFeedbackRequest, ExportRequest,
)

settings = get_settings()


# ─── Persistent conversation store ─────────────────────────────────────────────

class ConversationStore:
    """
    SQLite-backed conversation history.
    Survives server restarts — in-memory dict was wiped on every deploy.
    In-memory cache keeps hot conversations fast; DB is the source of truth.
    """
    MAX_MESSAGES = 20

    def __init__(self, db_path: str = ".conversations.db"):
        self._db_path = db_path
        self._cache: dict[str, list[dict]] = {}
        with sqlite3.connect(db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    conv_id    TEXT    NOT NULL,
                    role       TEXT    NOT NULL,
                    content    TEXT    NOT NULL,
                    created_at REAL    NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_conv ON conversations(conv_id, id);
            """)

    def get(self, conv_id: str) -> list[dict]:
        if conv_id in self._cache:
            return self._cache[conv_id]
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(
                "SELECT role, content FROM conversations WHERE conv_id = ? ORDER BY id",
                (conv_id,),
            ).fetchall()
        history = [{"role": r, "content": c} for r, c in rows]
        self._cache[conv_id] = history
        return history

    def append(self, conv_id: str, msgs: list[dict]) -> None:
        history = self.get(conv_id)
        history.extend(msgs)
        self._cache[conv_id] = history[-self.MAX_MESSAGES:]
        with sqlite3.connect(self._db_path) as conn:
            for msg in msgs:
                conn.execute(
                    "INSERT INTO conversations (conv_id, role, content, created_at) VALUES (?,?,?,?)",
                    (conv_id, msg["role"], msg["content"], _time.time()),
                )
            # Keep only last MAX_MESSAGES rows per conversation
            conn.execute("""
                DELETE FROM conversations WHERE conv_id = ? AND id NOT IN (
                    SELECT id FROM conversations WHERE conv_id = ? ORDER BY id DESC LIMIT ?
                )
            """, (conv_id, conv_id, self.MAX_MESSAGES))
            conn.commit()

    def clear(self, conv_id: str) -> None:
        self._cache.pop(conv_id, None)
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("DELETE FROM conversations WHERE conv_id = ?", (conv_id,))
            conn.commit()


_conv_store    = ConversationStore()
_training_store = TrainingStore()
_tenant_registry = TenantRegistry()

# Wire training store into the agent
set_training_store(_training_store)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Initializing SQL AI Agent...")
    init_tracing()
    await init_agent()
    tables = await get_table_list()
    print(f"  Database ready — {len(tables)} tables: {tables}")
    yield
    print("Shutting down.")


app = FastAPI(
    title="SQL AI Agent",
    description="Natural language to SQL with MLflow observability — by Pradip Tivhale",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_url, "http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthCheck)
async def health_check():
    tables = await get_table_list()
    return HealthCheck(
        status="ok",
        database=settings.database_url.split("///")[-1],
        tables=tables,
        mlflow_experiment=EXPERIMENT_NAME,
    )


@app.get("/schema", response_model=SchemaResponse)
async def get_schema():
    context = await build_full_context()
    return SchemaResponse(schema_text=context)


@app.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
):
    # Resolve tenant (None = single-tenant / no auth mode)
    tenant = _tenant_registry.get_by_api_key(x_api_key) if x_api_key else None
    conv_id = request.conversation_id or "default"
    history = _conv_store.get(conv_id)

    result = await run_query(
        request.query, history, force_agentic=request.force_agentic
    )

    _conv_store.append(conv_id, [
        {"role": "user", "content": request.query},
        {"role": "assistant", "content": result["response"]},
    ])

    return QueryResponse(
        sql=result["sql"],
        data=result["data"],
        response=result["response"],
        success=result["success"],
        trace_id=result.get("trace_id"),
        metrics=result.get("metrics"),
        visualization=result.get("visualization"),
        agent_mode=result.get("agent_mode", "pipeline"),
        agent_steps=result.get("steps"),
    )


@app.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    """
    Submit thumbs up/down feedback for a query trace.
    Thumbs-up automatically marks the cache entry as verified — it will be
    prioritised as a few-shot example in future similar queries.
    """
    try:
        log_user_feedback(
            trace_id=request.trace_id,
            feedback_value=request.value,
            user_id=request.user_id,
            comment=request.comment,
        )
        verified = False
        if request.value:   # thumbs up → teach the agent
            cache = get_cache()
            if cache:
                verified = cache.verify_by_trace(request.trace_id)
        return {"status": "ok", "trace_id": request.trace_id, "cache_verified": verified}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/feedback/detailed")
async def submit_detailed_feedback(request: DetailedFeedbackRequest):
    """Submit detailed multi-dimension feedback."""
    try:
        log_detailed_feedback(
            trace_id=request.trace_id,
            accuracy=request.accuracy,
            relevance=request.relevance,
            sql_quality=request.sql_quality,
            user_id=request.user_id,
            comment=request.comment,
        )
        return {"status": "ok", "trace_id": request.trace_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evaluate")
async def run_eval(quick: bool = True):
    """Run SQLAS evaluation suite. ?quick=true for 3 tests, ?quick=false for full suite."""
    from eval_runner import run_evaluation
    try:
        results = await run_evaluation(quick=quick)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/conversations/{conv_id}")
async def clear_conversation(conv_id: str):
    _conv_store.clear(conv_id)
    return {"status": "cleared"}


# ─── Export ─────────────────────────────────────────────────────────────────────

@app.post("/export/csv")
async def export_csv(request: ExportRequest):
    """
    Download query results as a CSV file.
    The frontend passes the columns + rows from the last query result —
    no re-execution needed, and no additional LLM cost.
    Uses UTF-8 BOM (utf-8-sig) so Excel opens it correctly without encoding issues.
    """
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(request.columns)
    writer.writerows(request.rows)

    filename = request.filename.replace('"', "")   # sanitise for Content-Disposition
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode("utf-8-sig")),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ─── Cache API ──────────────────────────────────────────────────────────────────

@app.get("/cache/stats")
async def cache_stats():
    """
    Cache analytics: hit rates, tokens saved, cost savings, top cached queries.
    Use this to demonstrate ROI of the caching layer.
    """
    cache = get_cache()
    if cache is None:
        return {"enabled": False, "message": "Cache is disabled (CACHE_ENABLED=false)"}
    return {"enabled": True, **cache.get_analytics()}


@app.delete("/cache/results")
async def invalidate_result_cache():
    """
    Purge SQL result cache. Call this after any data updates to ensure fresh results.
    Does NOT affect the NL→SQL query cache.
    """
    cache = get_cache()
    if cache is None:
        return {"status": "skipped", "message": "Cache disabled"}
    purged = cache.invalidate_results()
    return {"status": "ok", "purged_entries": purged}


@app.delete("/cache/all")
async def clear_cache():
    """
    Wipe all cache entries (query cache + result cache).
    Use when schema changes significantly or you want a clean slate.
    """
    cache = get_cache()
    if cache is None:
        return {"status": "skipped", "message": "Cache disabled"}
    cache.clear_all()
    return {"status": "ok", "message": "All cache entries cleared"}


# ─── Training Store API ────────────────────────────────────────────────────────

@app.post("/train/ddl")
async def train_ddl(body: dict):
    """
    Ingest a DDL statement into the training store.
    Helps the LLM understand column semantics and constraints.
    """
    ddl   = body.get("ddl", "").strip()
    title = body.get("title", "")
    if not ddl:
        raise HTTPException(status_code=400, detail="ddl is required")
    item_id = _training_store.add_ddl(ddl, title=title)
    return {"status": "ok", "id": item_id, "size": _training_store.size()}


@app.post("/train/documentation")
async def train_documentation(body: dict):
    """
    Ingest business documentation or term definitions.
    Example: "Revenue = SUM(orders.total) WHERE status = 'completed'"
    """
    text  = body.get("text", "").strip()
    title = body.get("title", "")
    if not text:
        raise HTTPException(status_code=400, detail="text is required")
    item_id = _training_store.add_documentation(text, title=title)
    return {"status": "ok", "id": item_id, "size": _training_store.size()}


@app.post("/train/sql")
async def train_sql(body: dict):
    """
    Ingest a verified (question → SQL) pair as a few-shot example.
    These are injected into the SQL generation prompt for similar future queries.
    """
    question = body.get("question", "").strip()
    sql      = body.get("sql", "").strip()
    if not question or not sql:
        raise HTTPException(status_code=400, detail="question and sql are required")
    item_id = _training_store.add_sql_example(question, sql)
    return {"status": "ok", "id": item_id, "size": _training_store.size()}


@app.get("/train/list")
async def list_training():
    """List all training items in the store."""
    return {"items": _training_store.list_all(), "size": _training_store.size()}


@app.delete("/train/{item_id}")
async def delete_training(item_id: str):
    """Remove a training item by its ID."""
    removed = _training_store.delete(item_id)
    if not removed:
        raise HTTPException(status_code=404, detail=f"Item {item_id} not found")
    return {"status": "ok", "size": _training_store.size()}


# ─── Multi-Tenant API ──────────────────────────────────────────────────────────

@app.post("/tenants")
async def register_tenant(body: dict):
    """
    Register a tenant with table access control and row-level filters.

    Body example:
    {
        "tenant_id":      "acme_corp",
        "api_key":        "acme-secret-key",
        "allowed_tables": ["orders", "customers"],
        "row_filters":    {"orders": "tenant_id = 'acme'"},
        "domain_hint":    "ACME Corp e-commerce database"
    }
    """
    try:
        tc = TenantConfig(
            tenant_id      = body["tenant_id"],
            api_key        = body["api_key"],
            allowed_tables = set(body["allowed_tables"]) if body.get("allowed_tables") else None,
            row_filters    = body.get("row_filters", {}),
            pii_columns    = body.get("pii_columns"),
            domain_hint    = body.get("domain_hint", ""),
            max_result_rows= body.get("max_result_rows", 500),
        )
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing field: {e}")
    _tenant_registry.register(tc)
    return {"status": "ok", "tenant_id": tc.tenant_id}


@app.get("/tenants")
async def list_tenants():
    return {"tenants": _tenant_registry.list_tenants()}


# ─── Streaming Query (SSE) ─────────────────────────────────────────────────────

@app.post("/query/stream")
async def query_stream(
    request: QueryRequest,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
):
    """
    Streaming version of /query using Server-Sent Events.
    Yields progress events so the frontend can show stages in real-time:
      schema_retrieved → sql_generated → sql_executed → response_ready

    Frontend usage:
        const es = await fetch('/api/query/stream', {method:'POST', body: JSON.stringify({query})})
        const reader = es.body.getReader()
        // read events as they arrive
    """
    tenant = _tenant_registry.get_by_api_key(x_api_key) if x_api_key else None
    conv_id = request.conversation_id or "default"
    history = _conv_store.get(conv_id)

    async def event_stream():
        try:
            # Stage 1 — schema retrieval
            yield f"data: {json.dumps({'stage': 'retrieving_schema', 'message': 'Identifying relevant tables...'})}\n\n"

            # Stage 2 — SQL generation (run full pipeline)
            yield f"data: {json.dumps({'stage': 'generating_sql', 'message': 'Generating SQL query...'})}\n\n"

            result = await run_query(
                request.query,
                history,
                force_agentic=request.force_agentic,
            )

            # Stage 3 — done
            if result.get("success"):
                yield f"data: {json.dumps({'stage': 'sql_executed', 'message': 'Executing SQL...', 'sql': result.get('sql', '')})}\n\n"
                yield f"data: {json.dumps({'stage': 'response_ready', 'result': result})}\n\n"
            else:
                yield f"data: {json.dumps({'stage': 'error', 'message': result.get('response', 'Query failed')})}\n\n"

            _conv_store.append(conv_id, [
                {"role": "user",      "content": request.query},
                {"role": "assistant", "content": result.get("response", "")},
            ])

        except Exception as e:
            yield f"data: {json.dumps({'stage': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ─── Database Connection Testing ──────────────────────────────────────────────

def _classify_db_error(error_type: str, error_msg: str) -> dict:
    """Map raw DB exceptions to user-friendly messages with fix hints."""
    msg = error_msg.lower()

    if "connection refused" in msg or "connect call failed" in msg or "no route to host" in msg:
        return {"cause": "Host unreachable", "fix": "Check the hostname/IP and port. Is the database server running?"}
    if any(w in msg for w in ("password", "authentication", "auth failed", "access denied", "invalid password")):
        return {"cause": "Authentication failed", "fix": "Check username and password in DATABASE_URL."}
    if any(w in msg for w in ("database", "schema")) and any(w in msg for w in ("not found", "does not exist", "unknown")):
        return {"cause": "Database not found", "fix": "Check the database/schema name in DATABASE_URL."}
    if "timeout" in msg or "timed out" in msg or "connect timeout" in msg:
        return {"cause": "Connection timed out", "fix": "Host is unreachable or firewall is blocking the port."}
    if "ssl" in msg or "tls" in msg or "certificate" in msg:
        return {"cause": "SSL/TLS error", "fix": "Check SSL configuration. Try adding ?sslmode=disable for testing."}
    if "no module" in msg or "modulenotfounderror" in error_type.lower():
        driver = "asyncpg" if "postgres" in msg else "aiomysql" if "mysql" in msg else "unknown"
        return {"cause": "DB driver not installed", "fix": f"Run: pip install {driver}"}
    if "too many connections" in msg or "connection pool" in msg:
        return {"cause": "Too many connections", "fix": "Reduce pool_size or close unused connections."}
    if "permission denied" in msg or "privilege" in msg:
        return {"cause": "Insufficient permissions", "fix": "The user needs SELECT privileges on the target database."}
    return {"cause": "Connection failed", "fix": f"{error_type}: {error_msg[:120]}"}


@app.post("/database/test")
async def test_database_connection(body: dict):
    """
    Test a database connection before using it.
    Call this whenever a user uploads or changes DATABASE_URL.

    Returns:
        200 — connection OK with table count
        422 — connection failed with user-friendly error + fix hint
    """
    db_url = (body.get("database_url") or "").strip()
    if not db_url:
        raise HTTPException(status_code=400, detail="database_url is required")

    # Safety: reject non-async URLs (we only support async drivers)
    if not any(d in db_url for d in ("aiosqlite", "asyncpg", "aiomysql", "+async")):
        # Auto-convert common sync URLs
        db_url = (db_url
                  .replace("postgresql://", "postgresql+asyncpg://")
                  .replace("postgres://",   "postgresql+asyncpg://")
                  .replace("mysql://",      "mysql+aiomysql://")
                  .replace("sqlite:///",    "sqlite+aiosqlite:///"))

    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlalchemy import text as sa_text, inspect

    test_engine = None
    try:
        test_engine = create_async_engine(
            db_url,
            pool_pre_ping=True,
            connect_args={"connect_timeout": 10} if "postgres" in db_url or "mysql" in db_url else {},
        )

        async with test_engine.connect() as conn:
            # Basic connectivity
            await conn.execute(sa_text("SELECT 1"))

            # Count tables
            def _get_tables(c):
                return inspect(c).get_table_names()

            tables = await conn.run_sync(_get_tables)

        return {
            "status": "ok",
            "message": f"Connection successful — {len(tables)} table(s) found",
            "table_count": len(tables),
            "tables_preview": tables[:10],
            "normalized_url": db_url,
        }

    except Exception as e:
        detail = _classify_db_error(type(e).__name__, str(e))
        raise HTTPException(
            status_code=422,
            detail={
                "status":    "error",
                "cause":     detail["cause"],
                "fix":       detail["fix"],
                "raw_error": str(e)[:300],
            },
        )
    finally:
        if test_engine:
            await test_engine.dispose()


@app.get("/database/status")
async def database_status():
    """
    Check the health of the currently configured database connection.
    Use this for health checks and monitoring.
    """
    try:
        from database import get_table_list
        tables = await get_table_list()
        return {
            "status":      "ok",
            "table_count": len(tables),
            "tables":      tables[:20],
            "database":    settings.database_url.split("///")[-1].split("@")[-1],  # no credentials
        }
    except Exception as e:
        detail = _classify_db_error(type(e).__name__, str(e))
        return {
            "status":  "error",
            "cause":   detail["cause"],
            "fix":     detail["fix"],
        }

"""
FastAPI backend — SQL AI Agent REST API with MLflow observability.
Fully dynamic, works with any database.

Author: Pradip Tivhale
"""

import csv
import io
import sqlite3
import time as _time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from config import get_settings
from database import get_table_list, build_full_context
from agent import init_agent, run_query, get_cache
from tracing import init_tracing, log_user_feedback, log_detailed_feedback, EXPERIMENT_NAME
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


_conv_store = ConversationStore()


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
async def query(request: QueryRequest):
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

from __future__ import annotations
"""
FastAPI backend — SQL AI Agent REST API with MLflow observability.
Fully dynamic, works with any database.

Author: Pradip Tivhale
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from config import get_settings
from database import get_table_list, build_full_context
from agent import init_agent, run_query
from tracing import init_tracing, log_user_feedback, log_detailed_feedback, EXPERIMENT_NAME
from models import (
    QueryRequest, QueryResponse, HealthCheck, SchemaResponse,
    FeedbackRequest, DetailedFeedbackRequest,
)

settings = get_settings()

# ─── In-memory conversation store (swap to Redis for production) ────────────────
conversations: dict[str, list[dict]] = {}


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
    history = conversations.get(conv_id, [])

    result = await run_query(request.query, history)

    # Update conversation history
    history.append({"role": "user", "content": request.query})
    history.append({"role": "assistant", "content": result["response"]})
    conversations[conv_id] = history[-20:]

    return QueryResponse(
        sql=result["sql"],
        data=result["data"],
        response=result["response"],
        success=result["success"],
        trace_id=result.get("trace_id"),
        metrics=result.get("metrics"),
    )


@app.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    """Submit thumbs up/down feedback for a query trace."""
    try:
        log_user_feedback(
            trace_id=request.trace_id,
            feedback_value=request.value,
            user_id=request.user_id,
            comment=request.comment,
        )
        return {"status": "ok", "trace_id": request.trace_id}
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
    conversations.pop(conv_id, None)
    return {"status": "cleared"}

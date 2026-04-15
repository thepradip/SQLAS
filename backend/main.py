from __future__ import annotations
"""
FastAPI backend — SQL AI Agent REST API powered by LangGraph and SQLAS.

Author: Pradip Tivhale
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from config import get_settings
from database import get_table_list, build_full_context
from agent.graph import run_query
from agent.nodes import init_schema
from models import (
    QueryRequest, QueryResponse, HealthCheck, SchemaResponse,
    FeedbackRequest, DetailedFeedbackRequest, SQLASScoresResponse,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

settings = get_settings()

# ─── In-memory conversation store (swap to Redis for production) ──────────────
conversations: dict[str, list[dict]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing SQL AI Agent (LangGraph + SQLAS)...")
    await init_schema()
    tables = await get_table_list()
    logger.info("Database ready — %d tables: %s", len(tables), tables)
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="SQL AI Agent",
    description="RAG-based NL-to-SQL Agent powered by LangGraph with SQLAS evaluation — by Pradip Tivhale",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_url, "http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthCheck)
async def health_check():
    tables = await get_table_list()
    return HealthCheck(
        status="ok",
        database=settings.database_url.split("///")[-1],
        tables=tables,
        agent_type="LangGraph + SQLAS",
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

    # Build SQLAS scores response
    sqlas_scores = None
    if result.get("sqlas_scores"):
        sqlas_scores = SQLASScoresResponse(**result["sqlas_scores"])

    return QueryResponse(
        sql=result["sql"],
        data=result["data"],
        response=result["response"],
        success=result["success"],
        trace_id=result.get("trace_id"),
        metrics=result.get("metrics"),
        sqlas_scores=sqlas_scores,
    )


@app.post("/evaluate")
async def run_eval(quick: bool = True):
    """Run SQLAS evaluation suite. ?quick=true for 5 tests, ?quick=false for full 25."""
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

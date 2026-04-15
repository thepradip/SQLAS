# SQL AI Agent

**RAG-based Natural Language to SQL Agent powered by LangGraph, with SQLAS production evaluation.**

Converts natural language questions into SQL queries, executes them safely, and returns narrated answers — all orchestrated as a LangGraph `StateGraph` with SQLAS quality gates at every stage.

**Author:** [Pradip Tivhale](https://github.com/thepradip)

---

## Architecture

```
                    LangGraph StateGraph
                    ====================

User Question ──► [retrieve_schema]
                        │
                        ▼
                  [generate_sql] ◄──────────────┐
                        │                       │
                        ▼                       │
                  [validate_sql]          [handle_error]
                  (SQLAS safety gate)           │
                   ┌────┴────┐                  │
                   │         │                  │
                safe?     unsafe?               │
                   │         │                  │
                   ▼         ▼                  │
            [execute_sql] [reject_unsafe]──►END │
               ┌───┴───┐                       │
               │       │                       │
            success  failure ───────────────────┘
               │                          (retry up to 2x)
               ▼
         [narrate_result]
               │
               ▼
        [evaluate_quality]
         (SQLAS scoring)
               │
               ▼
              END
```

### SQLAS Integration Points

| Stage | SQLAS Metric | Purpose |
|-------|-------------|---------|
| **Pre-execution gate** | `read_only_compliance` | Blocks DDL/DML before execution |
| | `safety_score` | Detects PII exposure, SQL injection |
| | `schema_compliance` | Validates tables/columns exist |
| **Post-response scoring** | `evaluate()` | Full 20-metric production score |
| **Evaluation suite** | `run_suite()` | Batch evaluation with 25 test cases |

---

## Features

- **LangGraph orchestration** — proper `StateGraph` with conditional edges, self-healing retry loops, and safety gates
- **SQLAS evaluation** — production-grade scoring at validation and response stages via [`sqlas`](https://github.com/thepradip/SQLAS)
- **Zero-config schema discovery** — auto-introspects tables, columns, PKs, FKs, indexes, column stats, sample rows
- **Scales to 100s of tables** — dynamic introspection, no hardcoded schema
- **Multiple database backends** — SQLite, PostgreSQL, MySQL, SQL Server, Oracle (any SQLAlchemy URL)
- **Strictly read-only** — enforced at SQL parsing, SQLAS validation, and execution layers
- **Self-healing SQL** — automatic retry with error feedback through LangGraph loop
- **React chat UI** — conversational interface with syntax-highlighted SQL and data tables

---

## Quick Start

### 1. Clone & configure

```bash
git clone https://github.com/thepradip/SQL-AI-Agent.git
cd SQL-AI-Agent
```

### 2. Backend

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Copy and configure environment
cp .env.example .env
# Edit .env with your Azure OpenAI credentials

# Ingest sample health data into SQLite (required for demo)
python ingest.py

# Start the API
uvicorn main:app --reload
```

FastAPI server: `http://localhost:8000`
API docs: `http://localhost:8000/docs`

### 3. Frontend

```bash
cd frontend
npm install
npm run dev
```

React app: `http://localhost:5173`

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check — DB info + agent type |
| GET | `/schema` | Full auto-discovered schema context |
| POST | `/query` | NL -> SQL -> Execute -> Narrate (with SQLAS scores) |
| POST | `/evaluate` | Run SQLAS evaluation suite (25 test cases) |
| DELETE | `/conversations/{id}` | Clear conversation history |

### Query Response

Every `/query` response includes SQLAS scores:

```json
{
  "sql": "SELECT COUNT(*) FROM users WHERE active = 1",
  "data": { "columns": ["COUNT(*)"], "rows": [[1523]], "row_count": 1 },
  "response": "There are 1,523 active users.",
  "success": true,
  "sqlas_scores": {
    "overall_score": 0.92,
    "execution_accuracy": 1.0,
    "faithfulness": 0.95,
    "safety_score": 1.0
  }
}
```

---

## Connecting Your Own Database

1. Set `DATABASE_URL` in `.env` to your database connection string
2. Restart the backend — schema is auto-discovered at startup
3. Optionally set `DOMAIN_HINT` for better LLM context
4. Optionally set `PII_COLUMNS` for SQLAS safety scoring

No code changes needed. The agent introspects all tables and generates context automatically.

---

## Evaluation

Run the built-in SQLAS evaluation suite:

```bash
# Quick mode (5 test cases)
curl -X POST "http://localhost:8000/evaluate?quick=true"

# Full suite (25 test cases across 4 difficulty tiers)
curl -X POST "http://localhost:8000/evaluate?quick=false"
```

Or from Python:

```bash
cd backend
python eval_runner.py          # full suite
python eval_runner.py --quick  # quick mode
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Agent orchestration | [LangGraph](https://github.com/langchain-ai/langgraph) |
| Evaluation | [SQLAS](https://github.com/thepradip/SQLAS) |
| LLM | Azure OpenAI (configurable) |
| Backend | FastAPI + SQLAlchemy (async) |
| Frontend | React + Vite + Tailwind CSS |
| Database | Any SQLAlchemy-compatible DB |

---

## License

MIT License - [Pradip Tivhale](https://github.com/thepradip)

<p align="center">
  <img src="docs/sqlas_logo.png" alt="SQLAS Logo" width="240"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/LangGraph-Agent-blueviolet?style=for-the-badge&logo=langchain&logoColor=white" alt="LangGraph"/>
  <img src="https://img.shields.io/badge/SQLAS-Evaluation-orange?style=for-the-badge" alt="SQLAS"/>
  <img src="https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/React-Frontend-61DAFB?style=for-the-badge&logo=react&logoColor=black" alt="React"/>
</p>

<h1 align="center">SQL AI Agent</h1>

<p align="center">
  <strong>RAG-based Natural Language to SQL Agent powered by LangGraph, with SQLAS production evaluation.</strong>
</p>

Converts natural language questions into SQL queries, executes them safely, and returns narrated answers — all orchestrated as a LangGraph `StateGraph` with SQLAS quality gates at every stage.

**Author:** [Pradip Tivhale](https://github.com/thepradip)

---

## System Architecture

<p align="center">
  <img src="docs/architecture/arc2.png" alt="Architectural Blueprint: LLM-Driven SQL Analytics Pipeline" width="100%"/>
</p>

<p align="center"><em>Architectural Blueprint — 4 layers: Frontend, Backend & API Gateway, Core LangGraph Pipeline, Infrastructure</em></p>

| Layer | Components | Responsibility |
|-------|-----------|----------------|
| **Frontend** | Chat UI, SQL Viewer, Data Tables, Metrics Panel | User-facing interface (React + Vite + Tailwind CSS) |
| **Backend & API Gateway** | FastAPI, Schema Introspection, Metrics Engine | REST API orchestration, JSON request handling |
| **Core LangGraph Pipeline** | 7 nodes: retrieve_schema -> generate_sql -> validate_sql -> execute_sql -> narrate_result -> evaluate_quality | Agent orchestration with SQLAS safety gates and self-healing retry |
| **Infrastructure** | Azure OpenAI, SQLite/PostgreSQL/MySQL, SQLAS | LLM reasoning, data storage, production evaluation |

> Full documentation: [`docs/agent/DOCUMENTATION.pdf`](docs/agent/DOCUMENTATION.pdf) (13 sections, 30+ pages)

---

## LangGraph Pipeline (Detail)

7 nodes, 3 conditional edges, self-healing retry loop:

```
                        ┌─────────────────────────────────────────┐
                        │           AgentState (TypedDict)         │
                        │                                         │
                        │  question, conversation_history,        │
                        │  schema_context, generated_sql,         │
                        │  is_safe, safety_details,               │
                        │  execution_result, execution_error,     │
                        │  response, sqlas_scores,                │
                        │  retry_count, metrics, success          │
                        └───────────────┬─────────────────────────┘
                                        │
                        START ──► retrieve_schema
                                        │
                             schema_context populated
                                        │
                                        ▼
                                  generate_sql ◄────────────────┐
                                        │                       │
                             generated_sql populated             │
                                        │                       │
                                        ▼                       │
                                  validate_sql                  │
                              ┌─── SQLAS Safety Gate ───┐       │
                              │                         │       │
                              │  read_only_compliance   │       │
                              │  safety_score           │       │
                              │  schema_compliance      │       │
                              └────────┬────────────────┘       │
                                  ┌────┴────┐                   │
                               safe?     unsafe?                │
                                  │         │                   │
                                  ▼         ▼                   │
                           execute_sql   reject_unsafe ──► END  │
                              ┌───┴───┐                         │
                           success  failure                     │
                              │         │                       │
                              │         ▼                       │
                              │    handle_error                 │
                              │    retry_count++                │
                              │      ┌───┴───┐                  │
                              │   < max?   >= max?              │
                              │      │         │                │
                              │      │    fail_after_retries    │
                              │      │         │                │
                              │      └─────────┼──► END         │
                              │                │                │
                              │      ┌─────────┘                │
                              │      └──────────────────────────┘
                              ▼
                        narrate_result
                              │
                        response populated
                              │
                              ▼
                       evaluate_quality
                     ┌─── SQLAS Scoring ────┐
                     │                      │
                     │  execution_accuracy   │
                     │  semantic_equivalence │
                     │  faithfulness         │
                     │  answer_relevance     │
                     │  safety_score         │
                     │  read_only_compliance │
                     │  overall_score        │
                     └──────────┬───────────┘
                                │
                               END
```

---

## Backend (Detail)

```
backend/
├── main.py                    FastAPI application
│   ├── POST /query            NL → LangGraph pipeline → response + SQLAS scores
│   ├── GET  /health           DB connection status + table list
│   ├── GET  /schema           Full auto-discovered schema context
│   ├── POST /evaluate         Run 25-case SQLAS evaluation suite
│   └── DELETE /conversations  Clear conversation history
│
├── agent/                     LangGraph agent package
│   ├── __init__.py            Exports: build_graph, run_query
│   ├── state.py               AgentState TypedDict (14 fields)
│   ├── nodes.py               7 node functions + SQLAS integration
│   │   ├── retrieve_schema    Inject cached DB context
│   │   ├── generate_sql       LLM generates SQL (or retry with error)
│   │   ├── validate_sql       SQLAS safety gate (3 checks)
│   │   ├── execute_sql        Read-only query via SQLAlchemy
│   │   ├── handle_error       Increment retry counter
│   │   ├── narrate_result     LLM summarizes results in NL
│   │   ├── evaluate_quality   SQLAS 20-metric scoring
│   │   ├── reject_unsafe      Terminal: safety rejection
│   │   └── fail_after_retries Terminal: max retries exceeded
│   └── graph.py               StateGraph definition + conditional edges
│       ├── route_after_validation   safe → execute | unsafe → reject
│       ├── route_after_execution    success → narrate | failure → retry
│       └── route_after_error        retry < max → generate | else → fail
│
├── config.py                  Pydantic Settings from .env
│   ├── AZURE_OPENAI_*         LLM configuration
│   ├── DATABASE_URL           Any SQLAlchemy async URL
│   ├── PII_COLUMNS            For SQLAS safety scoring
│   └── DOMAIN_HINT            Optional context for LLM
│
├── database.py                Database layer (any SQL DB)
│   ├── get_full_schema()      Introspect: tables, columns, PKs, FKs, indexes
│   ├── get_column_stats()     Per-column: min/max/avg/distinct/nulls/top values
│   ├── get_sample_rows()      Sample data for LLM context
│   ├── build_full_context()   Assemble complete RAG context string
│   └── execute_readonly_query() Strictly read-only with forbidden keyword guard
│
├── models.py                  Pydantic request/response models
│   ├── QueryRequest           { query, conversation_id }
│   ├── QueryResponse          { sql, data, response, sqlas_scores, metrics }
│   └── SQLASScoresResponse    { overall_score, execution_accuracy, ... }
│
├── eval_runner.py             SQLAS evaluation suite
│   ├── 25 TestCase definitions (easy/medium/hard/extra_hard)
│   └── run_evaluation()       Run agent + sqlas.evaluate() per case
│
├── ingest.py                  CSV → SQLite ingestion script
│   ├── health_demographics    2,000 rows, 14 columns, 6 indexes
│   └── physical_activity      multi-day activity logs, FK to demographics
│
└── requirements.txt           Dependencies
    ├── langgraph >= 0.2.0     Agent orchestration
    ├── sqlas >= 1.1.0         Evaluation framework (pip install sqlas)
    ├── fastapi                REST API
    ├── sqlalchemy + aiosqlite Async database
    └── openai                 Azure OpenAI LLM
```

---

## Frontend (Detail)

```
frontend/                      React + Vite + Tailwind CSS
├── src/
│   ├── App.jsx                Root — manages state, API calls
│   │   ├── sendQuery()        POST /query → update messages
│   │   ├── sendFeedback()     POST /feedback → thumbs up/down
│   │   └── clearChat()        DELETE /conversations
│   │
│   ├── components/
│   │   ├── Sidebar.jsx        Left panel
│   │   │   ├── DB status      Connection indicator, table list
│   │   │   ├── Schema explorer Collapsible full schema view
│   │   │   ├── Sample queries  8 clickable example questions
│   │   │   └── Clear button    Reset conversation
│   │   │
│   │   ├── ChatInterface.jsx  Main chat area
│   │   │   ├── Empty state     Welcome message + prompt
│   │   │   ├── Message list    Scrollable conversation
│   │   │   ├── Loading spinner "Generating SQL and analyzing..."
│   │   │   └── Input bar       Text input + send button
│   │   │
│   │   ├── MessageBubble.jsx  Individual message
│   │   │   ├── User bubble     Simple text with avatar
│   │   │   ├── Agent bubble    Response + status badges
│   │   │   ├── Metrics panel   Latency, row count, query type, SQL features
│   │   │   ├── Feedback        Thumbs up/down + comment box
│   │   │   └── SQLAS badges    Overall score, safety, accuracy
│   │   │
│   │   ├── CodeBlock.jsx      SQL display
│   │   │   ├── Collapsible     Toggle SQL visibility
│   │   │   ├── Syntax highlight Indigo-themed monospace
│   │   │   └── Copy button     One-click clipboard
│   │   │
│   │   └── DataTable.jsx      Query results
│   │       ├── Sortable table   Sticky headers, zebra rows
│   │       ├── Type-aware       Numeric right-aligned, text left
│   │       ├── Row counter      # column with index
│   │       ├── CSV export       One-click download
│   │       └── Truncation       "showing first 500" indicator
│   │
│   ├── index.css              Tailwind base + custom scrollbar
│   └── main.jsx               React entry point
│
├── index.html                 HTML shell
├── vite.config.js             Dev server + API proxy to :8000
├── tailwind.config.js         Dark theme configuration
└── package.json               Dependencies (react, lucide-react, react-markdown)
```

---

## SQLAS Integration Points

| Stage | SQLAS Metric | Purpose |
|-------|-------------|---------|
| **Pre-execution gate** | `read_only_compliance` | Blocks DDL/DML before execution |
| | `safety_score` | Detects PII exposure, SQL injection |
| | `schema_compliance` | Validates tables/columns exist |
| **Post-response scoring** | `evaluate()` | Full 20-metric production score |
| **Evaluation suite** | `run_suite()` | Batch evaluation with 25 test cases |
| **API response** | `SQLASScoresResponse` | Every query returns quality scores |

---

## Quick Start

### 1. Clone

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
  "sql": "SELECT COUNT(*) FROM health_demographics WHERE Blood_Pressure_Abnormality = 1",
  "data": {
    "columns": ["abnormal_blood_pressure_count"],
    "rows": [[987]],
    "row_count": 1,
    "execution_time_ms": 1.61
  },
  "response": "987 patients have abnormal blood pressure.",
  "success": true,
  "metrics": {
    "generation_latency_ms": 3947,
    "sql_execution_ms": 1.61,
    "narration_latency_ms": 3188,
    "total_latency_ms": 34535,
    "sqlas_overall": 0.9575
  },
  "sqlas_scores": {
    "overall_score": 0.9575,
    "execution_accuracy": 1.0,
    "semantic_equivalence": 0.8,
    "faithfulness": 1.0,
    "answer_relevance": 1.0,
    "safety_score": 1.0,
    "read_only_compliance": 1.0
  }
}
```

---

## Connecting Your Own Database

```env
# SQLite (default)
DATABASE_URL=sqlite+aiosqlite:///./health.db

# PostgreSQL
DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/dbname

# MySQL
DATABASE_URL=mysql+aiomysql://user:pass@host:3306/dbname
```

1. Set `DATABASE_URL` in `.env`
2. Restart the backend — schema is auto-discovered at startup
3. Optionally set `DOMAIN_HINT` for better LLM context
4. Optionally set `PII_COLUMNS` for SQLAS safety scoring

No code changes needed. Works with 100s of tables across any database.

---

## Evaluation

Run the built-in SQLAS evaluation suite (25 test cases, 4 difficulty tiers):

```bash
# Quick mode (5 test cases)
curl -X POST "http://localhost:8000/evaluate?quick=true"

# Full suite
curl -X POST "http://localhost:8000/evaluate?quick=false"

# Or from CLI
cd backend && python eval_runner.py --quick
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Agent orchestration | [LangGraph](https://github.com/langchain-ai/langgraph) `StateGraph` |
| Evaluation | [SQLAS](https://github.com/thepradip/SQLAS) — 20 metrics, 8 categories |
| LLM | Azure OpenAI (configurable) |
| Backend | FastAPI + SQLAlchemy (async) |
| Frontend | React 18 + Vite + Tailwind CSS |
| UI components | lucide-react + react-markdown |
| Database | Any SQLAlchemy-compatible DB |

---

## License

MIT License - [Pradip Tivhale](https://github.com/thepradip)

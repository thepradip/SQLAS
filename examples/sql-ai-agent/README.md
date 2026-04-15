# SQL AI Agent — RAG-based Natural Language to SQL

Full-stack application that maps natural language queries to SQL execution using a dynamic, schema-aware LLM agent with MLflow observability.

**Author:** [Pradip Tivhale](https://github.com/thepradip)

## Key Features

- **Zero-config schema discovery** — auto-introspects tables, columns, types, PKs, FKs, indexes, column statistics, and sample rows at startup. No hardcoded schema knowledge.
- **Scales to 100s of tables** — dynamic introspection works for any database size. Schema context is auto-generated regardless of table count.
- **Multiple database backends** — supports any SQLAlchemy-compatible database: SQLite, PostgreSQL, MySQL, SQL Server, Oracle. Just change the `DATABASE_URL`.
- **RAG-like context injection** — retrieved schema + column stats + sample rows are injected into the LLM prompt as rich context for accurate SQL generation.
- **Strictly read-only** — enforced at both SQL parsing and execution layers. No INSERT, UPDATE, DELETE, DROP, or DDL operations.
- **Self-healing SQL** — automatic retry with error feedback when generated SQL fails execution.
- **MLflow tracing** — full observability: every NL→SQL→Execute→Narrate step is traced.
- **SQLAS evaluation** — built-in evaluation endpoint using the SQLAS scoring framework.
- **React chat UI** — conversational interface with syntax-highlighted SQL, data tables, and feedback.

## Architecture

```
User Question
    ↓
[Schema Introspection] → Auto-discover all tables, columns, stats, relationships
    ↓
[Context Builder] → Build rich database context (schema + stats + samples)
    ↓
[LLM Agent] → Generate SQL from NL + context (Azure OpenAI)
    ↓
[Read-Only Executor] → Execute SQL with safety guards
    ↓
[Narrator] → LLM summarizes results in natural language
    ↓
Response (SQL + Data + Narrative)
```

## Prerequisites

- **Python**: 3.9+
- **Node.js**: v18+ (for frontend)
- **LLM Provider**: Azure OpenAI (configurable)

## Quick Start

### 1. Environment Configuration

```bash
cp .env.example .env
```

Edit `.env` with your settings:

```env
# Azure OpenAI
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o

# Database — any SQLAlchemy-compatible async URL
# SQLite (default):
DATABASE_URL=sqlite+aiosqlite:///./backend/health.db

# PostgreSQL:
# DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/dbname

# MySQL:
# DATABASE_URL=mysql+aiomysql://user:pass@host:3306/dbname
```

### 2. Backend

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Ingest sample health data (optional, for demo)
python ingest.py

# Start the API
uvicorn main:app --reload
```

FastAPI server: `http://localhost:8000`

### 3. Frontend

```bash
cd frontend
npm install
npm run dev
```

React app: `http://localhost:5173`

### 4. MLflow (optional)

```bash
cd backend
mlflow server --host 0.0.0.0 --port 5000
```

MLflow UI: `http://localhost:5000`

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check with DB info |
| GET | `/schema` | Full auto-discovered schema context |
| POST | `/query` | NL → SQL → Execute → Narrate |
| POST | `/feedback` | Thumbs up/down on query traces |
| POST | `/feedback/detailed` | Multi-dimension feedback |
| POST | `/evaluate` | Run SQLAS evaluation suite |
| DELETE | `/conversations/{id}` | Clear conversation history |

## Connecting Your Own Database

1. Set `DATABASE_URL` in `.env` to your database connection string
2. Restart the backend — schema is auto-discovered at startup
3. Optionally set `DOMAIN_HINT` for better LLM context (e.g., "This is a healthcare analytics database with patient records and activity logs.")

The agent will automatically introspect all tables, compute column statistics, and generate the full context. No code changes needed.

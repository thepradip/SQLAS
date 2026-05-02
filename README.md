<p align="center">
  <img src="assets/sqlas_logo.png" alt="AriaSQL Logo" width="200"/>
</p>

<h1 align="center">AriaSQL</h1>

<p align="center">
  <strong>Agentic SQL Agent — Natural Language to SQL with Multi-step Reasoning</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi&logoColor=white"/>
  <img src="https://img.shields.io/badge/React-Frontend-61DAFB?style=for-the-badge&logo=react&logoColor=black"/>
  <img src="https://img.shields.io/badge/SQLAS-v2.0-orange?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/License-MIT-blue?style=for-the-badge"/>
</p>

<p align="center">
  <a href="https://github.com/thepradip/AriaSQL">GitHub</a> ·
  <a href="https://github.com/thepradip/SQLAS">SQLAS Eval Framework</a>
</p>

---

AriaSQL transforms natural language into SQL using a **ReAct agentic loop** — the agent inspects your schema, reasons step-by-step, and runs multiple queries before answering. Works with any SQL database, any LLM, and scales to 100+ tables with zero configuration.

**Author:** [Pradip Tivhale](https://github.com/thepradip)

---

## Agentic Reasoning

- **ReAct loop** — Reason → call tool → observe result → repeat until confident
- **4 tools**: `list_tables`, `describe_table`, `execute_sql`, `final_answer`
- Auto-routing: complex queries use ReAct; simple queries use fast pipeline

---

## Intelligent Schema Retrieval (100+ tables)

BM25 + dense embedding hybrid search with RRF. FK-graph-aware. Token budget control.

---

## Semantic Query Cache

L1 exact → L2 semantic → L4 result TTL. Learning loop from user feedback.

---

## Any LLM, Any Database

Azure OpenAI · OpenAI · Anthropic · Ollama · any compatible endpoint.
SQLite · PostgreSQL · MySQL · any SQLAlchemy async URL.

---

## Production Safety

AST-based read-only (sqlglot) · query timeout · fetchmany OOM protection · persistent conversations.

---

## Quick Start

### 1. Clone

```bash
git clone https://github.com/thepradip/AriaSQL.git
cd AriaSQL
```

### 2. Backend

```bash
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env — minimum: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT_NAME

python ingest.py      # load sample health data
uvicorn main:app --reload
```

API: `http://localhost:8000` · Docs: `http://localhost:8000/docs`

### 3. Frontend

```bash
cd frontend && npm install && npm run dev
```

UI: `http://localhost:5173`

---

## Configuration

```env
# Required
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-key
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o
AZURE_OPENAI_API_VERSION=2024-12-01-preview
DATABASE_URL=sqlite+aiosqlite:///./health.db

# Alternative LLM providers
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
OLLAMA_BASE_URL=http://localhost:11434

# Agentic + cache
AGENTIC_MODE=true
CACHE_ENABLED=true
SEMANTIC_CACHE_THRESHOLD=0.92
RESULT_CACHE_TTL=300

# Large schema (100+ tables)
LARGE_SCHEMA_THRESHOLD=20
MAX_CONTEXT_TABLES=8
SCHEMA_TOKEN_BUDGET=12000
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=   # upgrade to hybrid BM25+embedding
```

---

## License

MIT — [Pradip Tivhale](https://github.com/thepradip)

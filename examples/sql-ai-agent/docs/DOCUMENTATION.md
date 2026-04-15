# SQL AI Agent — Production-Grade GenAI Solution for Health Data Analysis

**Author:** Pradip Tivhale
**Version:** 2.0.0
**Date:** March 2026
**Stack:** Python 3.11, FastAPI, React 18, Azure OpenAI (GPT-5.2), MLflow 3.10.1, SQLite/PostgreSQL

---

## Table of Contents

1. [Overview](#1-overview)
2. [Technical Architecture](#2-technical-architecture)
3. [Project Structure](#3-project-structure)
4. [Setup Instructions](#4-setup-instructions)
5. [Configuration](#5-configuration)
6. [Backend — SQL AI Agent](#6-backend--sql-ai-agent)
7. [Frontend — React Chat UI](#7-frontend--react-chat-ui)
8. [MLflow Observability](#8-mlflow-observability)
9. [SQLAS Evaluation Framework](#9-sqlas-evaluation-framework)
10. [API Reference](#10-api-reference)
11. [Design Decisions](#11-design-decisions)
12. [Security and Ethics](#12-security-and-ethics)
13. [Scaling to Production](#13-scaling-to-production)

---

## 1. Overview

A production-grade GenAI solution that translates natural language questions into SQL queries, executes them against any database, and returns concise natural language responses. Built with full MLflow observability, user feedback collection, and a RAGAS-equivalent evaluation framework (SQLAS).

### Key Capabilities

- **Natural Language to SQL** — Users ask questions in plain English; the agent generates optimized SQL
- **Database-Agnostic** — Dynamically introspects any SQL database (SQLite, PostgreSQL, MySQL)
- **Strictly Read-Only** — Three layers of protection ensure no data modification
- **On-the-Fly Joins** — Never pre-merges tables; joins only when queries require cross-table data
- **Auto-Retry** — On SQL errors, feeds error context back to LLM for self-correction
- **Full Observability** — MLflow traces every step: SQL generation, execution, narration, feedback
- **User Feedback** — Thumbs up/down from the UI, logged to MLflow for quality tracking
- **Production Evaluation** — SQLAS framework: 15 metrics across 6 categories, 25 test queries

---

## 2. Technical Architecture

```
  React Frontend (Vite + Tailwind)
  [Chat UI] [SQL Viewer] [Data Table] [Feedback] [Metrics Panel]
                        |
                    REST API (JSON)
                        |
  FastAPI Backend (Python 3.11+)
  +----------------------------------------------------------+
  | MLflow Traced Pipeline (AGENT span)                      |
  |                                                          |
  | [1. Generate SQL] --> [2. Execute SQL] --> [3. Narrate]  |
  |    (LLM Span)    |      (TOOL Span)  |    (LLM Span)   |
  |                   |          |        |                  |
  |                   |   [Auto-Retry]    |                  |
  |                   |   (on error)      |                  |
  +----------------------------------------------------------+
  |                                                          |
  | [Schema Introspection] [Metrics Engine] [Feedback Logger]|
  +----------------------------------------------------------+
            |                               |
  Any SQL Database                   Azure OpenAI
  (SQLite/PostgreSQL/MySQL)         (GPT-5.2-chat)
            |
  MLflow Server (Traces, Feedback, Metrics, UI)
```

---

## 3. Project Structure

```
Infogain/
+-- .env                          # Azure OpenAI credentials
+-- .env.example                  # Template
+-- DOCUMENTATION.md              # This document
|
+-- backend/
|   +-- main.py                   # FastAPI app: routes, lifespan, CORS
|   +-- config.py                 # Pydantic settings from env vars
|   +-- database.py               # Dynamic schema introspection, read-only execution
|   +-- agent.py                  # SQL AI agent (generate -> execute -> narrate)
|   +-- tracing.py                # MLflow tracing, metrics computation, feedback
|   +-- models.py                 # Pydantic request/response models
|   +-- eval_framework.py         # SQLAS: 15 metrics, 6 categories
|   +-- eval_runner.py            # 25-query test suite + runner
|   +-- ingest.py                 # CSV -> SQLite ingestion
|   +-- requirements.txt          # Python dependencies
|   +-- health.db                 # SQLite database
|
+-- frontend/
|   +-- package.json
|   +-- vite.config.js
|   +-- src/
|       +-- App.jsx               # State management, API calls, feedback
|       +-- components/
|           +-- ChatInterface.jsx # Chat input + message list
|           +-- MessageBubble.jsx # Messages + feedback buttons + metrics
|           +-- CodeBlock.jsx     # Expandable SQL viewer with copy
|           +-- DataTable.jsx     # Expandable result table
|           +-- Sidebar.jsx       # DB status, schema explorer, samples
|
+-- data/
    +-- health_dataset_1.csv      # 2,000 patients x 14 variables
    +-- health_dataset_2.csv      # 20,000 activity records (10 days/patient)
```

---

## 4. Setup Instructions

### Prerequisites

- Python 3.11+
- Node.js 18+
- Azure OpenAI access

### Step 1: Configure

```bash
cd Infogain
cp .env.example .env
# Edit .env with your Azure OpenAI credentials
```

### Step 2: Backend

```bash
cd backend
pip install -r requirements.txt
python ingest.py                  # Load data into SQLite
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Step 3: Frontend

```bash
cd frontend
npm install
npm run dev                       # http://localhost:5173
```

### Step 4: MLflow UI

```bash
cd backend
mlflow server --host 0.0.0.0 --port 5000
# Open http://localhost:5000 -> Experiment "sql-ai-agent" -> Traces tab
```

### Step 5: Run Evaluation

```bash
cd backend
python eval_runner.py --quick     # 5 test cases
python eval_runner.py             # Full 25 test cases
```

### Running Services

| Service | URL | Purpose |
|---------|-----|---------|
| React Frontend | http://localhost:5173 | User-facing chat interface |
| FastAPI Backend | http://localhost:8000 | REST API + SQL agent |
| MLflow UI | http://localhost:5000 | Traces, metrics, feedback viewer |

---

## 5. Configuration

All configuration via environment variables (loaded from `.env`):

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| AZURE_OPENAI_ENDPOINT | Yes | - | Azure OpenAI resource URL |
| AZURE_OPENAI_API_KEY | Yes | - | API key |
| AZURE_OPENAI_DEPLOYMENT_NAME | No | gpt-5.2-chat | Model deployment name |
| DATABASE_URL | No | sqlite+aiosqlite:///./health.db | Any SQLAlchemy async URL |
| MAX_RESULT_ROWS | No | 500 | Max rows returned per query |
| DOMAIN_HINT | No | "" | Optional domain context for LLM |

### Switching Databases

```bash
# PostgreSQL
DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/mydb

# MySQL
DATABASE_URL=mysql+aiomysql://user:pass@host:3306/mydb
```

No code changes needed — the agent auto-discovers the schema.

---

## 6. Backend — SQL AI Agent

### Agent Pipeline (agent.py)

Three-step pipeline, each traced as an MLflow span:

1. **SQL Generation** — Sends user query + rich schema context (tables, columns, stats, samples, indexes, FKs) to Azure OpenAI. LLM generates optimized SQL.

2. **SQL Execution** — Executes SQL with strict read-only enforcement. On failure, triggers auto-retry.

3. **Result Narration** — Sends raw results back to LLM for a concise, direct natural language answer. No unnecessary tables, no disclaimers — just the answer.

### Dynamic Schema Introspection (database.py)

On startup, auto-discovers:
- All tables, columns, types, nullability
- Primary keys, foreign keys, indexes
- Column statistics (min/max/avg for numeric; distinct count + top values for categorical)
- Sample rows (3 per table)
- Inter-table relationships from foreign keys

### Read-Only Enforcement (3 layers)

1. **LLM Prompt** — System prompt forbids DDL/DML
2. **SQL Validation** — Pre-execution check rejects non-SELECT statements
3. **Keyword Blocking** — Blocks INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, TRUNCATE, GRANT, REVOKE

---

## 7. Frontend — React Chat UI

Built with Vite + React 18 + Tailwind CSS.

### Components

| Component | Purpose |
|-----------|---------|
| ChatInterface.jsx | Chat input + message list |
| MessageBubble.jsx | Message with feedback buttons + expandable metrics panel |
| CodeBlock.jsx | Expandable SQL viewer with copy button |
| DataTable.jsx | Expandable result table with scroll |
| Sidebar.jsx | DB status, schema explorer, sample queries |

### Feedback UI

Each assistant message shows:
- **Thumbs up/down** buttons — sends to /feedback API -> MLflow
- **Comment box** (on thumbs down) — optional text feedback
- **Metrics panel** (expandable) — latency, SQL complexity, query type
- **Trace ID** — for linking to MLflow

---

## 8. MLflow Observability

### Tracing Architecture (tracing.py)

Every query creates an MLflow trace:

```
sql_agent_pipeline (AGENT)
+-- generate_sql (LLM)
+-- execute_sql (TOOL)
|   +-- retry_generate_sql (LLM)     [only on error]
|   +-- retry_execute_sql (TOOL)     [only on error]
+-- narrate_result (LLM)
```

`mlflow.openai.autolog()` auto-captures all Azure OpenAI calls (tokens, prompts, completions).

### What Gets Traced

| Data Point | Where |
|-----------|-------|
| User query | Root span input |
| Generated SQL | generate_sql span output |
| SQL execution result | execute_sql span output |
| Narrated response | narrate_result span output |
| All latencies | Span attributes |
| SQL complexity metrics | Root span attributes |
| Token usage | OpenAI autolog spans |
| User feedback | mlflow.log_feedback() |

### Pipeline Metrics (returned per query)

| Metric | Type | Description |
|--------|------|-------------|
| total_latency_ms | float | End-to-end pipeline time |
| generation_latency_ms | float | LLM SQL generation time |
| sql_execution_ms | float | Database query time |
| narration_latency_ms | float | LLM narration time |
| retry_count | int | Retry attempts |
| query_type | string | simple_select, filter, aggregation, join, join_aggregation |
| sql_length | int | SQL character count |
| join_count | int | Number of JOINs |
| has_aggregation | bool | Uses COUNT/SUM/AVG |

### Feedback APIs

| Type | Endpoint | MLflow Storage |
|------|----------|---------------|
| Thumbs up/down | POST /feedback | mlflow.log_feedback() + trace tag |
| Detailed rating | POST /feedback/detailed | Multi-dimension assessment |
| Comment | POST /feedback | Trace tag |

---

## 9. SQLAS Evaluation Framework

SQLAS (SQL Agent Scoring) is a **RAGAS-equivalent evaluation framework** purpose-built for SQL AI agents.

### RAGAS to SQLAS Mapping

| RAGAS Concept | SQLAS Equivalent | Description |
|--------------|-----------------|-------------|
| Faithfulness | Faithfulness | Claims in narration grounded in SQL results |
| Answer Relevance | Answer Relevance | Response answers the question |
| Context Precision | Schema Compliance | SQL uses valid tables/columns |
| Context Recall | Answer Completeness | All key data surfaced |
| - | Execution Accuracy | SQL returns correct results |
| - | Semantic Equivalence | LLM judges if SQL answers the intent |
| - | Safety Score | PII, injection, DDL detection |

### Production Composite Score (15 metrics, 6 categories)

```
SQLAS Score =
  40% Execution Accuracy       -- does SQL return correct results?
+ 15% Semantic Correctness     -- does SQL answer the user's intent? (LLM judge)
+ 15% Cost Efficiency          -- efficient query? (VES + scan + quality + schema)
+ 10% Latency                  -- fast? no errors? appropriate complexity?
+ 10% Task Success             -- user gets correct insight? (faithfulness + relevance)
+ 10% Safety                   -- read-only + PII + injection protection
```

### Detailed Weight Breakdown

| Category (Weight) | Metric | Weight | Method |
|-------------------|--------|--------|--------|
| **Execution Accuracy (40%)** | execution_accuracy | 40% | Automated: output match + structure + efficiency |
| **Semantic Correctness (15%)** | semantic_equivalence | 15% | LLM-as-judge |
| **Cost Efficiency (15%)** | efficiency_score | 5% | Automated: VES (BIRD benchmark) |
| | data_scan_efficiency | 5% | Automated: full scan detection |
| | sql_quality | 3% | LLM-as-judge: join/agg/filter correctness |
| | schema_compliance | 2% | Automated: valid tables/columns (sqlglot) |
| **Latency (10%)** | execution_success | 5% | Automated: ran without error |
| | query_complexity_appropriate | 3% | LLM-as-judge: not over/under-engineered |
| | empty_result_penalty | 2% | Automated: returned data when expected |
| **Task Success (10%)** | faithfulness | 4% | LLM-as-judge: claims grounded in data |
| | answer_relevance | 3% | LLM-as-judge: answers the question |
| | answer_completeness | 2% | LLM-as-judge: all key data surfaced |
| | fluency | 1% | LLM-as-judge: readability |
| **Safety (10%)** | read_only_compliance | 5% | Automated: no DDL/DML |
| | safety_score | 5% | Automated: PII + injection + restricted access |

### Execution Accuracy Formula

Unlike simple exact-match (Spider benchmark), our execution accuracy uses semantic comparison:

```
Execution Accuracy = 60% x Output Match + 20% x Structure Match + 20% x Efficiency

Output Match:   Row-by-row numeric comparison between predicted and gold SQL results.
                Ignores label differences (0 vs 'Male'), tolerates ROUND differences.
                Handles extra columns from CASE WHEN labels.

Structure Match: Same row count? Both return data?

Efficiency:     Predicted query speed vs gold query speed.
```

### Test Suite

25 test cases across 4 difficulty tiers:

| Tier | Count | Examples |
|------|-------|---------|
| Easy (7) | Simple filter/count | "How many patients have abnormal BP?" |
| Medium (8) | GROUP BY, percentages | "Average BMI smokers vs non-smokers?" |
| Hard (7) | Cross-table JOINs | "Average steps by CKD status?" |
| Extra Hard (3) | Correlation, bucketing | "Pearson correlation hemoglobin vs activity?" |

### Running Evaluations

```bash
python eval_runner.py --quick     # 5 tests (~3 min)
python eval_runner.py             # Full 25 tests (~15 min)
curl -X POST localhost:8000/evaluate?quick=true  # Via API
```

---

## 10. API Reference

### GET /health

```json
{"status": "ok", "database": "health.db", "tables": ["health_demographics", "physical_activity"], "mlflow_experiment": "sql-ai-agent"}
```

### GET /schema

Returns auto-discovered database schema with statistics.

### POST /query

```json
// Request
{"query": "How many patients have abnormal blood pressure?", "conversation_id": "uuid"}

// Response
{
  "sql": "SELECT COUNT(*) FROM health_demographics WHERE Blood_Pressure_Abnormality = 1",
  "data": {"columns": ["COUNT(*)"], "rows": [[987]], "row_count": 1, "truncated": false, "execution_time_ms": 1.44},
  "response": "987 patients have abnormal blood pressure.",
  "success": true,
  "trace_id": "tr-abc123...",
  "metrics": {"total_latency_ms": 6453, "generation_latency_ms": 3402, "sql_execution_ms": 1.44, "query_type": "filter", "...": "..."}
}
```

### POST /feedback

```json
{"trace_id": "tr-abc123...", "value": true, "comment": "Accurate", "user_id": "anonymous"}
```

### POST /feedback/detailed

```json
{"trace_id": "tr-abc123...", "accuracy": 5, "relevance": 4, "sql_quality": 5, "comment": "Good"}
```

### POST /evaluate?quick=true

Runs SQLAS evaluation suite. Returns summary + per-query details.

---

## 11. Design Decisions

### SQL vs Pandas

| Factor | Pandas (POC) | SQL (Production) |
|--------|-------------|-----------------|
| Scale | RAM-limited (millions) | Billions of rows |
| Performance | Full table scan in memory | Indexed queries, optimizer |
| Safety | exec() with restricted builtins | Read-only SQL validation |
| Portability | Python-only | Any SQL database |

### Dynamic Schema Introspection

No hardcoded table/column knowledge. Works with any database. Statistics-enriched prompts help LLM write accurate SQL.

### On-the-Fly Joins

Datasets remain separate. LLM generates JOINs only when needed. Aggregates N-side first to avoid row explosion.

### Clean Narration

Direct answers only. No markdown tables for single values. No disclaimers or health advice. The SQL, raw data, and metrics are available in expandable panels.

---

## 12. Security and Ethics

### Data Security

- Read-only enforcement (3 layers)
- SQL output limited to 500 rows
- Credentials in .env, never committed
- Safety scoring: PII detection, injection pattern detection

### Health Data Ethics

- Patient numbers are synthetic
- No data leaves local environment unless configured
- Feedback transparency

---

## 13. Scaling to Production

| Component | Dev | Production |
|-----------|-----|-----------|
| Database | SQLite | PostgreSQL with read replicas |
| Backend | Single uvicorn | Gunicorn + workers |
| Sessions | In-memory | Redis |
| MLflow | Local SQLite | Remote PostgreSQL + S3 artifacts |
| Frontend | Vite dev server | npm build + Nginx/CDN |
| Auth | None | OAuth2 / API key |

### Switch to PostgreSQL

```bash
DATABASE_URL=postgresql+asyncpg://user:password@host:5432/health_analytics
pip install asyncpg
```

No code changes — auto-discovers the new schema.

---

**Author:** Pradip Tivhale
**Built with:** Python, FastAPI, React, Azure OpenAI, MLflow 3.10.1, sqlglot

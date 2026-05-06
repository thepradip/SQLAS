<p align="center">
  <img src="assets/sqlas_logo.png" alt="AriaSQL Logo" width="200"/>
</p>

<h1 align="center">AriaSQL</h1>

<p align="center">
  <strong>Production-Grade Agentic SQL Agent — Natural Language to SQL with Multi-step Reasoning</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi&logoColor=white"/>
  <img src="https://img.shields.io/badge/React-Frontend-61DAFB?style=for-the-badge&logo=react&logoColor=black"/>
  <img src="https://img.shields.io/badge/SQLAS-v2.7.0-orange?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/License-MIT-blue?style=for-the-badge"/>
</p>

<p align="center">
  <a href="https://github.com/thepradip/AriaSQL">GitHub</a> ·
  <a href="https://github.com/thepradip/SQLAS">SQLAS Eval Framework</a>
</p>

---

AriaSQL transforms natural language into SQL using a **ReAct agentic loop** — the agent inspects your schema, reasons step-by-step, and executes queries before answering. Works with any SQL database, any LLM, and scales to 100+ tables with zero configuration.

**Author:** [Pradip Tivhale](https://github.com/thepradip)

---

## Quick Start

```bash
git clone https://github.com/thepradip/AriaSQL.git && cd AriaSQL
cd backend && pip install -r requirements.txt && cp .env.example .env
python ingest.py && uvicorn main:app --reload
# frontend: cd frontend && npm install && npm run dev
```

Or with Docker:
```bash
docker-compose up
```

---

## What's New — Latest Release

### Pre-Execution SQL Validator
AriaSQL now validates and auto-corrects SQL **before** it reaches the database:

| Issue | Action |
|-------|--------|
| LIMIT on non-top-N questions | Auto-removed — was silently truncating full results |
| TRIM() on numeric columns | Auto-removed — invalid on Postgres/BigQuery |
| Single `REPLACE('$','')` for currency | Warning — commas left in break CAST |
| MAX() on total/amount columns | Warning — SUM() likely intended |
| JOIN without aggregation on 1:N | Warning — row explosion risk |

All validation events are logged to MLflow spans (`validation.codes`, `validation.auto_fixed`).

### Eval Framework Upgrades
- **Row count match** — runs `COUNT(*)` on both generated and gold SQL; catches LIMIT truncation that previously scored as PASS
- **Table identity check** — parses FROM/JOIN and flags wrong table names against `expected_tables`
- **Scalar path** — single-value results (correlation, count, average) use tight relative tolerance instead of loose `tol=1.0`

### Prompt Rule Fixes (from real tester data)
- Currency: always `REPLACE(REPLACE(col,'$',''),',','')` — single REPLACE misses commas
- Aggregation: `SUM()` for totals, never `MAX()`; `SUM()` not `AVG()` for event counts
- LIMIT: only added when question explicitly asks for top-N
- NULL: `WHERE col IS NOT NULL` before aggregating nullable columns
- Zero results: explain why based on data, not "no results found"

### Regression Test Suite
**91 tests across 16 groups** — each mapped to a real tester failure observed in evaluation data:
Result completeness, table identity, currency cleaning, aggregation type, scalar precision, schema hallucination, full table scan, TRIM on numeric, NULL handling, row duplication, unsafe intent, invalid cross-dataset join, empty results, flag ambiguity, SQL validator, failure classification.

---

## Core Features

### Agentic Reasoning
- **ReAct loop** — Reason → call tool → observe result → repeat until confident
- **4 tools**: `list_tables`, `describe_table`, `execute_sql`, `final_answer`
- Auto-routing: complex queries use ReAct; simple queries use fast pipeline
- Mandatory planning: `create_plan` before `execute_sql` prevents first-attempt failures

### Intelligent Schema Retrieval (100+ tables)
- BM25 + dense embedding hybrid search with Reciprocal Rank Fusion (RRF)
- FK-graph-aware context expansion — auto-includes FK partners for complex queries
- Pre-generation FK completeness validation — warns before SQL is sent to LLM
- Token budget control — never overflows context window

### Semantic Query Cache
- L1 exact → L2 semantic → L4 result TTL
- Learning loop from user feedback — cache improves over time
- Few-shot example retrieval from similar past queries

### SQL Generation Rules
17 enforced prompt rules covering:
- Aggregation defaults (SUM vs MAX/AVG)
- Currency cleaning (double REPLACE)
- NULL handling before aggregation
- LIMIT only for top-N queries
- TRIM restricted to text columns
- Zero-result explanation required
- Row explosion prevention (aggregate N-side first)

### Any LLM, Any Database
- **LLMs**: Azure OpenAI · OpenAI · Anthropic · Ollama · any compatible endpoint
- **Databases**: SQLite · PostgreSQL · MySQL · any SQLAlchemy async URL

### Production Safety
- AST-based read-only enforcement (sqlglot) — cannot bypass via CTE injection
- Query timeout + fetchmany OOM protection
- Persistent multi-turn conversations
- Drift monitor — detects schema changes

---

## Evaluation — SQLAS v2.7.0

```bash
python backend/eval_runner.py --quick                                    # 5 test cases
python backend/eval_runner.py                                            # full suite
python backend/eval_runner.py --provider anthropic:claude-opus-4-7      # specific LLM
python backend/eval_runner.py --compare azure,anthropic --quick         # A/B compare
```

**50+ metrics across 9 dimensions:**

| Dimension | Key Metrics | Weight |
|-----------|-------------|--------|
| Execution Accuracy | Output match, row count match, table identity, scalar precision | 25% |
| Semantic Correctness | LLM judge, exact match, multi-gold SQL | 10% |
| Agentic Quality | Planning, schema grounding, steps efficiency | 10% |
| Context Quality | RAGAS-mapped precision, recall, entity recall | 8% |
| Cost Efficiency | VES, data scan efficiency, SQL quality | 10% |
| Task Success | Faithfulness, relevance, completeness, fluency | 8% |
| Result + Visualization | Result coverage, chart validity, data alignment | 7% |
| Guardrails | Read-only, SQL injection, prompt injection, PII | 15% |
| Execution Quality | Success rate, complexity match, empty result | 7% |

### Failure Classification
Every evaluation produces a named failure category:

```python
from sqlas import classify_failure

analysis = classify_failure(sql, scores, details)
print(analysis.primary)    # FailureCategory.LIMIT_TRUNCATION
print(analysis.top_hint()) # "Remove LIMIT — question asks for full results"
```

Categories: `LIMIT_TRUNCATION` · `WRONG_TABLE` · `WRONG_AGGREGATION` · `SCALAR_MISMATCH` · `ROW_EXPLOSION` · `SCHEMA_HALLUCINATION` · `FULL_TABLE_SCAN` · `TRIM_ON_NUMERIC` · `UNSAFE_QUERY` · `CURRENCY_NOT_CLEANED` · `NULL_IN_AGGREGATION` · and more.

### Hardness Classification
```python
from sqlas import auto_classify_hardness
auto_classify_hardness(sql)  # → "easy" | "medium" | "hard" | "extra-hard"
```
Auto-set on every `evaluate()` call. Follows BIRD benchmark criteria.

### Report Generation
```python
from sqlas import generate_report
report = generate_report(scores_list, format="markdown")  # or "json"
scores.to_json()               # CI artifact storage
scores.to_markdown_report()    # PR comment
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | DB status, table count, MLflow experiment |
| GET | `/schema` | Full auto-discovered schema context |
| POST | `/query` | NL → SQL → Execute → Narrate |
| POST | `/feedback` | Thumbs up/down — teaches the few-shot cache |
| POST | `/feedback/detailed` | Multi-dimension rating |
| POST | `/export/csv` | Download results as CSV |
| GET | `/cache/stats` | Hit rates, tokens saved, cost savings |
| DELETE | `/cache/results` | Flush result cache after data updates |
| POST | `/evaluate` | Run SQLAS evaluation suite |
| DELETE | `/conversations/{id}` | Clear conversation history |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Agent | Custom ReAct loop — tool calling, MLflow tracing |
| Pre-exec validation | `sql_validator.py` — AST + regex, auto-fix before execution |
| Schema retrieval | BM25 + dense embeddings + Reciprocal Rank Fusion |
| Semantic cache | SQLite-backed L1/L2/L4, verified few-shot learning |
| Evaluation | [SQLAS v2.7.0](https://github.com/thepradip/SQLAS) — 50+ metrics, failure classification |
| LLM | Azure OpenAI · OpenAI · Anthropic · Ollama · any compatible |
| Backend | FastAPI + SQLAlchemy async |
| Frontend | React 18 + Vite + Tailwind CSS |
| Observability | MLflow — traces, spans, feedback, validation events |
| Database | SQLite · PostgreSQL · MySQL (any SQLAlchemy async URL) |

---

## License

MIT — [Pradip Tivhale](https://github.com/thepradip)

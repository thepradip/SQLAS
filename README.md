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

```bash
git clone https://github.com/thepradip/AriaSQL.git && cd AriaSQL
cd backend && pip install -r requirements.txt && cp .env.example .env
python ingest.py && uvicorn main:app --reload
# frontend: cd frontend && npm install && npm run dev
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

## Evaluation — SQLAS v2.0

```bash
python backend/eval_runner.py --quick                                    # 5 test cases
python backend/eval_runner.py                                            # 28 test cases
python backend/eval_runner.py --provider anthropic:claude-opus-4-7      # test a specific LLM
python backend/eval_runner.py --compare azure,anthropic:claude-opus-4-7,ollama:sqlcoder --quick
```

45 metrics across 9 dimensions — correctness, agentic quality, cache ROI, safety, faithfulness:

| Dimension | Weight |
|-----------|--------|
| Execution Accuracy | 25% |
| Semantic Correctness | 10% |
| **Agentic Quality** (planning, grounding, steps) | **10%** |
| Context Quality (RAGAS-mapped) | 8% |
| Cost Efficiency (VES, scan, SQL quality) | 10% |
| Task Success (faithfulness, relevance) | 8% |
| Result + Visualization | 7% |
| Guardrails (read-only, injection, PII) | 15% |
| Execution Quality | 7% |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Agent | Custom ReAct loop — tool calling, MLflow tracing |
| Schema retrieval | BM25 + dense embeddings + Reciprocal Rank Fusion |
| Semantic cache | SQLite-backed L1/L2/L4, verified few-shot learning |
| Evaluation | [SQLAS v2.0](https://pypi.org/project/sqlas/) — 45 metrics, 9 categories |
| LLM | Azure OpenAI · OpenAI · Anthropic · Ollama · any compatible |
| Backend | FastAPI + SQLAlchemy async |
| Frontend | React 18 + Vite + Tailwind CSS |
| Observability | MLflow — traces, spans, feedback |
| Database | SQLite · PostgreSQL · MySQL (any SQLAlchemy async URL) |

---

## License

MIT — [Pradip Tivhale](https://github.com/thepradip)

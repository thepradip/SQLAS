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
- Agent inspects schema before querying — no hallucinated column names
- Handles multi-step questions that require multiple SQL queries
- Auto-routing: complex queries use ReAct; simple queries use fast pipeline
- Every reasoning step visible in the UI and traced in MLflow

---

## Intelligent Schema Retrieval (100+ tables)

- **BM25 + dense embedding hybrid search** with Reciprocal Rank Fusion (RRF)
- **FK-graph-aware** table selection — JOIN-required tables always included
- **Token budget control** — injects only 8-12 relevant tables per query (not all 100+)
- Schema stats cached to disk — instant restart after first run

---

## Semantic Query Cache

Three cache levels — after warmup, 60-80% of queries never reach the LLM:

| Level | Mechanism | Latency | Tokens saved |
|---|---|---|---|
| L1 | Exact hash match | <1ms | ~9,500 (~$0.047) |
| L2 | Semantic cosine ≥ 0.92 | ~5ms | ~8,600 (~$0.043) |
| L4 | SQL result TTL (5 min) | <1ms | DB execution time |

**Learning loop** — user thumbs-up marks a query as *verified*. Verified queries are injected as few-shot examples into future similar queries, improving SQL accuracy over time without retraining.

```
GET /cache/stats  →  hit_rate, tokens_saved, cost_saved_usd, top_queries
DELETE /cache/results  →  flush result cache after data updates
```

---

## License

MIT — [Pradip Tivhale](https://github.com/thepradip)

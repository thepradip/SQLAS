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
- Auto-routing: complex queries use ReAct; simple queries use fast pipeline

---

## Intelligent Schema Retrieval (100+ tables)

- **BM25 + dense embedding hybrid search** with Reciprocal Rank Fusion
- **FK-graph-aware** table selection, token budget control, disk-cached stats

---

## Semantic Query Cache

L1 exact → L2 semantic → L4 result TTL. 60-80% cache hit rate after warmup.
Learning loop: thumbs-up → verified few-shot examples improve future queries.

---

## Any LLM, Any Database

**LLM providers** — swap without changing any business logic:

```env
# Use one of:
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o          # Azure OpenAI (default)
OPENAI_API_KEY=sk-...                         # OpenAI direct
ANTHROPIC_API_KEY=sk-ant-...                  # Anthropic Claude
OLLAMA_BASE_URL=http://localhost:11434        # Local models (Llama, SQLCoder)
```

**Databases** — any SQLAlchemy-compatible URL:

```env
DATABASE_URL=sqlite+aiosqlite:///./my.db
DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/dbname
DATABASE_URL=mysql+aiomysql://user:pass@host:3306/dbname
```

---

## Production Safety

- **AST-based read-only** via sqlglot — blocks write ops inside CTEs, not just keywords
- **Query timeout** — configurable (default 30s), enforced via `asyncio.wait_for`
- **Memory-safe** — `fetchmany()` bounds result sets, no OOM on large tables
- **Persistent conversations** — SQLite-backed, survive server restarts
- **SQL injection detection**, **PII access scoring**, **prompt injection** checks

---

## License

MIT — [Pradip Tivhale](https://github.com/thepradip)

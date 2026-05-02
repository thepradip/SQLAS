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

```
User Query
    │
    ├─ Simple (count, avg, filter) ──── Fast pipeline  → <2s
    │
    └─ Complex (correlate, compare, analyze)
           │
           ▼  ReAct Agent
    list_tables → describe_table → execute_sql → (repeat) → final_answer
```

---

## License

MIT — [Pradip Tivhale](https://github.com/thepradip)

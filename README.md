# SQLAS — SQL Agent Scoring Framework

**A RAGAS-equivalent evaluation library for Text-to-SQL and Agentic SQL agents.**

[![PyPI](https://img.shields.io/pypi/v/sqlas)](https://pypi.org/project/sqlas/)
[![Python](https://img.shields.io/pypi/pyversions/sqlas)](https://pypi.org/project/sqlas/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Evaluate SQL agents across production metrics — correctness, response quality, guardrails, visualization, **agentic reasoning quality**, and **cache performance** — aligned with Spider, BIRD, RAGAS, and MLflow standards.

---

## Install

```bash
pip install sqlas                # core
pip install "sqlas[mlflow]"      # + MLflow integration
```

---

## Quick Start

```python
from sqlas import evaluate

def llm_judge(prompt: str) -> str:
    return openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
    ).choices[0].message.content

scores = evaluate(
    question      = "How many active users are there?",
    generated_sql = "SELECT COUNT(*) FROM users WHERE active = 1",
    gold_sql      = "SELECT COUNT(*) FROM users WHERE active = 1",
    db_path       = "my_database.db",
    llm_judge     = llm_judge,
    response      = "There are 1,523 active users.",
    result_data   = {"columns": ["COUNT(*)"], "rows": [[1523]],
                     "row_count": 1, "execution_time_ms": 2.1},
)

print(scores.overall_score)   # 0.95
print(scores.summary())
```

---

## Run a Test Suite

```python
from sqlas import run_suite, TestCase

test_cases = [
    TestCase(
        question  = "How many users signed up this month?",
        gold_sql  = "SELECT COUNT(*) FROM users WHERE created_at >= '2026-03-01'",
        category  = "easy",
    ),
    TestCase(
        question  = "Average order value by country",
        gold_sql  = "SELECT country, AVG(total) FROM orders GROUP BY country",
        category  = "medium",
    ),
]

def my_agent(question: str) -> dict:
    sql      = generate_sql(question)
    result   = execute(sql)
    response = narrate(result)
    return {"sql": sql, "response": response, "data": result}

results = run_suite(
    test_cases  = test_cases,
    agent_fn    = my_agent,
    llm_judge   = llm_judge,
    db_path     = "my_database.db",
    pass_threshold = 0.6,
)

print(results["summary"]["overall_score"])    # 0.88
print(results["summary"]["by_category"])      # {"easy": 0.92, "medium": 0.84}
```

---

## Agentic SQL Agents (v2.0.0)

For ReAct-style agents that use tool calling, pass `agent_steps` to unlock agentic quality scoring.

```python
from sqlas import evaluate, WEIGHTS_V4

# agent_steps = list of tool calls from your ReAct loop
# [{tool: "describe_table", args: {...}}, {tool: "execute_sql", ...}, ...]

scores = evaluate(
    question      = "Compare activity levels of smokers vs non-smokers",
    generated_sql = result["sql"],
    llm_judge     = llm_judge,
    execute_fn    = execute_fn,
    response      = result["response"],
    result_data   = result["data"],
    agent_steps   = result["agent_steps"],   # ReAct loop trace
    agent_result  = result,                  # for cache metrics
    weights       = WEIGHTS_V4,              # includes 10% agentic score
)

print(scores.agent_mode)        # "react"
print(scores.steps_taken)       # 4
print(scores.schema_grounding)  # 1.0  — schema inspected before SQL
print(scores.planning_quality)  # 0.85 — LLM judge on reasoning quality
print(scores.tokens_saved)      # 8600 — from semantic cache
```

Individual agentic metrics:

```python
from sqlas import steps_efficiency, schema_grounding, planning_quality, agentic_score

steps_efficiency(3)                              # 1.0 — optimal step count
steps_efficiency(7)                              # 0.6 — penalised
schema_grounding(steps)                          # 1.0 — schema inspected first
score, d = agentic_score(question, steps, llm_judge)
```

---

## Cache Performance Metrics (v2.0.0)

Track the ROI of your semantic caching layer.

```python
from sqlas import cache_hit_score, tokens_saved_score, few_shot_score

cache_hit_score(agent_result)       # 1.0 if served from cache
tokens_saved_score(agent_result)    # normalised savings, includes cost_saved_usd
few_shot_score(agent_result)        # 1.0 if verified examples were injected
```

---

## Weight Profiles

| Profile | Metrics | Use when |
|---|---|---|
| `WEIGHTS` | 15 | Standard NL→SQL pipeline |
| `WEIGHTS_V2` | 20 | + RAGAS context quality |
| `WEIGHTS_V3` | 30 | + Guardrails + visualization |
| `WEIGHTS_V4` | 28 | + Agentic quality ← **recommended for ReAct agents** |

```
WEIGHTS_V4 breakdown
  25% Execution Accuracy
  10% Semantic Correctness
   8% Context Quality
  10% Cost Efficiency
   7% Execution Quality
   8% Task Success
   7% Result + Visualization
  15% Guardrails
  10% Agentic Quality  ← new in v2.0.0
```

---

## Individual Metrics

```python
from sqlas import (
    execution_accuracy, semantic_equivalence, schema_compliance,
    faithfulness, answer_relevance, safety_score, guardrail_score,
    context_precision, context_recall, visualization_score,
    steps_efficiency, schema_grounding, planning_quality,
    cache_hit_score, tokens_saved_score,
)
```

---

## RAGAS Mapping

| RAGAS | SQLAS | Method |
|---|---|---|
| Faithfulness | `faithfulness` | LLM judge |
| Answer Relevance | `answer_relevance` | LLM judge |
| Answer Correctness | `execution_accuracy` | Automated |
| Answer Similarity | `result_set_similarity` | Automated (Jaccard) |
| Context Precision | `context_precision` | Automated |
| Context Recall | `context_recall` | Automated |
| Context Entity Recall | `entity_recall` | Automated |
| Noise Sensitivity | `noise_robustness` | Automated |

---

## LLM-Agnostic Judge

Any function `(prompt: str) -> str` works:

```python
# OpenAI
def judge(p): return openai.chat.completions.create(model="gpt-4o",
    messages=[{"role":"user","content":p}]).choices[0].message.content

# Anthropic
def judge(p): return anthropic.messages.create(model="claude-opus-4-7",
    max_tokens=500, messages=[{"role":"user","content":p}]).content[0].text

# Ollama (local)
def judge(p): return requests.post("http://localhost:11434/api/generate",
    json={"model":"llama3","prompt":p}).json()["response"]
```

---

## Use Any Database

```python
# PostgreSQL
def execute_fn(sql): return psycopg2_conn.cursor().execute(sql).fetchall()

# Snowflake
def execute_fn(sql): return snowflake_cursor.execute(sql).fetchall()

# BigQuery
def execute_fn(sql): return bq_client.query(sql).result()

scores = evaluate(..., execute_fn=execute_fn)   # takes precedence over db_path
```

---

## Production Features

- **AST-based read-only enforcement** — sqlglot parse tree validates SQL, not keyword matching
- **Timeout guard** — SQL execution timeout prevents server hangs
- **LLM resilience** — all judge calls wrapped with error handling
- **Type-checked** — ships `py.typed` for mypy/pyright
- **Structured logging** — uses Python `logging`, not print statements

---

## License

MIT — [thepradip](https://github.com/thepradip)

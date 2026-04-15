# SQLAS — SQL Agent Scoring Framework

**Production-grade evaluation framework for Text-to-SQL and SQL AI agents. 20 metrics. 8 categories. Any LLM.**

SQLAS scores your SQL agent the way production demands — execution accuracy, semantic correctness, context quality, cost efficiency, safety, and more. Built on industry benchmarks (Spider, BIRD) and real-world observability patterns (Arize, MLflow).

**Author:** [Pradip Tivhale](https://github.com/thepradip)

---

## Install

```bash
# From PyPI
pip install sqlas

# From source
git clone https://github.com/thepradip/SQLAS.git
cd SQLAS
pip install .

# With MLflow integration
pip install sqlas[mlflow]

# With dev tools
pip install sqlas[dev]
```

---

## Quick Start

```python
from sqlas import evaluate

# Your LLM judge function (any LLM: OpenAI, Anthropic, local, etc.)
def my_llm_judge(prompt: str) -> str:
    return client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
    ).choices[0].message.content

# Evaluate a single query
scores = evaluate(
    question="How many active users are there?",
    generated_sql="SELECT COUNT(*) FROM users WHERE active = 1",
    gold_sql="SELECT COUNT(*) FROM users WHERE active = 1",
    db_path="my_database.db",
    llm_judge=my_llm_judge,
    response="There are 1,523 active users.",
    result_data={"columns": ["COUNT(*)"], "rows": [[1523]], "row_count": 1, "execution_time_ms": 2.1},
)

print(scores.overall_score)  # 0.95
print(scores.summary())
```

---

## Evaluate Without Gold SQL

Gold SQL is optional. Without it, SQLAS uses semantic equivalence (LLM judge) and execution success:

```python
scores = evaluate(
    question="Show top 10 products by revenue",
    generated_sql="SELECT name, SUM(price * qty) AS rev FROM orders GROUP BY name ORDER BY rev DESC LIMIT 10",
    llm_judge=my_llm_judge,
    response="The top products are...",
    result_data={"columns": ["name", "rev"], "rows": [...], "row_count": 10, "execution_time_ms": 15},
)
```

---

## Run a Test Suite

```python
from sqlas import run_suite, TestCase

test_cases = [
    TestCase(
        question="How many users signed up this month?",
        gold_sql="SELECT COUNT(*) FROM users WHERE created_at >= '2026-03-01'",
        category="easy",
    ),
    TestCase(
        question="Average order value by country",
        gold_sql="SELECT country, AVG(total) FROM orders GROUP BY country",
        category="medium",
    ),
]

def my_agent(question: str) -> dict:
    # Your SQL agent pipeline
    sql = generate_sql(question)
    result = execute(sql)
    response = narrate(result)
    return {"sql": sql, "response": response, "data": result}

results = run_suite(
    test_cases=test_cases,
    agent_fn=my_agent,
    llm_judge=my_llm_judge,
    db_path="my_database.db",
    pass_threshold=0.6,  # configurable
)

print(results["summary"]["overall_score"])  # 0.88
```

---

## Metric Framework

### v1: Production Composite Score (15 metrics, 6 categories)

The default `WEIGHTS` profile uses 15 metrics:

```
SQLAS = 40% Execution Accuracy
      + 15% Semantic Correctness
      + 15% Cost Efficiency
      + 10% Execution Quality
      + 10% Task Success
      + 10% Safety
```

### v2: Full RAGAS-Mapped Score (20 metrics, 8 categories)

Use `WEIGHTS_V2` for the full 20-metric evaluation with context quality:

```python
from sqlas import evaluate, WEIGHTS_V2

scores = evaluate(..., weights=WEIGHTS_V2)
```

```
SQLAS v2 = 35% Execution Accuracy
         + 13% Semantic Correctness
         + 10% Context Quality (NEW — RAGAS-mapped)
         + 12% Cost Efficiency
         +  8% Execution Quality
         +  8% Task Success
         +  4% Result Similarity (NEW)
         + 10% Safety
```

### Detailed Breakdown (v2 — 20 metrics)

| Category | Metric | v1 Weight | v2 Weight | Method |
|---|---|---|---|---|
| **Execution Accuracy** | execution_accuracy | 40% | 35% | Automated: output + structure + efficiency |
| **Semantic Correctness** | semantic_equivalence | 15% | 13% | LLM-as-judge |
| **Context Quality** | context_precision | — | 3% | Automated: schema element precision vs gold |
| | context_recall | — | 3% | Automated: schema element recall vs gold |
| | entity_recall | — | 2% | Automated: strict entity-level recall |
| | noise_robustness | — | 2% | Automated: irrelevant schema resistance |
| **Cost Efficiency** | efficiency_score | 5% | 4% | Automated: VES |
| | data_scan_efficiency | 5% | 4% | Automated: scan detection |
| | sql_quality | 3% | 2% | LLM: join/agg/filter |
| | schema_compliance | 2% | 2% | Automated: sqlglot |
| **Execution Quality** | execution_success | 5% | 4% | Automated |
| | complexity_match | 3% | 2% | LLM-as-judge |
| | empty_result_penalty | 2% | 2% | Automated |
| **Task Success** | faithfulness | 4% | 3% | LLM-as-judge |
| | answer_relevance | 3% | 2% | LLM-as-judge |
| | answer_completeness | 2% | 2% | LLM-as-judge |
| | fluency | 1% | 1% | LLM-as-judge |
| **Result Similarity** | result_set_similarity | — | 4% | Automated: Jaccard on result sets |
| **Safety** | read_only_compliance | 5% | 5% | Automated: DDL/DML |
| | safety_score | 5% | 5% | Automated: PII/injection |

### Custom Weights

```python
my_weights = {
    "execution_accuracy": 0.50,  # increase correctness weight
    "semantic_equivalence": 0.10,
    "safety_score": 0.15,        # stricter safety
    # ... other metrics (must sum to 1.0)
}

scores = evaluate(..., weights=my_weights)
```

---

## Use Individual Metrics

```python
from sqlas import execution_accuracy, schema_compliance, safety_score
from sqlas import context_precision, context_recall, entity_recall

# Just check execution accuracy
score, details = execution_accuracy(
    generated_sql="SELECT COUNT(*) FROM users",
    gold_sql="SELECT COUNT(*) FROM users",
    db_path="my.db",
)

# Just check schema compliance
score, details = schema_compliance(
    sql="SELECT name FROM users",
    valid_tables={"users", "orders"},
    valid_columns={"users": {"id", "name", "email"}, "orders": {"id", "user_id", "total"}},
)

# Just check safety
score, details = safety_score(
    sql="SELECT * FROM users",
    pii_columns=["email", "phone", "ssn"],
)

# Context quality (requires gold SQL)
precision, details = context_precision(
    generated_sql="SELECT name, age FROM users WHERE active = 1",
    gold_sql="SELECT name FROM users WHERE active = 1",
)
# precision < 1.0 — 'age' is extra

recall, details = context_recall(
    generated_sql="SELECT name FROM users",
    gold_sql="SELECT name FROM users WHERE active = 1",
)
# recall < 1.0 — 'active' is missing
```

---

## Metric Mapping (vs. RAG Evaluation Standards)

| Standard Metric | SQLAS Equivalent | Description |
|---|---|---|
| Faithfulness | `faithfulness` | Claims grounded in SQL result data |
| Answer Relevance | `answer_relevance` | Response answers the question |
| Answer Correctness | `execution_accuracy` | SQL returns correct results |
| Answer Similarity | `result_set_similarity` | Result set Jaccard similarity |
| Context Precision | `context_precision` | Only relevant schema elements used |
| Context Recall | `context_recall` | All required schema elements used |
| Context Entity Recall | `entity_recall` | Strict entity match (tables, columns, literals, functions) |
| Noise Sensitivity | `noise_robustness` | Resistance to irrelevant schema context |
| — | `semantic_equivalence` | SQL answers the intent (LLM judge) |
| — | `safety_score` | PII + injection + DDL protection |
| — | `schema_compliance` | Valid tables/columns via AST |

---

## Production Features

- **Read-only DB**: All query execution uses read-only connections
- **Timeout guard**: SQL execution timeout (default 30s) prevents hangs
- **LLM resilience**: All LLM judge calls wrapped with error handling
- **Input validation**: Empty SQL, missing db_path, weight sum checks
- **Structured logging**: Uses Python `logging` module (not print)
- **Type-checked**: Ships `py.typed` marker for mypy/pyright

---

## LLM Judge

SQLAS is **LLM-agnostic**. Provide any function `(prompt: str) -> str`:

```python
# OpenAI
def judge(prompt):
    return openai_client.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": prompt}]
    ).choices[0].message.content

# Anthropic
def judge(prompt):
    return anthropic_client.messages.create(
        model="claude-sonnet-4-20250514", max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
    ).content[0].text

# Local (Ollama)
def judge(prompt):
    import requests
    return requests.post("http://localhost:11434/api/generate",
        json={"model": "llama3", "prompt": prompt}
    ).json()["response"]
```

---

## Example: SQL AI Agent (RAG-based)

A full-stack NL-to-SQL application is included in [`examples/sql-ai-agent/`](examples/sql-ai-agent/). It demonstrates:

- Dynamic schema introspection (any database, zero hardcoding)
- RAG-like context injection (schema + column stats + sample rows into LLM prompt)
- Read-only query execution with safety guards
- React frontend with chat interface
- MLflow tracing for full observability
- SQLAS evaluation integration

See the [example README](examples/sql-ai-agent/README.md) for setup instructions.

---

## License

MIT License - [Pradip Tivhale](https://github.com/thepradip)

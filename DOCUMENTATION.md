# SQLAS — SQL Agent Scoring Framework

## Complete Technical Documentation

**Version:** 1.1.0
**Author:** SQLAS Contributors
**License:** MIT

---

## Table of Contents

1. [Overview](#1-overview)
2. [Installation](#2-installation)
3. [Architecture](#3-architecture)
4. [Quick Start](#4-quick-start)
5. [Core Concepts](#5-core-concepts)
6. [API Reference](#6-api-reference)
   - 6.1 [evaluate()](#61-evaluate)
   - 6.2 [evaluate_batch()](#62-evaluate_batch)
   - 6.3 [run_suite()](#63-run_suite)
   - 6.4 [SQLASScores](#64-sqlasscores)
   - 6.5 [TestCase](#65-testcase)
   - 6.6 [WEIGHTS & WEIGHTS_V2](#66-weights--weights_v2)
   - 6.7 [compute_composite_score()](#67-compute_composite_score)
7. [Metrics Reference](#7-metrics-reference)
   - 7.1 [Execution Accuracy](#71-execution-accuracy)
   - 7.2 [Semantic Correctness](#72-semantic-correctness)
   - 7.3 [Context Quality (RAGAS-mapped)](#73-context-quality-ragas-mapped)
   - 7.4 [Cost Efficiency](#74-cost-efficiency)
   - 7.5 [Execution Quality](#75-execution-quality)
   - 7.6 [Task Success](#76-task-success)
   - 7.7 [Result Similarity](#77-result-similarity)
   - 7.8 [Safety & Governance](#78-safety--governance)
8. [RAGAS Mapping](#8-ragas-mapping)
9. [LLM Judge Interface](#9-llm-judge-interface)
10. [Production Features](#10-production-features)
11. [Configuration & Custom Weights](#11-configuration--custom-weights)
12. [Testing](#12-testing)
13. [Changelog](#13-changelog)

---

## 1. Overview

SQLAS (SQL Agent Scoring) is a production-grade evaluation framework for Text-to-SQL and SQL AI agents. It provides **20 metrics** across **8 categories**, designed as a RAGAS-equivalent for the SQL domain.

### Why SQLAS?

Existing evaluation frameworks like RAGAS are built for RAG (Retrieval-Augmented Generation) pipelines. SQL agents have fundamentally different requirements:

- **Execution correctness** — Does the SQL return the right data?
- **Schema awareness** — Does the agent use the correct tables and columns?
- **Query efficiency** — Is the SQL well-optimized?
- **Safety** — Does the agent avoid destructive operations and PII exposure?

SQLAS fills this gap with metrics specifically designed for SQL agent evaluation, while maintaining familiar patterns from RAGAS.

### Key Features

- **20 production-grade metrics** across 8 categories
- **LLM-agnostic** — works with OpenAI, Anthropic, local models, or any `(prompt: str) -> str` function
- **Automated + LLM-based** metrics for comprehensive evaluation
- **RAGAS-compatible** concepts adapted for the SQL domain
- **Production-hardened** — read-only DB, timeouts, LLM error handling
- **Lightweight** — only depends on `sqlglot` (optional: `mlflow`)
- **Type-checked** — ships `py.typed` for mypy/pyright
- **Backward compatible** — v1 WEIGHTS unchanged, new metrics default to 0.0

---

## 2. Installation

```bash
# Core package
pip install sqlas

# With MLflow integration
pip install sqlas[mlflow]

# With development tools
pip install sqlas[dev]
```

### Requirements

- Python >= 3.10
- sqlglot >= 20.0 (automatic)
- mlflow >= 3.0 (optional, for tracing)

---

## 3. Architecture

### Package Structure

```
sqlas/
├── __init__.py        # Public API exports (all 20 metrics + core types)
├── core.py            # SQLASScores, TestCase, WEIGHTS, WEIGHTS_V2, compute_composite_score
├── evaluate.py        # evaluate(), evaluate_batch() — main orchestration
├── correctness.py     # execution_accuracy, syntax_valid, semantic_equivalence, result_set_similarity
├── context.py         # context_precision, context_recall, entity_recall, noise_robustness
├── quality.py         # sql_quality, schema_compliance, complexity_match
├── production.py      # data_scan_efficiency, execution_result
├── response.py        # faithfulness, answer_relevance, answer_completeness, fluency
├── safety.py          # safety_score, read_only_compliance
├── runner.py          # run_suite() — test suite runner
└── py.typed           # PEP 561 type marker
```

### Data Flow

```
evaluate()
  ├── syntax_valid()              ← Automated (sqlglot)
  ├── execution_accuracy()        ← Automated (DB execution)
  ├── semantic_equivalence()      ← LLM Judge
  ├── context_precision()         ← Automated (sqlglot AST)
  ├── context_recall()            ← Automated (sqlglot AST)
  ├── entity_recall()             ← Automated (sqlglot AST)
  ├── noise_robustness()          ← Automated (sqlglot AST)
  ├── result_set_similarity()     ← Automated (DB execution)
  ├── schema_compliance()         ← Automated (sqlglot AST)
  ├── sql_quality()               ← LLM Judge
  ├── complexity_match()          ← LLM Judge
  ├── execution_result()          ← Automated (result data)
  ├── data_scan_efficiency()      ← Automated (pattern detection)
  ├── faithfulness()              ← LLM Judge
  ├── answer_relevance()          ← LLM Judge
  ├── answer_completeness()       ← LLM Judge
  ├── fluency()                   ← LLM Judge
  ├── read_only_compliance()      ← Automated (regex)
  ├── safety_score()              ← Automated (regex + patterns)
  └── compute_composite_score()   ← Weighted aggregation
```

### Metric Types

| Type | Count | Requires | Examples |
|------|-------|----------|----------|
| **Automated** | 13 | sqlglot, DB, or regex | execution_accuracy, syntax_valid, context_precision |
| **LLM-based** | 7 | llm_judge function | semantic_equivalence, faithfulness, sql_quality |

---

## 4. Quick Start

### Single Query Evaluation

```python
from sqlas import evaluate

# Define your LLM judge (any provider)
def my_llm_judge(prompt: str) -> str:
    return client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
    ).choices[0].message.content

# Evaluate
scores = evaluate(
    question="How many active users are there?",
    generated_sql="SELECT COUNT(*) FROM users WHERE active = 1",
    gold_sql="SELECT COUNT(*) FROM users WHERE active = 1",
    db_path="my_database.db",
    llm_judge=my_llm_judge,
    response="There are 1,523 active users.",
    result_data={
        "columns": ["COUNT(*)"],
        "rows": [[1523]],
        "row_count": 1,
        "execution_time_ms": 2.1,
    },
)

print(scores.overall_score)  # 0.95
print(scores.summary())
```

### Without Gold SQL

```python
scores = evaluate(
    question="Show top 10 products by revenue",
    generated_sql="SELECT name, SUM(price * qty) AS rev FROM orders GROUP BY name ORDER BY rev DESC LIMIT 10",
    llm_judge=my_llm_judge,
    response="The top products are...",
    result_data={"columns": ["name", "rev"], "rows": [...], "row_count": 10, "execution_time_ms": 15},
)
```

When `gold_sql` is not provided:
- `execution_accuracy` defaults to 1.0 if `result_data` is provided, 0.0 otherwise
- Context metrics (`context_precision`, `context_recall`, etc.) default to 0.0
- `result_set_similarity` defaults to 0.0
- LLM-based metrics still work (semantic_equivalence, sql_quality, etc.)

### Test Suite

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
    sql = generate_sql(question)
    result = execute(sql)
    response = narrate(result)
    return {"sql": sql, "response": response, "data": result}

results = run_suite(
    test_cases=test_cases,
    agent_fn=my_agent,
    llm_judge=my_llm_judge,
    db_path="my_database.db",
    pass_threshold=0.6,
)

print(results["summary"]["overall_score"])
```

---

## 5. Core Concepts

### Composite Score Formula

The overall SQLAS score is a weighted sum of individual metrics:

```
overall_score = SUM(metric_value * metric_weight) for all metrics in WEIGHTS
```

All metric values are normalized to 0.0-1.0. Weights sum to 1.0.

### Two Weight Profiles

**WEIGHTS (v1)** — 15 metrics, backward compatible:
```
40% Execution Accuracy
15% Semantic Correctness
15% Cost Efficiency
10% Execution Quality
10% Task Success
10% Safety
```

**WEIGHTS_V2** — 20 metrics, includes RAGAS context quality:
```
35% Execution Accuracy
13% Semantic Correctness
10% Context Quality (NEW)
12% Cost Efficiency
 8% Execution Quality
 8% Task Success
 4% Result Similarity (NEW)
10% Safety
```

### Graceful Degradation

Every parameter except `question`, `generated_sql`, and `llm_judge` is optional. SQLAS computes whatever metrics are possible with the given inputs:

| Input Provided | Metrics Enabled |
|---|---|
| `generated_sql` only | syntax_valid, read_only_compliance, safety_score, data_scan_efficiency |
| + `llm_judge` | + semantic_equivalence, sql_quality, complexity_match |
| + `gold_sql` | + context_precision, context_recall, entity_recall, noise_robustness |
| + `db_path` | + execution_accuracy, result_set_similarity, efficiency_score |
| + `response` + `result_data` | + faithfulness, answer_relevance, answer_completeness, fluency |
| + `valid_tables` + `valid_columns` | + schema_compliance (with actual validation) |

---

## 6. API Reference

### 6.1 evaluate()

```python
def evaluate(
    question: str,
    generated_sql: str,
    llm_judge: LLMJudge,
    gold_sql: str | None = None,
    db_path: str | None = None,
    response: str | None = None,
    result_data: dict | None = None,
    valid_tables: set[str] | None = None,
    valid_columns: dict[str, set[str]] | None = None,
    schema_context: str = "",
    expected_nonempty: bool = True,
    pii_columns: list[str] | None = None,
    weights: dict | None = None,
) -> SQLASScores
```

**Parameters:**

| Parameter | Type | Required | Description |
|---|---|---|---|
| `question` | `str` | Yes | User's natural language question |
| `generated_sql` | `str` | Yes | SQL produced by the agent |
| `llm_judge` | `(str) -> str` | Yes | LLM function for judge-based metrics |
| `gold_sql` | `str \| None` | No | Ground-truth SQL for comparison |
| `db_path` | `str \| None` | No | Path to SQLite database |
| `response` | `str \| None` | No | Agent's natural language response |
| `result_data` | `dict \| None` | No | Query results: `{columns, rows, row_count, execution_time_ms}` |
| `valid_tables` | `set[str] \| None` | No | Set of valid table names |
| `valid_columns` | `dict[str, set[str]] \| None` | No | Dict of `{table: {col1, col2, ...}}` |
| `schema_context` | `str` | No | Brief schema text for LLM judge |
| `expected_nonempty` | `bool` | No | Whether non-empty result is expected (default: True) |
| `pii_columns` | `list[str] \| None` | No | Custom PII column names for safety |
| `weights` | `dict \| None` | No | Custom weight dict (default: WEIGHTS) |

**Returns:** `SQLASScores` with all metrics and `overall_score`.

**Input Validation:**
- Empty `generated_sql` returns immediately with error in `details`
- Non-existent `db_path` returns immediately with error in `details`
- Custom `weights` that don't sum to ~1.0 emit a warning log

---

### 6.2 evaluate_batch()

```python
def evaluate_batch(
    test_cases: list[dict],
    llm_judge: LLMJudge,
    db_path: str | None = None,
    valid_tables: set[str] | None = None,
    valid_columns: dict[str, set[str]] | None = None,
    schema_context: str = "",
    pii_columns: list[str] | None = None,
    weights: dict | None = None,
) -> list[SQLASScores]
```

Each dict in `test_cases` should have:
- `question` (required)
- `generated_sql` (required)
- `gold_sql` (optional)
- `response` (optional)
- `result_data` (optional)
- `expected_nonempty` (optional, default True)
- `schema_context` (optional, overrides the function-level default)

**Returns:** List of `SQLASScores`, one per test case.

---

### 6.3 run_suite()

```python
def run_suite(
    test_cases: list[TestCase],
    agent_fn,
    llm_judge: LLMJudge,
    db_path: str | None = None,
    valid_tables: set[str] | None = None,
    valid_columns: dict[str, set[str]] | None = None,
    weights: dict | None = None,
    pass_threshold: float = 0.6,
    verbose: bool = True,
) -> dict
```

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `test_cases` | `list[TestCase]` | Test case objects |
| `agent_fn` | `(str) -> dict` | Agent function: receives question, returns `{"sql": ..., "response": ..., "data": ...}` |
| `llm_judge` | `(str) -> str` | LLM judge function |
| `pass_threshold` | `float` | Minimum score to count as PASS (default: 0.6) |
| `verbose` | `bool` | Print progress to stdout (default: True) |

**Returns:**

```python
{
    "summary": {
        "total_tests": int,
        "overall_score": float,
        "pass_rate": float,
        "time_seconds": float,
        "execution_accuracy": float,
        "semantic_equivalence": float,
        "context_precision": float,
        "context_recall": float,
        "entity_recall": float,
        "noise_robustness": float,
        "result_set_similarity": float,
        "sql_quality": float,
        "schema_compliance": float,
        "efficiency_score": float,
        "data_scan_efficiency": float,
        "faithfulness": float,
        "answer_relevance": float,
        "answer_completeness": float,
        "fluency": float,
        "read_only_compliance": float,
        "safety_score": float,
        "by_category": {"easy": 0.92, "medium": 0.85, ...},
    },
    "details": [SQLASScores, ...],
}
```

---

### 6.4 SQLASScores

Dataclass containing all metric scores for a single evaluation.

**Fields:**

| Field | Type | Default | Category |
|---|---|---|---|
| `execution_accuracy` | `float` | 0.0 | Correctness |
| `syntax_valid` | `float` | 0.0 | Correctness |
| `semantic_equivalence` | `float` | 0.0 | Correctness |
| `schema_compliance` | `float` | 0.0 | Quality |
| `sql_quality` | `float` | 0.0 | Quality |
| `complexity_match` | `float` | 0.0 | Quality |
| `execution_success` | `float` | 0.0 | Production |
| `execution_time_ms` | `float` | 0.0 | Production |
| `efficiency_score` | `float` | 0.0 | Production |
| `data_scan_efficiency` | `float` | 0.0 | Production |
| `result_row_count` | `int` | 0 | Production |
| `empty_result_penalty` | `float` | 0.0 | Production |
| `row_explosion_detected` | `bool` | False | Production |
| `faithfulness` | `float` | 0.0 | Response |
| `answer_relevance` | `float` | 0.0 | Response |
| `answer_completeness` | `float` | 0.0 | Response |
| `fluency` | `float` | 0.0 | Response |
| `read_only_compliance` | `float` | 0.0 | Safety |
| `safety_score` | `float` | 0.0 | Safety |
| `context_precision` | `float` | 0.0 | Context |
| `context_recall` | `float` | 0.0 | Context |
| `entity_recall` | `float` | 0.0 | Context |
| `noise_robustness` | `float` | 0.0 | Context |
| `result_set_similarity` | `float` | 0.0 | Context |
| `overall_score` | `float` | 0.0 | Composite |
| `details` | `dict` | `{}` | Diagnostics |

**Methods:**

- `to_dict() -> dict` — Export all scores as a flat dictionary
- `summary() -> str` — Human-readable multi-line summary

---

### 6.5 TestCase

```python
@dataclass
class TestCase:
    question: str                          # Natural language question
    gold_sql: str | None = None            # Ground-truth SQL
    expected_tables: list[str] | None = None  # Tables the query should reference
    expects_join: bool = False             # Whether a JOIN is expected
    expected_nonempty: bool = True         # Whether non-empty results are expected
    category: str = "general"              # Difficulty tier (e.g., "easy", "hard")
```

---

### 6.6 WEIGHTS & WEIGHTS_V2

Two predefined weight dictionaries:

**WEIGHTS (v1 — 15 metrics):**

```python
{
    "execution_accuracy": 0.40,
    "semantic_equivalence": 0.15,
    "efficiency_score": 0.05,
    "data_scan_efficiency": 0.05,
    "sql_quality": 0.03,
    "schema_compliance": 0.02,
    "execution_success": 0.05,
    "complexity_match": 0.03,
    "empty_result_penalty": 0.02,
    "faithfulness": 0.04,
    "answer_relevance": 0.03,
    "answer_completeness": 0.02,
    "fluency": 0.01,
    "read_only_compliance": 0.05,
    "safety_score": 0.05,
}
```

**WEIGHTS_V2 (v2 — 20 metrics):**

```python
{
    "execution_accuracy": 0.35,
    "semantic_equivalence": 0.13,
    "context_precision": 0.03,
    "context_recall": 0.03,
    "entity_recall": 0.02,
    "noise_robustness": 0.02,
    "efficiency_score": 0.04,
    "data_scan_efficiency": 0.04,
    "sql_quality": 0.02,
    "schema_compliance": 0.02,
    "execution_success": 0.04,
    "complexity_match": 0.02,
    "empty_result_penalty": 0.02,
    "faithfulness": 0.03,
    "answer_relevance": 0.02,
    "answer_completeness": 0.02,
    "fluency": 0.01,
    "result_set_similarity": 0.04,
    "read_only_compliance": 0.05,
    "safety_score": 0.05,
}
```

---

### 6.7 compute_composite_score()

```python
def compute_composite_score(scores: SQLASScores, weights: dict | None = None) -> float
```

Computes the weighted sum of all metrics. Uses `WEIGHTS` by default. Boolean fields are converted to 1.0/0.0. Missing fields default to 0.0.

---

## 7. Metrics Reference

### 7.1 Execution Accuracy

**File:** `correctness.py` | **Weight (v1):** 40% | **Type:** Automated

```python
def execution_accuracy(generated_sql: str, gold_sql: str, db_path: str) -> tuple[float, dict]
```

Executes both queries against the database and compares results.

**Formula:** `60% Output Match + 20% Structure Match + 20% Efficiency`

- **Output Match:** Row-by-row numeric comparison with tolerance (0.5). Ignores label differences (e.g., `0` vs `'Male'`), tolerates `ROUND` variations, handles extra columns.
- **Structure Match:** Same row count yields 1.0; otherwise proportional ratio.
- **Efficiency:** `min(gold_time / pred_time, 1.0)` — rewards queries that are at least as fast as the gold standard.

**Details returned:**
```python
{
    "output_score": 0.95,
    "structural_score": 1.0,
    "efficiency_score": 0.98,
    "predicted_rows": 5,
    "gold_rows": 5,
}
```

**Requirements:** `gold_sql`, `db_path`

---

### 7.2 Semantic Correctness

#### syntax_valid()

**File:** `correctness.py` | **Type:** Automated

```python
def syntax_valid(sql: str, dialect: str = "sqlite") -> float
```

Parses SQL with sqlglot. Returns `1.0` if valid, `0.0` if not.

#### semantic_equivalence()

**File:** `correctness.py` | **Weight (v1):** 15% | **Type:** LLM Judge

```python
def semantic_equivalence(
    question: str, generated_sql: str, llm_judge: LLMJudge, gold_sql: str | None = None
) -> tuple[float, dict]
```

LLM evaluates whether the generated SQL correctly answers the user's question. Considers:
- Correct data retrieval
- Right tables, columns, filters
- Correct aggregations
- Necessary JOINs

**Score guide:** 1.0 = perfect, 0.7-0.9 = minor issues, 0.4-0.6 = partial, 0.0-0.3 = major errors

---

### 7.3 Context Quality (RAGAS-mapped)

All context metrics use sqlglot AST extraction to compare schema elements between generated and gold SQL. They require `gold_sql` — without it, they default to 0.0.

#### context_precision()

**File:** `context.py` | **Weight (v2):** 3% | **Type:** Automated

```python
def context_precision(generated_sql: str, gold_sql: str, dialect: str = "sqlite") -> tuple[float, dict]
```

**RAGAS equivalent:** Context Precision

Of all schema elements (tables + columns) referenced in the generated SQL, what fraction are also in the gold SQL?

**Formula:** `|generated ∩ gold| / |generated|`

Penalizes referencing unnecessary schema elements. A score of 0.8 means 20% of referenced elements are not needed.

**Example:**
```python
# Generated SQL references 'age' which is not in gold SQL
precision, details = context_precision(
    "SELECT name, age FROM users WHERE active = 1",  # extra: age
    "SELECT name FROM users WHERE active = 1",
)
# precision = 0.75 (3 of 4 elements are relevant)
```

#### context_recall()

**File:** `context.py` | **Weight (v2):** 3% | **Type:** Automated

```python
def context_recall(generated_sql: str, gold_sql: str, dialect: str = "sqlite") -> tuple[float, dict]
```

**RAGAS equivalent:** Context Recall

Of all schema elements required by the gold SQL, what fraction does the generated SQL also reference?

**Formula:** `|generated ∩ gold| / |gold|`

Penalizes missing necessary elements. A score of 0.5 means half the required elements are missing.

**Example:**
```python
# Generated SQL missing 'active' column
recall, details = context_recall(
    "SELECT name FROM users",
    "SELECT name FROM users WHERE active = 1",
)
# recall < 1.0 — 'active' is missing
```

#### entity_recall()

**File:** `context.py` | **Weight (v2):** 2% | **Type:** Automated

```python
def entity_recall(generated_sql: str, gold_sql: str, dialect: str = "sqlite") -> tuple[float, dict]
```

**RAGAS equivalent:** Context Entity Recall

Stricter than `context_recall` — also checks literal values and function usage.

Entity types checked:
- Table names
- Column names
- Literal values (e.g., `'active'`, `1`, `'2026-03-01'`)
- Function names (e.g., `COUNT`, `AVG`, `ROUND`)

**Example:**
```python
# Generated SQL uses SELECT * instead of COUNT(*)
recall, details = entity_recall(
    "SELECT * FROM users WHERE active = 1",
    "SELECT COUNT(*) FROM users WHERE active = 1",
)
# recall < 1.0 — 'count' function is missing
```

#### noise_robustness()

**File:** `context.py` | **Weight (v2):** 2% | **Type:** Automated

```python
def noise_robustness(
    generated_sql: str, gold_sql: str,
    valid_tables: set[str] | None = None,
    valid_columns: dict[str, set[str]] | None = None,
    dialect: str = "sqlite",
) -> tuple[float, dict]
```

**RAGAS equivalent:** Noise Sensitivity

Measures resistance to irrelevant schema context. Checks if the generated SQL references tables/columns that exist in the full schema but are NOT needed by the gold SQL.

**Formula:** `1.0 - (noise_count / total_generated_elements)`

When `valid_tables`/`valid_columns` are provided, only counts extras that exist in the schema (real noise). Without them, falls back to comparing against gold SQL elements only.

---

### 7.4 Cost Efficiency

#### sql_quality()

**File:** `quality.py` | **Weight (v1):** 3% | **Type:** LLM Judge

```python
def sql_quality(question: str, generated_sql: str, llm_judge: LLMJudge, schema_context: str = "") -> tuple[float, dict]
```

LLM evaluates four dimensions:
- **Join Correctness** — Are JOINs logically correct?
- **Aggregation Accuracy** — Correct GROUP BY, COUNT, SUM, AVG?
- **Filter Accuracy** — WHERE clauses correct?
- **Efficiency** — No unnecessary subqueries?

Returns the LLM's `Overall_Quality` score (average of the four dimensions).

#### schema_compliance()

**File:** `quality.py` | **Weight (v1):** 2% | **Type:** Automated

```python
def schema_compliance(
    sql: str, valid_tables: set[str], valid_columns: dict[str, set[str]], dialect: str = "sqlite"
) -> tuple[float, dict]
```

Uses sqlglot AST to extract referenced tables and columns, then validates against the provided schema. Returns average of table_score and column_score. SQL keywords (COUNT, SUM, etc.) are filtered out.

#### data_scan_efficiency()

**File:** `production.py` | **Weight (v1):** 5% | **Type:** Automated

```python
def data_scan_efficiency(sql: str, result_row_count: int = 0) -> tuple[float, dict]
```

Detects inefficient patterns:
- `SELECT *` without specifying columns (-0.2)
- No WHERE, GROUP BY, or LIMIT — potential full scan (-0.3)
- Large result from JOIN — possible cartesian product (-0.3)
- No LIMIT on detail query returning many rows (-0.1)

#### complexity_match()

**File:** `quality.py` | **Weight (v1):** 3% | **Type:** LLM Judge

```python
def complexity_match(question: str, generated_sql: str, llm_judge: LLMJudge) -> tuple[float, dict]
```

LLM assesses if query complexity matches the question:
- **Over-engineering:** Unnecessary subqueries/CTEs for a simple question
- **Under-engineering:** Missing GROUP BY, JOIN, or aggregation
- **Correct strategy:** Aggregate before joining for 1:N relationships

---

### 7.5 Execution Quality

#### execution_result()

**File:** `production.py` | **Type:** Automated

```python
def execution_result(data: dict | None, expected_nonempty: bool = True) -> dict
```

Evaluates execution outcome from `result_data`:
- `execution_success`: 1.0 if data present, 0.0 if None
- `empty_result_penalty`: 0.0 if expected non-empty but got 0 rows, else 1.0
- `row_explosion_detected`: True if row_count > 50,000
- `execution_time_ms`: from result data
- `result_row_count`: from result data

---

### 7.6 Task Success

All response metrics require `response` and `result_data` to be provided. Without them, they default to 0.0.

#### faithfulness()

**File:** `response.py` | **Weight (v1):** 4% | **Type:** LLM Judge

**RAGAS equivalent:** Faithfulness

Checks if every factual claim in the agent's response is supported by the SQL result data. The LLM lists claims, marks each as SUPPORTED/UNSUPPORTED, and computes `faithfulness = supported / total`.

#### answer_relevance()

**File:** `response.py` | **Weight (v1):** 3% | **Type:** LLM Judge

**RAGAS equivalent:** Answer Relevance

Does the response directly answer the user's question? 1.0 = perfectly relevant, 0.0 = off-topic.

#### answer_completeness()

**File:** `response.py` | **Weight (v1):** 2% | **Type:** LLM Judge

Does the response surface ALL key information from the SQL result? 1.0 = all key points covered, 0.0 = most omitted.

#### fluency()

**File:** `response.py` | **Weight (v1):** 1% | **Type:** LLM Judge

Readability and coherence on a 1-5 scale (normalized to 0.0-1.0).

---

### 7.7 Result Similarity

#### result_set_similarity()

**File:** `correctness.py` | **Weight (v2):** 4% | **Type:** Automated

```python
def result_set_similarity(generated_sql: str, gold_sql: str, db_path: str) -> tuple[float, dict]
```

**RAGAS equivalent:** Answer Similarity

Executes both queries and computes Jaccard similarity on normalized result sets.

**Formula:** `80% Jaccard + 20% Column Count Match`

- Rows are normalized (strings lowered, floats rounded to 2 decimal places)
- Jaccard = `|intersection| / |union|`
- Column match = 1.0 if same column count, else proportional ratio

**Details returned:**
```python
{
    "jaccard": 0.95,
    "column_match": 1.0,
    "generated_row_count": 5,
    "gold_row_count": 5,
    "intersection_size": 5,
    "union_size": 5,
}
```

---

### 7.8 Safety & Governance

#### read_only_compliance()

**File:** `safety.py` | **Weight (v1):** 5% | **Type:** Automated

```python
def read_only_compliance(sql: str) -> float
```

Returns 1.0 (safe) or 0.0 (unsafe). Checks for forbidden keywords: `INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, `CREATE`, `TRUNCATE`, `GRANT`, `REVOKE`, `ATTACH`, `DETACH`.

#### safety_score()

**File:** `safety.py` | **Weight (v1):** 5% | **Type:** Automated

```python
def safety_score(sql: str, response: str = "", pii_columns: list[str] | None = None) -> tuple[float, dict]
```

Comprehensive safety evaluation:

| Check | Penalty | Pattern |
|---|---|---|
| DDL/DML keywords | -0.5 each | INSERT, UPDATE, DELETE, DROP, ALTER, etc. |
| Stacked queries | -0.3 | `; DROP`, `; DELETE`, etc. |
| UNION injection | -0.3 | `UNION SELECT` |
| Tautology | -0.3 | `OR 1=1` |
| PII column access | -0.2 each | password, ssn, email, phone_number, etc. |

PII detection uses word-boundary matching (`\b`) to avoid false positives (e.g., `ip_address_log` does NOT match `address`).

Custom PII columns can be provided via `pii_columns` parameter.

---

## 8. RAGAS Mapping

| RAGAS Metric | SQLAS Equivalent | SQL-Domain Meaning |
|---|---|---|
| **Faithfulness** | `faithfulness` | Claims in response grounded in SQL result data |
| **Answer Relevance** | `answer_relevance` | Response answers the user's question |
| **Answer Correctness** | `execution_accuracy` | SQL returns correct results |
| **Answer Similarity** | `result_set_similarity` | Result set Jaccard similarity |
| **Context Precision** | `context_precision` | Only relevant schema elements used |
| **Context Recall** | `context_recall` | All required schema elements used |
| **Context Entity Recall** | `entity_recall` | Strict entity match (tables, columns, literals, functions) |
| **Noise Sensitivity** | `noise_robustness` | Resistance to irrelevant schema context |
| — | `semantic_equivalence` | SQL answers the user's intent (LLM judge) |
| — | `sql_quality` | Join/aggregation/filter correctness |
| — | `schema_compliance` | Valid tables/columns via AST |
| — | `safety_score` | PII + injection + DDL protection |
| — | `read_only_compliance` | No destructive SQL statements |

---

## 9. LLM Judge Interface

SQLAS is **LLM-agnostic**. The `llm_judge` parameter accepts any function with signature `(prompt: str) -> str`.

### OpenAI

```python
from openai import OpenAI
client = OpenAI()

def judge(prompt: str) -> str:
    return client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
    ).choices[0].message.content
```

### Anthropic

```python
from anthropic import Anthropic
client = Anthropic()

def judge(prompt: str) -> str:
    return client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    ).content[0].text
```

### Azure OpenAI

```python
from openai import AzureOpenAI
client = AzureOpenAI(
    azure_endpoint="https://your-resource.openai.azure.com/",
    api_key="your-key",
    api_version="2024-12-01-preview",
)

def judge(prompt: str) -> str:
    return client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
    ).choices[0].message.content
```

### Local (Ollama)

```python
import requests

def judge(prompt: str) -> str:
    return requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "llama3", "prompt": prompt},
    ).json()["response"]
```

### Error Handling

All LLM judge calls are wrapped in `try/except`. If the judge function raises any exception (rate limit, network error, timeout), the metric returns `(0.0, {"error": "..."})` instead of crashing the evaluation.

---

## 10. Production Features

### Read-Only Database Connections

All SQL execution uses read-only SQLite connections:
```python
sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
```
This prevents any data corruption from generated SQL, even if it contains DDL/DML.

### SQL Execution Timeout

A progress handler aborts any query exceeding 30 seconds:
```python
conn.set_progress_handler(timeout_callback, 1_000_000)
```
This prevents indefinite hangs from cartesian products or recursive queries.

### LLM Error Resilience

Every LLM judge call is wrapped with error handling:
```python
try:
    result = llm_judge(prompt)
except Exception as e:
    logger.warning("LLM judge failed: %s", e)
    return 0.0, {"error": str(e)}
```

### Input Validation

`evaluate()` validates inputs before processing:
- Empty or non-string `generated_sql` -> returns error
- Non-existent `db_path` -> returns error
- Custom `weights` not summing to ~1.0 -> warning logged

### Structured Logging

All modules use `logging.getLogger(__name__)`:
```python
import logging
logging.basicConfig(level=logging.INFO)
# Now SQLAS will log warnings for LLM failures, validation issues, etc.
```

### Type Checking

SQLAS ships a `py.typed` marker (PEP 561), so mypy/pyright can type-check your code against the package's annotations.

---

## 11. Configuration & Custom Weights

### Custom Weight Profile

```python
from sqlas import evaluate

my_weights = {
    "execution_accuracy": 0.50,   # increase correctness weight
    "semantic_equivalence": 0.10,
    "safety_score": 0.15,         # stricter safety
    "read_only_compliance": 0.10,
    "faithfulness": 0.05,
    "sql_quality": 0.05,
    "schema_compliance": 0.05,
    # all weights must sum to 1.0
}

scores = evaluate(..., weights=my_weights)
```

Any metric key not in your custom weights dict will not contribute to the composite score. You can include any subset of the 20 available metrics.

### Using WEIGHTS_V2

```python
from sqlas import evaluate, WEIGHTS_V2

# Use the 20-metric profile with RAGAS context quality
scores = evaluate(..., weights=WEIGHTS_V2)
```

### Custom PII Columns

```python
scores = evaluate(
    ...,
    pii_columns=["social_security_number", "credit_card", "date_of_birth", "home_address"],
)
```

### Custom Pass Threshold

```python
from sqlas import run_suite

results = run_suite(
    ...,
    pass_threshold=0.7,  # stricter than default 0.6
)
```

---

## 12. Testing

### Running the Package Tests

```bash
# From the sqlas-package directory
pip install sqlas[dev]
pytest tests/ -v
```

### Test Coverage

| Test File | Tests | Coverage |
|---|---|---|
| `tests/test_sqlas.py` | 24 | Execution accuracy, syntax, schema, safety, composites, input validation, PII |
| `tests/test_context.py` | 19 | All 5 new context metrics + AST extraction |

### Writing Your Own Tests

```python
from sqlas import execution_accuracy, syntax_valid, context_precision

def test_my_agent_query():
    score, details = execution_accuracy(
        generated_sql="SELECT COUNT(*) FROM users WHERE active = 1",
        gold_sql="SELECT COUNT(*) FROM users WHERE active = 1",
        db_path="test.db",
    )
    assert score >= 0.9

def test_context_quality():
    precision, _ = context_precision(
        generated_sql="SELECT name, age FROM users",
        gold_sql="SELECT name FROM users",
    )
    assert precision < 1.0  # 'age' is unnecessary
```

---

## 13. Changelog

### [1.1.0] — 2026-03-30

**Added:**
- New `context.py` module with 4 RAGAS-mapped context metrics (context_precision, context_recall, entity_recall, noise_robustness)
- `result_set_similarity` metric (Jaccard on result sets)
- `WEIGHTS_V2` with all 20 metrics across 8 categories
- `py.typed` marker for PEP 561
- Configurable `pass_threshold` in `run_suite()`
- Input validation in `evaluate()`
- Structured logging via `logging` module

**Fixed:**
- Read-only DB connections prevent data corruption
- SQL execution timeout guard (30s default)
- LLM judge error handling across all metrics
- PII detection uses word boundaries (no false positives)
- `syntax_valid` simplified to trust sqlglot
- `evaluate_batch` forwards all parameters

### [1.0.1] — 2026-03-26

**Initial release:**
- 15 production metrics across 6 categories
- LLM-agnostic judge interface
- MLflow integration via optional dependency

# SQLAS — SQL Agent Scoring Framework

**A RAGAS-equivalent evaluation library for Text-to-SQL and Agentic SQL agents.**

[![PyPI](https://img.shields.io/pypi/v/sqlas)](https://pypi.org/project/sqlas/)
[![Python](https://img.shields.io/pypi/pyversions/sqlas)](https://pypi.org/project/sqlas/)
[![Tests](https://img.shields.io/badge/tests-140%20passing-brightgreen)](https://github.com/thepradip/SQLAS)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Evaluate SQL agents across **50+ metrics** — correctness, quality, safety, agentic reasoning, schema retrieval, prompt versioning, guardrails, and cache ROI. Aligned with Spider, BIRD, RAGAS, and MLflow standards.

**Author:** [thepradip](https://github.com/thepradip)

---

## Install

```bash
pip install sqlas                # core
pip install "sqlas[mlflow]"      # + MLflow
pip install "sqlas[ui]"          # + Streamlit UI
pip install "sqlas[all]"         # everything
```

---

## What's New in v2.7.0

| Feature | Description |
|---------|-------------|
| **Multi-gold SQL** | `execution_accuracy_best_of(sql, gold_sqls)` — evaluate against all valid gold queries, take best score. `TestCase.gold_sqls: list[str]` |
| **Hardness classification** | `auto_classify_hardness(sql)` → `easy/medium/hard/extra-hard` per BIRD criteria. Auto-set on every `evaluate()` call |
| **Exact match metric** | `exact_match(generated, gold)` — normalized string comparison. Exposed as `SQLASScores.exact_match_score` |
| **Failure classification** | `classify_failure(sql, scores, details)` → named `FailureCategory` with `top_hint()` actionable fix |
| **Batch crash isolation** | One failing test case no longer kills the entire batch |
| **LLM retry with backoff** | `_retry_llm_judge()` retries 3× (1s→2s→4s) on all 13 LLM judge call sites |
| **Weight normalization** | Custom weights auto-normalized to 1.0 instead of silently distorting scores |
| **LLM judge cache** | `enable_judge_cache()` — opt-in in-memory cache prevents re-scoring identical pairs in CI |
| **Report generation** | `generate_report(scores_list, format="markdown"\|"json")`, `to_json()`, `to_markdown_report()` |
| **Non-deterministic detection** | `NOW()`, `RANDOM()`, `CURRENT_TIMESTAMP` trigger `nondeterministic_warning` in details |
| **execute_fn timeout** | 30s wall-clock timeout with thread-safe SQLite fallback |
| **Safety patterns** | `UNION ALL SELECT`, `EXCEPT SELECT`, `WAITFOR DELAY`, file injection, NL prompt injection synonyms |

---

## Quick Start

```python
from sqlas import evaluate

def llm_judge(prompt: str) -> str:
    return openai_client.chat.completions.create(
        model="gpt-4o", messages=[{"role":"user","content":prompt}]
    ).choices[0].message.content

scores = evaluate(
    question      = "How many active users?",
    generated_sql = "SELECT COUNT(*) FROM users WHERE active = 1",
    gold_sql      = "SELECT COUNT(*) FROM users WHERE active = 1",
    db_path       = "my.db",
    llm_judge     = llm_judge,
    response      = "There are 1,523 active users.",
    result_data   = {"columns":["COUNT(*)"],"rows":[[1523]],"row_count":1,"execution_time_ms":2.1},
)

print(scores.overall_score)           # 0.95
print(scores.correctness_score)       # 0.88
print(scores.verdict)                 # PASS
print(scores.hardness)                # "easy"
print(scores.exact_match_score)       # 1.0
print(scores.to_markdown_report())    # Markdown for PR comments
```

---

## Failure Classification (v2.7)

Know exactly *why* a query failed — not just a score.

```python
from sqlas import classify_failure, FailureCategory

analysis = classify_failure(
    sql     = "SELECT id FROM users LIMIT 100",
    scores  = {"execution_accuracy": 1.0, "row_count_match": 0.12},
    details = {"row_count_match": {"pred_count": 100, "gold_count": 839}},
)

print(analysis.primary)        # FailureCategory.LIMIT_TRUNCATION
print(analysis.summary())      # "FAIL [limit_truncation] (score=1.000)"
print(analysis.top_hint())     # "Remove LIMIT — question asks for full results, not top-N"
print(analysis.evidence)       # {"limit_truncation": "LIMIT in SQL, 100 rows vs 839 expected"}
```

**All failure categories:**

| Category | Source |
|----------|--------|
| `LIMIT_TRUNCATION` | LIMIT silently cut result (100 vs 839 rows) |
| `WRONG_TABLE` | `accounting_transactions` used instead of `accounting` |
| `WRONG_AGGREGATION` | MAX instead of SUM, AVG instead of SUM |
| `SCALAR_MISMATCH` | Correlation or count value differs |
| `ROW_EXPLOSION` | 1:N join inflated row count |
| `SCHEMA_HALLUCINATION` | Invented table/column names (`counts`, `adm_count`, `n`) |
| `FULL_TABLE_SCAN` | SELECT * with no WHERE/LIMIT |
| `TRIM_ON_NUMERIC` | TRIM() on REAL column — invalid on Postgres |
| `UNSAFE_QUERY` | DDL/DML attempted |
| `CURRENCY_NOT_CLEANED` | Single REPLACE missed commas in `$1,234` |
| `NULL_IN_AGGREGATION` | AVG/SUM without IS NOT NULL |
| `JOIN_WITHOUT_FK` | Banking joined to users with no foreign key |
| `FAITHFULNESS_DROP` | Narration not grounded in SQL result |

---

## Multi-gold SQL (v2.7)

When a question has multiple valid SQL formulations, evaluate against all and take the best score:

```python
from sqlas import evaluate, TestCase

# Single evaluate call
scores = evaluate(
    question      = "Count active users",
    generated_sql = "SELECT COUNT(*) FROM users WHERE status = 'active'",
    gold_sql      = "SELECT COUNT(*) FROM users WHERE active = 1",   # primary gold
    db_path       = "my.db",
    llm_judge     = llm_judge,
)

# Batch with multiple gold SQLs per question
test_case = TestCase(
    question  = "Count active users",
    gold_sqls = [
        "SELECT COUNT(*) FROM users WHERE active = 1",
        "SELECT COUNT(*) FROM users WHERE status = 'active'",
        "SELECT COUNT(id) FROM users WHERE is_active = true",
    ],
)
```

---

## Hardness Classification (v2.7)

```python
from sqlas import auto_classify_hardness

auto_classify_hardness("SELECT COUNT(*) FROM users")
# → "easy"

auto_classify_hardness("SELECT u.id, SUM(o.total) FROM users u JOIN orders o ON u.id=o.user_id GROUP BY u.id HAVING SUM(o.total) > 1000")
# → "hard"

auto_classify_hardness("WITH ranked AS (SELECT *, ROW_NUMBER() OVER (...) FROM ...) SELECT ...")
# → "extra-hard"
```

Follows BIRD benchmark criteria. Auto-set on every `evaluate()` call as `SQLASScores.hardness`.

---

## Report Generation (v2.7)

```python
from sqlas import generate_report

# Batch markdown report — paste into PRs or CI comments
results = evaluate_batch(test_cases, llm_judge, db_path="my.db")
print(generate_report(results, questions, format="markdown"))

# JSON for artifact storage
print(generate_report(results, format="json"))

# Per-query reports
print(scores.to_json())
print(scores.to_markdown_report(question="How many users?", sql=generated_sql))
```

---

## LLM Judge Cache (v2.7)

Prevent re-scoring identical prompts in CI runs:

```python
from sqlas import enable_judge_cache, clear_judge_cache

enable_judge_cache()          # opt-in — identical prompts return cached result
results = evaluate_batch(...)
clear_judge_cache()           # clear between test runs
```

---

## Three-Dimension Scoring

`PASS` only when **all three** dimensions meet their thresholds:

```python
from sqlas import evaluate_correctness, evaluate_quality, evaluate_safety

c = evaluate_correctness(question, sql, llm_judge, gold_sql=gold, execute_fn=db)
q = evaluate_quality(question, sql, llm_judge, response=text, result_data=data)
s = evaluate_safety(sql, question=question, pii_columns=["email","ssn"])

print(c.score, c.verdict)   # 0.85  PASS   (threshold 0.5)
print(q.score, q.verdict)   # 0.72  PASS   (threshold 0.6)
print(s.score, s.verdict)   # 0.45  FAIL   (threshold 0.9 — PII detected)
```

`evaluate_safety()` makes **zero LLM calls** — pure regex + sqlglot AST.

---

## Guardrail Pipeline

```python
from sqlas import GuardrailPipeline

pipeline = GuardrailPipeline(pii_columns=["email","ssn","password"])

r = pipeline.check_input("List every user's SSN")    # blocks malicious NL intent
r = pipeline.check_sql(generated_sql)                # blocks injection/PII SQL
r = pipeline.check_output(response, result_data)     # blocks PII in response
```

**Injection patterns detected:** `UNION ALL SELECT`, `EXCEPT SELECT`, `INTERSECT SELECT`, stacked mutations, tautologies, time-based injection, file write/read, `WAITFOR DELAY`.

**NL prompt injection detected:** ignore/override/discard instructions, jailbreak, bypass guardrails, pretend unrestricted.

---

## Spider / BIRD Benchmark

```python
from sqlas.benchmarks import run_spider_benchmark

results = run_spider_benchmark(
    agent_fn   = my_agent,
    llm_judge  = llm_judge,
    spider_dir = "./spider",
    n_samples  = 50,          # stratified by difficulty → ~$0.25
    mlflow_run = True,
)
print(results["summary"]["overall_score"])
```

---

## Prompt Versioning

```python
from sqlas import PromptRegistry

registry = PromptRegistry()
registry.register("You are a SQL analyst...", version_id="v1")
registry.record("v1", scores)

status = registry.detect_regression("v1", window=50, threshold=0.05)
if status["regressed"]:
    for hint in status["hints"]:
        print(hint["hint"])   # actionable prompt fix suggestion
```

---

## Observability Integrations

```python
from sqlas.integrations import log_all

log_all(results,
    mlflow_experiment = "sql-agent-v2",
    wandb_project     = "sql-evals",
    langsmith_project = "my-sql-agent",
)
```

---

## Run a Test Suite

```python
from sqlas import run_suite, TestCase, WEIGHTS_V4, build_schema_info

tables, columns = build_schema_info(db_path="my.db")

results = run_suite(
    test_cases     = test_cases,
    agent_fn       = my_agent,
    llm_judge      = llm_judge,
    execute_fn     = execute_fn,
    valid_tables   = tables,
    valid_columns  = columns,
    weights        = WEIGHTS_V4,
    pass_threshold = 0.6,
)
print(results["summary"]["overall_score"])
print(results["summary"]["by_category"])
```

---

## Metrics Overview

| Dimension | Key Metrics |
|-----------|-------------|
| **Correctness** | Execution accuracy, exact match, multi-gold SQL, semantic equivalence, result set similarity |
| **SQL Quality** | SQL quality (LLM), schema compliance, complexity match, data scan efficiency |
| **Context (RAGAS)** | Context precision, recall, entity recall, noise robustness |
| **Response** | Faithfulness, answer relevance, completeness, fluency |
| **Agentic** | Steps efficiency, schema grounding, planning quality, tool use accuracy, plan compliance, first attempt success |
| **Safety** | Read-only compliance, SQL injection, prompt injection, PII access, PII leakage |
| **Production** | Execution success, VES efficiency, row explosion detection, empty result, result coverage |
| **Cache** | Cache hit score, tokens saved, few-shot examples used |
| **Visualization** | Chart spec validity, data alignment, LLM chart validation |

## Weight Profiles

| Profile | Metrics | Best for |
|---------|---------|----------|
| `WEIGHTS` | 15 | Standard NL→SQL pipeline |
| `WEIGHTS_V2` | 20 | + RAGAS context quality |
| `WEIGHTS_V3` | 30 | + Guardrails + visualization |
| `WEIGHTS_V4` | 28 | + Agentic quality ← ReAct agents |

---

## Changelog

### v2.7.0
- `classify_failure()` + `FailureCategory` enum — named failure classification with actionable hints
- `auto_classify_hardness()` — BIRD-aligned easy/medium/hard/extra-hard (auto-set on every eval)
- `exact_match()` + `SQLASScores.exact_match_score`
- `execution_accuracy_best_of()` + `TestCase.gold_sqls` — multi-gold SQL evaluation
- `generate_report()` — batch markdown/JSON report; `to_json()`, `to_markdown_report()` on SQLASScores
- `enable_judge_cache()` / `clear_judge_cache()` — opt-in LLM judge caching
- LLM retry with exponential backoff (3×) on all 13 LLM judge call sites
- Batch eval crash isolation — one failure no longer kills the batch
- Weight normalization — auto-normalize to 1.0 instead of silently distorting
- execute_fn timeout (30s) with thread-safe SQLite fallback
- Non-deterministic query detection (NOW, RANDOM, CURRENT_TIMESTAMP)
- Safety: UNION ALL SELECT, EXCEPT, WAITFOR DELAY, file injection, NL synonyms
- Division-by-zero guards in all context metrics

### v2.6.0
- Spider/BIRD benchmark (`run_spider_benchmark`, `run_bird_benchmark`)
- MLflow, W&B, LangSmith integrations (`sqlas.integrations`)
- Streamlit UI (`python -m sqlas ui`)
- React evaluation dashboard (`sqlas-ui/`)

### v2.5.0
- `plan_compliance()` — measures create_plan enforcement before execute_sql
- `first_attempt_success()` — measures SQL retry rate

### v2.4.0
- `PromptRegistry` — prompt versioning, regression detection, improvement hints
- `schema_retrieval_quality()` — precision/recall/F1 for schema index

### v2.3.0
- `GuardrailPipeline` — 3-stage safety: input → SQL → output (zero LLM cost)
- `FeedbackStore` — verified gold SQL from user thumbs-up

### v2.2.0
- Three-dimension scoring: `correctness_score`, `quality_score`, `safety_composite_score`
- `verdict` — AND logic: PASS only when all three pass thresholds

---

## License

MIT — [thepradip](https://github.com/thepradip) · [pypi.org/project/sqlas](https://pypi.org/project/sqlas/)

# SQLAS — SQL Agent Scoring Framework

**A RAGAS-equivalent evaluation library for Text-to-SQL and Agentic SQL agents.**

[![PyPI](https://img.shields.io/pypi/v/sqlas)](https://pypi.org/project/sqlas/)
[![Python](https://img.shields.io/pypi/pyversions/sqlas)](https://pypi.org/project/sqlas/)
[![Tests](https://img.shields.io/badge/tests-140%20passing-brightgreen)](https://github.com/thepradip/SQLAS)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Evaluate SQL agents across **45 metrics** — correctness, quality, safety, agentic reasoning, schema retrieval, prompt versioning, guardrails, and cache ROI. Aligned with Spider, BIRD, RAGAS, and MLflow standards.

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

## What's New in v2.6.0

| Feature | Description |
|---|---|
| `run_spider_benchmark()` | Spider/BIRD evaluation with smart stratified sampling (~$0.25 for 50 questions) |
| `log_to_mlflow/wandb/langsmith()` | One-call logging to all observability platforms |
| Streamlit UI | `python -m sqlas ui` → interactive evaluation dashboard |
| React UI | Full evaluation dashboard in `sqlas-ui/` |

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

print(scores.overall_score)              # 0.95
print(scores.correctness_score)          # 0.88
print(scores.quality_score)              # 0.93
print(scores.safety_composite_score)     # 1.00
print(scores.verdict)                    # PASS
```

---

## Three-Dimension Scoring (v2.2)

`PASS` only when **all three** dimensions meet their thresholds:

```python
from sqlas import evaluate_correctness, evaluate_quality, evaluate_safety

c = evaluate_correctness(question, sql, llm_judge, gold_sql=gold, execute_fn=db)
q = evaluate_quality(question, sql, llm_judge, response=text, result_data=data)
s = evaluate_safety(sql, question=question, pii_columns=["email","ssn"])

print(c.score, c.verdict)   # 0.85  PASS   (threshold 0.5)
print(q.score, q.verdict)   # 0.72  PASS   (threshold 0.6)
print(s.score, s.verdict)   # 0.45  FAIL   (threshold 0.9 — PII detected)
print(s.issues)             # ["PII_ACCESS: 'email'"]
```

`evaluate_safety()` makes **zero LLM calls** — pure regex + sqlglot AST.

---

## Spider / BIRD Benchmark (v2.6)

```python
from sqlas.benchmarks import run_spider_benchmark

results = run_spider_benchmark(
    agent_fn   = my_agent,
    llm_judge  = llm_judge,
    spider_dir = "./spider",     # download from yale-lily.github.io/spider
    n_samples  = 50,             # stratified 20/30/30/20 by difficulty → ~$0.25
    mlflow_run = True,
)
print(results["summary"]["overall_score"])   # 0.783
print(results["summary"]["safety_score"])    # 0.991  ← Spider has no safety baseline
print(results["cost_estimate_usd"])          # 0.25
```

---

## Observability Integrations (v2.6)

```python
from sqlas.integrations import log_all

results = sqlas.run_suite(test_cases, agent_fn, llm_judge)

log_all(
    results,
    mlflow_experiment = "sql-agent-v2",
    wandb_project     = "sql-evals",
    langsmith_project = "my-sql-agent",
)
```

---

## Guardrail Pipeline (v2.3)

```python
from sqlas import GuardrailPipeline

pipeline = GuardrailPipeline(pii_columns=["email","ssn","password"])

r = pipeline.check_input("List every user's SSN")    # blocks malicious intent
r = pipeline.check_sql(generated_sql)                # blocks injection/PII SQL
r = pipeline.check_output(response, result_data)     # blocks PII in response
```

---

## Feedback Learning Loop (v2.3)

```python
from sqlas import FeedbackStore, FeedbackEntry

store = FeedbackStore()
store.store(FeedbackEntry(question="How many active users?",
    sql="SELECT COUNT(*) FROM users WHERE status='active'",
    is_correct=True, score=0.95))

# Future evaluations auto-use stored gold SQL
c = evaluate_correctness(question, sql, llm_judge, feedback_store=store)
print(c.details["gold_sql_source"])   # "feedback_store"
```

---

## Prompt Versioning (v2.4)

```python
from sqlas import PromptRegistry

registry = PromptRegistry()
registry.register("You are a SQL analyst...", version_id="v1")
registry.record("v1", scores)

status = registry.detect_regression("v1", window=50, threshold=0.05)
if status["regressed"]:
    for hint in status["hints"]:
        print(hint["hint"])   # "Add to prompt: Only cite exact numbers..."
```

---

## Schema Retrieval Quality (v2.4)

```python
from sqlas import schema_retrieval_quality

score, details = schema_retrieval_quality(
    retrieved_tables = schema_index.retrieve(question),
    generated_sql    = agent_sql,
    gold_tables      = test_case.expected_tables,
)
print(details["precision"])   # 0.87
print(details["recall"])      # 1.00
print(details["missing"])     # []
```

---

## Plan Compliance & First-Attempt Success (v2.5)

```python
from sqlas.agentic import plan_compliance, first_attempt_success

# Did agent call create_plan before execute_sql?
score, d = plan_compliance(agent_steps)
print(score)   # 1.0 = compliant, 0.0 = skipped planning

# Did SQL succeed without retries?
score, d = first_attempt_success(agent_result)
print(score)   # 1.0 = first attempt, 0.7 = 1 retry, 0.0 = failed
```

---

## Streamlit UI (v2.6)

```bash
pip install "sqlas[ui]"
python -m sqlas ui    # → http://localhost:8501
```

## React UI (v2.6)

```bash
cd sqlas-ui
npm install
npm run dev           # → http://localhost:5173
```

6 pages: Dashboard · Evaluate · Benchmark · Results (all 45 metrics) · History · Settings

---

## Run a Test Suite

```python
from sqlas import run_suite, TestCase, WEIGHTS_V4, build_schema_info

tables, columns = build_schema_info(db_path="my.db")   # auto-extract schema

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

## Weight Profiles

| Profile | Metrics | Best for |
|---|---|---|
| `WEIGHTS` | 15 | Standard NL→SQL pipeline |
| `WEIGHTS_V2` | 20 | + RAGAS context quality |
| `WEIGHTS_V3` | 30 | + Guardrails + visualization |
| `WEIGHTS_V4` | 28 | + Agentic quality ← ReAct agents |

---

## RAGAS Mapping

| RAGAS | SQLAS | Notes |
|---|---|---|
| Faithfulness | `faithfulness` | Claims grounded in SQL result |
| Answer Relevance | `answer_relevance` | Answers the question |
| Context Precision | `context_precision` | Right schema elements used |
| Context Recall | `context_recall` | All required elements present |
| — | `plan_compliance` | Agent planned before executing |
| — | `first_attempt_success` | SQL succeeded without retries |
| — | `safety_score` | SQL-specific: injection + PII |

---

## Changelog

### v2.6.0
- Spider/BIRD benchmark with stratified sampling (`run_spider_benchmark`, `run_bird_benchmark`)
- MLflow, W&B, LangSmith, Prometheus integrations (`sqlas.integrations`)
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
- Standalone `evaluate_correctness()`, `evaluate_quality()`, `evaluate_safety()`

### v2.1.0
- `build_schema_info()` — auto-extract schema from any database
- `result_coverage` — truncation-aware GROUP BY penalty

### v2.0.0
- Agentic metrics: `steps_efficiency`, `schema_grounding`, `planning_quality`, `agentic_score`
- Cache metrics: `cache_hit_score`, `tokens_saved_score`, `few_shot_score`
- `WEIGHTS_V4` with 10% agentic quality dimension
- `read_only_compliance` upgraded to sqlglot AST

---

## License

MIT — [thepradip](https://github.com/thepradip) · [pypi.org/project/sqlas](https://pypi.org/project/sqlas/)

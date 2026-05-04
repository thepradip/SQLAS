# SQLAS — SQL Agent Scoring Framework

**A RAGAS-equivalent evaluation library for Text-to-SQL and Agentic SQL agents.**

[![PyPI](https://img.shields.io/pypi/v/sqlas)](https://pypi.org/project/sqlas/)
[![Python](https://img.shields.io/pypi/pyversions/sqlas)](https://pypi.org/project/sqlas/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-140%20passing-brightgreen)](https://github.com/thepradip/SQLAS)

Evaluate SQL agents across 45 metrics — correctness, quality, safety, agentic reasoning, schema retrieval, prompt versioning, and guardrails. Aligned with Spider, BIRD, RAGAS, and MLflow standards.

**Author:** [thepradip](https://github.com/thepradip)

---

## Install

```bash
pip install sqlas                # core
pip install "sqlas[mlflow]"      # + MLflow integration
```

---

## What's New in v2.4.0

| Feature | Description |
|---|---|
| `PromptRegistry` | Version prompts, compare A/B, detect regressions, get data-driven improvement hints |
| `schema_retrieval_quality` | Measure precision/recall of schema index — did it return the right tables? |
| `evaluate_correctness/quality/safety` | Three standalone evaluators — run only what you need |
| `GuardrailPipeline` | Three-stage safety: input → SQL → output (zero LLM cost) |
| `FeedbackStore` | Thumbs-up stores verified gold SQL, auto-improves `execution_accuracy` |
| Three-dimension verdict | `PASS` only when correctness + quality + safety ALL pass their thresholds |
| `result_coverage` | Penalises truncated GROUP BY (score 0.3) — catches big-dataset evaluation blind spots |

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

print(scores.overall_score)          # 0.95
print(scores.correctness_score)      # 0.88  (v2.2)
print(scores.quality_score)          # 0.93  (v2.2)
print(scores.safety_composite_score) # 1.00  (v2.2)
print(scores.verdict)                # PASS  (v2.2 — AND logic)
print(scores.summary())
```

---

## Three-Dimension Scoring (v2.2)

`PASS` requires **all three** dimensions to exceed their thresholds. A safe-but-wrong query no longer masks as PASS.

```python
from sqlas import evaluate_correctness, evaluate_quality, evaluate_safety

# Run only the metrics you need — each is fully independent
c = evaluate_correctness(question, sql, llm_judge, gold_sql=gold, execute_fn=db)
q = evaluate_quality(question, sql, llm_judge, response=text, result_data=data)
s = evaluate_safety(sql, question=question, pii_columns=["email","ssn"])

print(c.score, c.verdict)   # 0.85  PASS   (threshold 0.5)
print(q.score, q.verdict)   # 0.72  PASS   (threshold 0.6)
print(s.score, s.verdict)   # 0.45  FAIL   (threshold 0.9 — PII detected)
print(s.issues)             # ["PII_ACCESS: 'email'", "PII_ACCESS: 'ssn'"]
```

`evaluate_safety()` makes **zero LLM calls** — pure regex + sqlglot AST.

---

## Three-Stage Guardrail Pipeline (v2.3)

```python
from sqlas import GuardrailPipeline

pipeline = GuardrailPipeline(pii_columns=["email", "ssn", "password"])

# Stage 1 — before sending to LLM
r = pipeline.check_input("List every user's SSN and password")
if r.blocked: return {"error": r.block_reason}
# → BLOCK: DANGEROUS_INPUT: pii_bulk_request

# Stage 2 — after SQL generation, before execution
r = pipeline.check_sql("SELECT email, password FROM users")
if r.blocked: return {"error": r.block_reason}
# → score=0.80, issues=["PII_ACCESS: 'email'", "PII_ACCESS: 'password'"]

# Stage 3 — before returning response to user
r = pipeline.check_output(response, result_data)
if r.blocked: return {"error": r.block_reason}
# → scans result rows for PII patterns, blocks if found
```

---

## Prompt Versioning & Regression Detection (v2.4)

```python
from sqlas import PromptRegistry

registry = PromptRegistry()

# Register versions
registry.register("You are a SQL analyst...", version_id="v1", description="baseline")
registry.register("...Only cite exact numbers from the SQL result.", version_id="v2", description="grounding fix")

# Record scores after each evaluation
scores = evaluate(...)
registry.record("v2", scores)

# Compare versions
comp = registry.compare("v1", "v2")
print(comp["winner"])           # "v2"
print(comp["delta_overall"])    # +0.09
print(comp["improvements"])     # [{"metric": "faithfulness", "delta": "+0.27", ...}]

# Auto-detect regressions
status = registry.detect_regression("v2", window=50, threshold=0.05)
if status["regressed"]:
    for hint in status["hints"]:
        print(f"[{hint['severity']}] {hint['metric']} = {hint['score']}")
        print(f"  Fix: {hint['hint']}")
# [WARNING] faithfulness = 0.61
#   Fix: Add to prompt: 'Only cite exact numbers from the SQL result...'
```

---

## Schema Retrieval Quality (v2.4)

Measures whether the schema index returned the right tables for a query — not just whether the SQL used valid tables.

```python
from sqlas import schema_retrieval_quality

score, details = schema_retrieval_quality(
    retrieved_tables = schema_index.retrieve(question),   # what index returned
    generated_sql    = agent_sql,
    gold_tables      = test_case.expected_tables,         # ground truth
)

print(details["precision"])   # 0.50 — 2 of 4 retrieved tables were needed
print(details["recall"])      # 1.00 — both needed tables were retrieved
print(details["irrelevant"])  # ["lab_results", "medications"]
print(details["missing"])     # [] — no JOIN table was dropped
```

---

## Feedback Loop (v2.3)

Thumbs-up feedback stores verified gold SQL — future evaluations of the same question use it automatically.

```python
from sqlas import FeedbackStore, FeedbackEntry

store = FeedbackStore()

# User gives thumbs up → store as gold SQL
store.store(FeedbackEntry(
    question   = "How many active users?",
    sql        = "SELECT COUNT(*) FROM users WHERE status = 'active'",
    is_correct = True,
    score      = scores.overall_score,
))

# Next evaluation auto-uses stored gold SQL
c = evaluate_correctness(question, agent_sql, llm_judge, feedback_store=store)
# execution_accuracy is now verified (1.0) instead of unverified (0.5)
print(c.details["gold_sql_source"])  # "feedback_store"
```

---

## Any Database (v2.1)

```python
from sqlas import build_schema_info, run_suite

# Auto-extract schema from any database
tables, columns = build_schema_info(db_path="my.db")               # SQLite
tables, columns = build_schema_info(execute_fn=pg_execute_fn)       # PostgreSQL / Snowflake / BigQuery

results = run_suite(
    test_cases    = test_cases,
    agent_fn      = my_agent,
    llm_judge     = llm_judge,
    execute_fn    = execute_fn,
    valid_tables  = tables,      # 100+ tables — no problem
    valid_columns = columns,
)
```

---

## Run a Test Suite

```python
from sqlas import run_suite, TestCase

test_cases = [
    TestCase(question="How many users signed up this month?",
             gold_sql="SELECT COUNT(*) FROM users WHERE created_at >= '2026-03-01'",
             expected_tables=["users"], category="easy"),
    TestCase(question="Average order value by country",
             gold_sql="SELECT country, AVG(total) FROM orders GROUP BY country",
             expected_tables=["orders"], category="medium"),
]

def my_agent(question: str) -> dict:
    sql = generate_sql(question)
    return {"sql": sql, "response": narrate(sql), "data": execute(sql)}

results = run_suite(
    test_cases     = test_cases,
    agent_fn       = my_agent,
    llm_judge      = llm_judge,
    execute_fn     = execute_fn,
    pass_threshold = 0.6,
    verbose        = True,
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
| `WEIGHTS_V4` | 28 | + Agentic quality — ReAct agents |

---

## RAGAS Mapping

| RAGAS | SQLAS | Notes |
|---|---|---|
| Faithfulness | `faithfulness` | Claims grounded in SQL result |
| Answer Relevance | `answer_relevance` | Answers the question |
| Answer Correctness | `execution_accuracy` | SQL returns correct results |
| Context Precision | `context_precision` | Right schema elements used |
| Context Recall | `context_recall` | All required schema elements present |
| Noise Sensitivity | `noise_robustness` | Irrelevant schema ignored |
| — | `schema_retrieval_quality` | Did the index return the right tables? |
| — | `result_coverage` | Truncated GROUP BY detection |
| — | `agentic_score` | ReAct planning quality |

---

## LLM-Agnostic Judge

```python
# OpenAI
def judge(p): return openai.chat.completions.create(model="gpt-4o",
    messages=[{"role":"user","content":p}]).choices[0].message.content

# Anthropic
def judge(p): return anthropic.messages.create(model="claude-opus-4-7",
    max_tokens=500, messages=[{"role":"user","content":p}]).content[0].text

# Ollama (local, free)
def judge(p): return requests.post("http://localhost:11434/api/generate",
    json={"model":"llama3","prompt":p,"stream":False}).json()["response"]
```

---

## Changelog

### v2.4.0
- `PromptRegistry` — version prompts, compare A/B, detect regressions, get improvement hints
- `schema_retrieval_quality` — precision/recall/F1 for schema index evaluation
- `prompt_id` + `schema_retrieval_*` fields on `SQLASScores`

### v2.3.0
- `GuardrailPipeline` — 3-stage safety: `check_input`, `check_sql`, `check_output`
- `FeedbackStore` + `FeedbackEntry` — verified gold SQL from user thumbs-up
- `evaluate_correctness/quality/safety` — standalone metric evaluators

### v2.2.0
- Three-dimension scoring: `correctness_score`, `quality_score`, `safety_composite_score`
- `verdict` — AND logic: `PASS` only when all three pass thresholds
- `CorrectnessResult`, `QualityResult`, `SafetyResult` dataclasses

### v2.1.0
- `build_schema_info()` — auto-extract schema from any DB
- `result_coverage` — truncation-aware GROUP BY penalty
- `execution_accuracy` capped at 0.5 without gold SQL (was incorrectly 1.0)
- 100+ table support with focused schema context

### v2.0.0
- Agentic quality: `steps_efficiency`, `schema_grounding`, `planning_quality`, `agentic_score`
- Cache metrics: `cache_hit_score`, `tokens_saved_score`, `few_shot_score`
- `WEIGHTS_V4` — 28-metric profile with 10% agentic dimension
- `read_only_compliance` upgraded to sqlglot AST

---

## License

MIT — [thepradip](https://github.com/thepradip) · [pypi.org/project/sqlas](https://pypi.org/project/sqlas/)

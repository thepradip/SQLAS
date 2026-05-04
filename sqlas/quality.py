"""
SQL Quality & Structure Metrics.
- SQL Quality (join/aggregation/filter correctness via LLM)
- Schema Compliance (valid tables/columns via sqlglot)
- Complexity Match (appropriate complexity via LLM)

Author: SQLAS Contributors
"""

import logging

import sqlglot

from sqlas.core import LLMJudge, _parse_score

logger = logging.getLogger(__name__)


def sql_quality(
    question: str,
    generated_sql: str,
    llm_judge: LLMJudge,
    schema_context: str = "",
) -> tuple[float, dict]:
    """
    LLM evaluates join correctness, aggregation accuracy, filter accuracy, efficiency.

    Returns:
        (overall_score, {join_correctness, aggregation_accuracy, filter_accuracy, efficiency})
    """
    schema_block = ("**Referenced Tables Schema:**\n" + schema_context[:1500]) if schema_context else ""
    prompt = f"""You are a senior SQL reviewer. Evaluate the quality of this SQL query.

**User Question:** {question}

**Generated SQL:**
```sql
{generated_sql}
```

{schema_block}

Rate each 0.0-1.0:
1. **Join_Correctness**: Are JOINs logically correct? (1.0 if no joins needed and none used)
2. **Aggregation_Accuracy**: Correct GROUP BY, COUNT, SUM, AVG? (1.0 if no aggregation needed)
3. **Filter_Accuracy**: WHERE clauses correct?
4. **Efficiency**: No unnecessary subqueries or redundant operations?

Respond EXACTLY:
Join_Correctness: [score]
Aggregation_Accuracy: [score]
Filter_Accuracy: [score]
Efficiency: [score]
Overall_Quality: [average]
Issues: [list or "none"]"""

    try:
        result = llm_judge(prompt)
    except Exception as e:
        logger.warning("LLM judge failed in sql_quality: %s", e)
        return 0.0, {"error": str(e)}

    scores = {}
    for line in result.strip().split("\n"):
        for dim in ["Join_Correctness", "Aggregation_Accuracy", "Filter_Accuracy", "Efficiency", "Overall_Quality"]:
            if line.startswith(dim + ":"):
                val, _ = _parse_score(line, dim)
                scores[dim.lower()] = val

    overall = min(scores.get("overall_quality", 0.0), 1.0)
    return overall, {
        "join_correctness": scores.get("join_correctness", 0),
        "aggregation_accuracy": scores.get("aggregation_accuracy", 0),
        "filter_accuracy": scores.get("filter_accuracy", 0),
        "efficiency": scores.get("efficiency", 0),
    }


def schema_compliance(
    sql: str,
    valid_tables: set[str],
    valid_columns: dict[str, set[str]],
    dialect: str = "sqlite",
) -> tuple[float, dict]:
    """
    Check all referenced tables and columns exist in the schema.
    Uses sqlglot for AST parsing.

    Args:
        sql: Generated SQL
        valid_tables: Set of valid table names
        valid_columns: Dict of {table_name: {col1, col2, ...}}
        dialect: SQL dialect for parsing

    Returns:
        (score, details)
    """
    try:
        parsed = sqlglot.parse_one(sql, dialect=dialect)
    except Exception:
        return 0.0, {"error": "parse_failed"}

    referenced_tables = set()
    for table in parsed.find_all(sqlglot.exp.Table):
        if table.name:
            referenced_tables.add(table.name.lower())

    valid_tables_lower = {t.lower() for t in valid_tables}
    invalid_tables = referenced_tables - valid_tables_lower
    table_score = 1.0 if not invalid_tables else max(0, 1 - len(invalid_tables) / max(len(referenced_tables), 1))

    referenced_cols = set()
    for col in parsed.find_all(sqlglot.exp.Column):
        if col.name:
            referenced_cols.add(col.name.lower())

    all_valid_cols = set()
    for cols in valid_columns.values():
        all_valid_cols.update(c.lower() for c in cols)

    sql_keywords = {"count", "sum", "avg", "min", "max", "round", "coalesce", "cast", "case", "cnt", "null"}
    invalid_cols = (referenced_cols - all_valid_cols) - sql_keywords
    col_score = 1.0 if not invalid_cols else max(0, 1 - len(invalid_cols) / max(len(referenced_cols), 1))

    return round((table_score + col_score) / 2, 4), {
        "invalid_tables": list(invalid_tables),
        "invalid_columns": list(invalid_cols),
        "table_score": table_score,
        "column_score": col_score,
    }


def complexity_match(
    question: str,
    generated_sql: str,
    llm_judge: LLMJudge,
) -> tuple[float, dict]:
    """
    LLM judges whether SQL complexity is appropriate for the question.
    Detects over-engineering and under-engineering.
    """
    prompt = f"""You are a SQL expert. Assess if the query complexity matches the question.

**Question:** {question}

**SQL:**
```sql
{generated_sql}
```

Check:
- Over-engineering: unnecessary subqueries/CTEs for a simple question
- Under-engineering: missing GROUP BY, JOIN, or aggregation
- Correct join strategy: aggregate before joining for 1:N relationships

Score 0.0-1.0:
- 1.0: Exactly as complex as needed
- 0.7-0.9: Minor issues
- 0.4-0.6: Noticeable issues
- 0.0-0.3: Major issues

Respond EXACTLY:
Complexity_Match: [score]
Reasoning: [one sentence]"""

    try:
        result = llm_judge(prompt)
    except Exception as e:
        logger.warning("LLM judge failed in complexity_match: %s", e)
        return 0.0, {"error": str(e)}

    score, reasoning = _parse_score(result, "Complexity_Match")
    return score, {"reasoning": reasoning}

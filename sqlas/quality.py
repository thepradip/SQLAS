"""
SQL Quality & Structure Metrics.
- SQL Quality (join/aggregation/filter correctness via LLM)
- Schema Compliance (valid tables/columns via sqlglot)
- Complexity Match (appropriate complexity via LLM)
- Dialect Correctness (dialect-specific syntax validation)
- Join Path Correctness (FK/expected join path validation)
- Aggregation/Grain Correctness (GROUP BY grain, duplicate counting, DISTINCT)

Author: SQLAS Contributors
"""

import logging
import re

import sqlglot

from sqlas.core import LLMJudge, _parse_score, _retry_llm_judge

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
        result = _retry_llm_judge(llm_judge, prompt)
    except Exception as e:
        logger.warning("LLM judge failed in sql_quality: %s", e)
        return 0.0, {"error": str(e), "llm_error": True}

    scores: dict = {}
    any_parsed = False
    for line in result.strip().split("\n"):
        for dim in ["Join_Correctness", "Aggregation_Accuracy", "Filter_Accuracy", "Efficiency", "Overall_Quality"]:
            if line.startswith(dim + ":"):
                val, _, ok = _parse_score(line, dim)
                scores[dim.lower()] = val
                if ok:
                    any_parsed = True

    overall = min(scores.get("overall_quality", 0.0), 1.0)
    details: dict = {
        "join_correctness": scores.get("join_correctness", 0),
        "aggregation_accuracy": scores.get("aggregation_accuracy", 0),
        "filter_accuracy": scores.get("filter_accuracy", 0),
        "efficiency": scores.get("efficiency", 0),
    }
    if not any_parsed:
        details["llm_parse_warning"] = True
    return overall, details


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


def dialect_correctness(
    sql: str,
    dialect: str,
    llm_judge: "LLMJudge | None" = None,
) -> "tuple[float, dict]":
    """
    Validate SQL syntax against a specific SQL dialect.

    Uses sqlglot to transpile and parse the SQL for the target dialect.
    Catches dialect-specific errors such as:
    - Snowflake QUALIFY / ILIKE / FLATTEN
    - BigQuery STRUCT / ARRAY_AGG / EXCEPT columns
    - Databricks PIVOT / UNPIVOT / AI_QUERY
    - MySQL IFNULL vs COALESCE, backtick quoting
    - Postgres-specific functions (DATE_PART, EXTRACT, FILTER clause)

    If llm_judge is provided and sqlglot parse succeeds, the LLM does a second
    pass to catch semantic dialect mistakes (e.g., wrong date arithmetic for
    the target warehouse).

    Args:
        sql:       Generated SQL.
        dialect:   Target dialect: "sqlite", "postgres", "snowflake",
                   "bigquery", "databricks", "mysql", "duckdb".
        llm_judge: Optional LLM judge for semantic dialect check.

    Returns:
        (score 0.0–1.0, details)
    """
    SUPPORTED_DIALECTS = {
        "sqlite", "postgres", "postgresql", "snowflake",
        "bigquery", "databricks", "mysql", "duckdb", "tsql", "spark",
    }
    dialect_lower = dialect.lower()

    # Syntax check via sqlglot — use RAISE so parse errors are surfaced as exceptions
    parse_errors: list[str] = []
    transpile_warnings: list[str] = []
    try:
        statements = sqlglot.parse(sql, dialect=dialect_lower, error_level=sqlglot.ErrorLevel.RAISE)
        for stmt in statements:
            if stmt is None:
                parse_errors.append("null statement returned from parser")
    except sqlglot.errors.ParseError as e:
        parse_errors.extend(str(err)[:200] for err in (e.errors or [str(e)]))
    except Exception as e:
        parse_errors.append(f"unexpected parse error: {str(e)[:200]}")

    # Transpile round-trip to catch semantic incompatibilities
    if not parse_errors:
        try:
            sqlglot.transpile(sql, read=dialect_lower, write=dialect_lower)
        except Exception as e:
            transpile_warnings.append(str(e)[:200])

    if parse_errors:
        score = 0.0
        details: dict = {
            "dialect": dialect,
            "parse_errors": parse_errors,
            "syntax_valid": False,
        }
        return score, details

    base_score = 0.9 if transpile_warnings else 1.0

    if dialect_lower not in SUPPORTED_DIALECTS:
        return base_score, {
            "dialect": dialect,
            "note": f"dialect '{dialect}' not in known set — syntax check only",
            "syntax_valid": True,
        }

    # Optional LLM semantic check
    if llm_judge is None:
        return base_score, {
            "dialect": dialect,
            "syntax_valid": True,
            "transpile_warnings": transpile_warnings or ["none"],
            "llm_check": False,
        }

    prompt = f"""You are a {dialect} SQL expert. Review this SQL for dialect-specific issues.

**Target dialect:** {dialect}

**SQL:**
```sql
{sql}
```

Check for dialect-specific problems:
- Functions that don't exist in {dialect} (e.g., ILIKE in MySQL, QUALIFY in Postgres)
- Date arithmetic syntax differences
- String functions, regex, or JSON path syntax
- Window function syntax differences
- Any features not supported in {dialect}

Score 0.0-1.0:
- 1.0: Fully valid {dialect} SQL
- 0.7: Minor dialect mismatch (would run with small fix)
- 0.4: Wrong dialect functions or syntax
- 0.0: SQL won't run in {dialect} at all

Respond EXACTLY:
Dialect_Score: [score]
Issues: [list or "none"]
Reasoning: [one sentence]"""

    try:
        result = _retry_llm_judge(llm_judge, prompt)
    except Exception as e:
        logger.warning("LLM judge failed in dialect_correctness: %s", e)
        return base_score, {
            "dialect": dialect,
            "syntax_valid": True,
            "llm_error": str(e),
        }

    llm_score, reasoning, parse_ok = _parse_score(result, "Dialect_Score")
    issues = "none"
    for line in result.strip().split("\n"):
        if line.startswith("Issues:"):
            issues = line.split(":", 1)[-1].strip()

    combined = round(min(base_score, llm_score), 4)
    details = {
        "dialect": dialect,
        "syntax_valid": True,
        "llm_dialect_score": llm_score,
        "issues": issues,
        "reasoning": reasoning,
        "transpile_warnings": transpile_warnings or ["none"],
    }
    if not parse_ok:
        details["llm_parse_warning"] = True
    return combined, details


def join_path_correctness(
    sql: str,
    fk_map: "dict[str, list[tuple[str, str, str, str]]] | None" = None,
    llm_judge: "LLMJudge | None" = None,
    schema_context: str = "",
    dialect: str = "sqlite",
) -> "tuple[float, dict]":
    """
    Validate that JOINs use correct foreign-key paths.

    Schema compliance checks whether tables/columns exist, but not whether the
    joins use the right keys. This metric catches wrong-key joins that return
    incorrect results (e.g., joining on name instead of id, or using the wrong
    FK in a multi-FK table).

    Args:
        sql:            Generated SQL.
        fk_map:         FK definitions as {table: [(local_col, ref_table, ref_col, rel_type), ...]}.
                        If None, falls back to LLM-only evaluation.
        llm_judge:      LLM judge function (optional if fk_map is provided).
        schema_context: Optional schema text for LLM context.
        dialect:        SQL dialect for sqlglot parsing.

    Returns:
        (score 0.0–1.0, details)
    """
    upper = sql.upper()
    if "JOIN" not in upper:
        return 1.0, {"note": "no JOINs in query"}

    fk_violations: list[str] = []
    join_pairs: list[dict] = []

    # AST-based FK check when fk_map is provided
    if fk_map:
        try:
            parsed = sqlglot.parse_one(sql, dialect=dialect)
            for join in parsed.find_all(sqlglot.exp.Join):
                # Extract ON condition columns
                on_expr = join.args.get("on")
                if on_expr is None:
                    continue
                join_cols: list[tuple[str, str]] = []
                for eq in on_expr.find_all(sqlglot.exp.EQ):
                    left, right = eq.left, eq.right
                    if isinstance(left, sqlglot.exp.Column) and isinstance(right, sqlglot.exp.Column):
                        join_cols.append((
                            f"{left.table}.{left.name}".lower(),
                            f"{right.table}.{right.name}".lower(),
                        ))
                        join_pairs.append({"left": f"{left.table}.{left.name}", "right": f"{right.table}.{right.name}"})

                # Validate against fk_map
                fk_map_lower = {
                    t.lower(): [(lc.lower(), rt.lower(), rc.lower()) for lc, rt, rc, *_ in fks]
                    for t, fks in fk_map.items()
                }
                for left_ref, right_ref in join_cols:
                    left_table = left_ref.split(".")[0] if "." in left_ref else ""
                    right_table = right_ref.split(".")[0] if "." in right_ref else ""
                    left_col = left_ref.split(".")[-1]
                    right_col = right_ref.split(".")[-1]

                    valid = False
                    if left_table in fk_map_lower:
                        for lc, rt, rc in fk_map_lower[left_table]:
                            if lc == left_col and rt == right_table and rc == right_col:
                                valid = True
                    if right_table in fk_map_lower:
                        for lc, rt, rc in fk_map_lower[right_table]:
                            if lc == right_col and rt == left_table and rc == left_col:
                                valid = True
                    if not valid and (left_col or right_col):
                        fk_violations.append(
                            f"JOIN {left_ref} = {right_ref} — not found in FK map"
                        )
        except Exception as e:
            logger.debug("FK AST parse error in join_path_correctness: %s", e)

        if not llm_judge:
            if fk_violations:
                score = max(0.0, round(1.0 - 0.25 * len(fk_violations), 4))
                return score, {"fk_violations": fk_violations, "join_pairs": join_pairs}
            return 1.0, {"fk_violations": ["none"], "join_pairs": join_pairs}

    # LLM judge for semantic join correctness
    if llm_judge is None:
        return 1.0, {"note": "no fk_map and no llm_judge provided — check skipped"}

    schema_block = f"\n**Schema / FK relationships:**\n{schema_context[:1000]}" if schema_context else ""
    fk_block = ""
    if fk_violations:
        fk_block = f"\n**Detected FK violations (AST):**\n" + "\n".join(f"- {v}" for v in fk_violations)

    prompt = f"""You are a SQL expert evaluating whether JOINs use correct foreign key paths.

**Generated SQL:**
```sql
{sql}
```
{schema_block}
{fk_block}

Evaluate:
1. Do all JOINs use the correct columns to link tables?
2. Are the join keys the actual FK/PK relationships (not just same-named columns)?
3. Are join directions correct (no reversed FK references)?
4. Are any intermediate tables missing (join shortcuts that skip required tables)?

Score 0.0-1.0:
- 1.0: All JOINs use correct FK/PK paths
- 0.7: Minor join issue (correct tables, slightly wrong column)
- 0.4: JOIN on wrong column — results will be incorrect or produce duplicates
- 0.0: Fundamentally wrong join path — cartesian or completely wrong key

Respond EXACTLY:
Join_Path_Score: [score]
Issues: [list or "none"]
Reasoning: [one sentence]"""

    try:
        result = _retry_llm_judge(llm_judge, prompt)
    except Exception as e:
        logger.warning("LLM judge failed in join_path_correctness: %s", e)
        if fk_violations:
            score = max(0.0, round(1.0 - 0.25 * len(fk_violations), 4))
            return score, {"fk_violations": fk_violations, "llm_error": str(e)}
        return 1.0, {"llm_error": str(e)}

    score, reasoning, parse_ok = _parse_score(result, "Join_Path_Score")
    issues = "none"
    for line in result.strip().split("\n"):
        if line.startswith("Issues:"):
            issues = line.split(":", 1)[-1].strip()

    # Combine AST FK violations with LLM score
    if fk_violations:
        score = round(min(score, max(0.0, 1.0 - 0.25 * len(fk_violations))), 4)

    details: dict = {
        "reasoning": reasoning,
        "issues": issues,
        "fk_violations": fk_violations or ["none"],
        "join_pairs": join_pairs,
    }
    if not parse_ok:
        details["llm_parse_warning"] = True
    return score, details


def aggregation_grain_correctness(
    question: str,
    sql: str,
    llm_judge: LLMJudge,
    schema_context: str = "",
) -> "tuple[float, dict]":
    """
    Evaluate aggregation grain — one of the most common SQL agent failure modes.

    Detects:
    - Wrong grouping level (too granular or too coarse)
    - Duplicate counting (missing DISTINCT in COUNT)
    - Incorrect denominator in ratio/percentage calculations
    - Bad date grain (daily data requested but monthly aggregation produced)
    - Fan-out from JOINs before aggregation (1:N multiplies counts)
    - Missing aggregation entirely (returns detail rows when summary expected)

    Args:
        question:       Original user question.
        sql:            Generated SQL.
        llm_judge:      LLM judge function.
        schema_context: Optional schema context.
    """
    upper = sql.upper()

    # Heuristic pre-checks
    grain_risks: list[str] = []

    has_group = "GROUP BY" in upper
    has_agg = bool(re.search(r"\b(COUNT|SUM|AVG|MIN|MAX|COUNT_DISTINCT)\s*\(", upper))
    has_join = "JOIN" in upper
    has_distinct = "DISTINCT" in upper

    if has_join and has_agg and not has_distinct:
        grain_risks.append(
            "JOIN before aggregation without DISTINCT — possible fan-out duplicate counting"
        )
    if has_agg and not has_group and re.search(r"\bCOUNT\s*\(\s*[^*\s]", upper):
        grain_risks.append("COUNT(col) without GROUP BY — returns single row, verify intent")

    schema_block = f"\n**Schema:**\n{schema_context[:800]}" if schema_context else ""

    prompt = f"""You are a SQL expert evaluating aggregation grain correctness.

**User Question:** {question}

**Generated SQL:**
```sql
{sql}
```
{schema_block}

Evaluate:
1. Is the GROUP BY at the right grain for the question asked?
2. Is COUNT/SUM/AVG applied to the right column and without fan-out from JOINs?
3. For percentages/ratios: is the denominator correct?
4. For time series: is the date truncation grain (day/week/month) correct?
5. Is DISTINCT needed in COUNT to avoid duplicate counting?

Score 0.0-1.0:
- 1.0: Aggregation grain is exactly correct
- 0.7: Minor grain issue (slightly wrong period, one GROUP BY column off)
- 0.4: Wrong grain that produces incorrect totals (fan-out, wrong denominator)
- 0.0: Fundamentally wrong aggregation (missing GROUP BY, counting wrong thing)

Respond EXACTLY:
Grain_Score: [score]
Issues: [list or "none"]
Reasoning: [one sentence]"""

    try:
        result = _retry_llm_judge(llm_judge, prompt)
    except Exception as e:
        logger.warning("LLM judge failed in aggregation_grain_correctness: %s", e)
        return 0.0, {"error": str(e), "llm_error": True}

    score, reasoning, parse_ok = _parse_score(result, "Grain_Score")
    issues = "none"
    for line in result.strip().split("\n"):
        if line.startswith("Issues:"):
            issues = line.split(":", 1)[-1].strip()

    details: dict = {
        "reasoning": reasoning,
        "issues": issues,
        "grain_risks": grain_risks or ["none"],
    }
    if not parse_ok:
        details["llm_parse_warning"] = True
    return score, details


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
        result = _retry_llm_judge(llm_judge, prompt)
    except Exception as e:
        logger.warning("LLM judge failed in complexity_match: %s", e)
        return 0.0, {"error": str(e), "llm_error": True}

    score, reasoning, parse_ok = _parse_score(result, "Complexity_Match")
    details: dict = {"reasoning": reasoning}
    if not parse_ok:
        details["llm_parse_warning"] = True
    return score, details

"""
Context Quality Metrics (RAGAS-mapped for SQL agents).
- Context Precision (schema element precision)
- Context Recall (schema element recall)
- Entity Recall (strict entity-level recall)
- Noise Robustness (irrelevant schema resistance)

Author: SQLAS Contributors
"""

import sqlglot


# ── Shared AST Extraction ──────────────────────────────────────────────────

def _extract_sql_elements(sql: str, dialect: str = "sqlite") -> dict:
    """Extract tables, columns, literals, and functions from SQL AST.

    Returns:
        {
            "tables": set of lowered table names,
            "columns": set of lowered column names,
            "table_columns": set of (table, column) tuples,
            "literals": set of string representations of literal values,
            "functions": set of lowered function names,
        }
    """
    try:
        parsed = sqlglot.parse_one(sql, dialect=dialect)
    except Exception:
        return {
            "tables": set(),
            "columns": set(),
            "table_columns": set(),
            "literals": set(),
            "functions": set(),
        }

    tables = set()
    for table in parsed.find_all(sqlglot.exp.Table):
        if table.name:
            tables.add(table.name.lower())

    columns = set()
    table_columns = set()
    for col in parsed.find_all(sqlglot.exp.Column):
        if col.name:
            col_name = col.name.lower()
            columns.add(col_name)
            tbl = col.table.lower() if col.table else ""
            if tbl:
                table_columns.add((tbl, col_name))
            else:
                table_columns.add(("", col_name))

    literals = set()
    for lit in parsed.find_all(sqlglot.exp.Literal):
        literals.add(str(lit.this).lower())

    functions = set()
    # Built-in aggregate / scalar expressions
    for cls in (sqlglot.exp.Count, sqlglot.exp.Sum, sqlglot.exp.Avg,
                sqlglot.exp.Min, sqlglot.exp.Max):
        for _ in parsed.find_all(cls):
            functions.add(cls.key.lower())
    # Named functions (e.g. ROUND, COALESCE, custom UDFs)
    for func in parsed.find_all(sqlglot.exp.Anonymous):
        if func.name:
            functions.add(func.name.lower())

    return {
        "tables": tables,
        "columns": columns,
        "table_columns": table_columns,
        "literals": literals,
        "functions": functions,
    }


# ── Public API ─────────────────────────────────────────────────────────────

def context_precision(
    generated_sql: str,
    gold_sql: str,
    dialect: str = "sqlite",
) -> tuple[float, dict]:
    """
    RAGAS Context Precision for SQL agents.

    Of all schema elements (tables + columns) referenced in the generated SQL,
    what fraction are also referenced in the gold SQL? Penalizes referencing
    unnecessary schema elements.

    Args:
        generated_sql: SQL produced by the agent
        gold_sql: Ground-truth SQL
        dialect: SQL dialect for parsing

    Returns:
        (precision score 0.0–1.0, details dict)
    """
    gen = _extract_sql_elements(generated_sql, dialect)
    gold = _extract_sql_elements(gold_sql, dialect)

    gen_elements = gen["tables"] | gen["columns"]
    gold_elements = gold["tables"] | gold["columns"]

    if not gen_elements:
        return 1.0, {"generated_elements": [], "gold_elements": list(gold_elements),
                      "extra_elements": [], "precision": 1.0}

    overlap = gen_elements & gold_elements
    extra = gen_elements - gold_elements
    precision = len(overlap) / len(gen_elements)

    return round(precision, 4), {
        "generated_elements": sorted(gen_elements),
        "gold_elements": sorted(gold_elements),
        "extra_elements": sorted(extra),
        "precision": round(precision, 4),
    }


def context_recall(
    generated_sql: str,
    gold_sql: str,
    dialect: str = "sqlite",
) -> tuple[float, dict]:
    """
    RAGAS Context Recall for SQL agents.

    Of all schema elements (tables + columns) required by the gold SQL,
    what fraction does the generated SQL also reference? Penalizes missing
    necessary elements.

    Args:
        generated_sql: SQL produced by the agent
        gold_sql: Ground-truth SQL
        dialect: SQL dialect for parsing

    Returns:
        (recall score 0.0–1.0, details dict)
    """
    gen = _extract_sql_elements(generated_sql, dialect)
    gold = _extract_sql_elements(gold_sql, dialect)

    gen_elements = gen["tables"] | gen["columns"]
    gold_elements = gold["tables"] | gold["columns"]

    if not gold_elements:
        return 1.0, {"missing_elements": [], "recall": 1.0}

    overlap = gen_elements & gold_elements
    missing = gold_elements - gen_elements
    recall = len(overlap) / len(gold_elements)

    return round(recall, 4), {
        "missing_elements": sorted(missing),
        "matched_elements": sorted(overlap),
        "total_gold_elements": len(gold_elements),
        "recall": round(recall, 4),
    }


def entity_recall(
    generated_sql: str,
    gold_sql: str,
    dialect: str = "sqlite",
) -> tuple[float, dict]:
    """
    RAGAS Context Entity Recall for SQL agents.

    Strict entity-level check: are all entities (tables, columns, literals,
    functions) from the gold SQL present in the generated SQL?

    This is stricter than context_recall because it also checks literal values
    (e.g. WHERE status = 'active') and function usage (COUNT, SUM, etc.).

    Args:
        generated_sql: SQL produced by the agent
        gold_sql: Ground-truth SQL
        dialect: SQL dialect for parsing

    Returns:
        (recall score 0.0–1.0, details dict)
    """
    gen = _extract_sql_elements(generated_sql, dialect)
    gold = _extract_sql_elements(gold_sql, dialect)

    gold_entities = gold["tables"] | gold["columns"] | gold["literals"] | gold["functions"]
    gen_entities = gen["tables"] | gen["columns"] | gen["literals"] | gen["functions"]

    if not gold_entities:
        return 1.0, {"missing_entities": [], "matched_entities": [],
                      "total_gold_entities": 0}

    matched = gold_entities & gen_entities
    missing = gold_entities - gen_entities
    recall = len(matched) / len(gold_entities)

    return round(recall, 4), {
        "missing_entities": sorted(missing),
        "matched_entities": sorted(matched),
        "total_gold_entities": len(gold_entities),
    }


def noise_robustness(
    generated_sql: str,
    gold_sql: str,
    valid_tables: set[str] | None = None,
    valid_columns: dict[str, set[str]] | None = None,
    dialect: str = "sqlite",
) -> tuple[float, dict]:
    """
    RAGAS Noise Sensitivity for SQL agents.

    Does the agent avoid pulling in irrelevant schema elements? Measures
    resilience to a large, noisy schema by checking if the generated SQL
    references tables/columns that exist in the full schema but are NOT
    needed by the gold SQL.

    Args:
        generated_sql: SQL produced by the agent
        gold_sql: Ground-truth SQL
        valid_tables: Full set of available table names (optional)
        valid_columns: Full dict of {table: {col1, col2, ...}} (optional)
        dialect: SQL dialect for parsing

    Returns:
        (robustness score 0.0–1.0, details dict)
    """
    gen = _extract_sql_elements(generated_sql, dialect)
    gold = _extract_sql_elements(gold_sql, dialect)

    gold_elements = gold["tables"] | gold["columns"]
    gen_elements = gen["tables"] | gen["columns"]

    if not gen_elements:
        return 1.0, {"noise_tables": [], "noise_columns": [], "noise_count": 0}

    # Elements in generated SQL that are NOT in gold SQL
    extra = gen_elements - gold_elements

    # If full schema is provided, only count extras that exist in the schema
    # (i.e. real noise, not hallucinated tables/columns)
    if valid_tables or valid_columns:
        all_schema = set()
        if valid_tables:
            all_schema.update(t.lower() for t in valid_tables)
        if valid_columns:
            for cols in valid_columns.values():
                all_schema.update(c.lower() for c in cols)
        noise = extra & all_schema
    else:
        noise = extra

    noise_tables = noise & gen["tables"]
    noise_columns = noise & gen["columns"]
    noise_count = len(noise)

    score = max(0.0, 1.0 - (noise_count / len(gen_elements)))

    return round(score, 4), {
        "noise_tables": sorted(noise_tables),
        "noise_columns": sorted(noise_columns),
        "noise_count": noise_count,
    }

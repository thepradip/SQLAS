"""
Core SQL Correctness Metrics.
- Execution Accuracy (output + structure + efficiency)
- Syntax Validity (sqlglot parse)
- Semantic Equivalence (LLM-as-Judge)
- Result Set Similarity (Jaccard on normalized result sets)

Author: SQLAS Contributors
"""

import logging
import time
import sqlite3

import sqlglot

from sqlas.core import LLMJudge, ExecuteFn, _parse_score

logger = logging.getLogger(__name__)

# Default SQL execution timeout in seconds
_DEFAULT_TIMEOUT_S = 30
_PROGRESS_INTERVAL = 1_000_000  # check timeout every N SQLite VM instructions


def _connect_readonly(db_path: str, timeout_s: int = _DEFAULT_TIMEOUT_S) -> sqlite3.Connection:
    """Open a read-only SQLite connection with a timeout guard."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=timeout_s)
    _start = time.monotonic()

    def _progress_handler():
        if time.monotonic() - _start > timeout_s:
            return 1  # non-zero → abort query
        return 0

    conn.set_progress_handler(_progress_handler, _PROGRESS_INTERVAL)
    return conn


# ── Helpers ─────────────────────────────────────────────────────────────────

def _extract_row_numbers(row) -> list[float]:
    return sorted([round(float(v), 2) for v in row if isinstance(v, (int, float)) and v is not None])


def _values_found_in(needle: list[float], haystack: list[float], tol: float = 0.5) -> float:
    if not needle:
        return 1.0
    remaining = list(haystack)
    matched = 0
    for nv in needle:
        best_idx, best_diff = -1, float("inf")
        for i, hv in enumerate(remaining):
            diff = abs(nv - hv)
            if diff < best_diff:
                best_diff, best_idx = diff, i
        if best_idx >= 0 and best_diff <= tol:
            remaining.pop(best_idx)
            matched += 1
    return matched / len(needle)


def _row_values_match(pred_nums: list[float], gold_nums: list[float], tol: float = 0.5) -> float:
    if not gold_nums and not pred_nums:
        return 1.0
    if not gold_nums:
        return 0.8
    if not pred_nums:
        return 0.0
    if len(pred_nums) < len(gold_nums):
        subset_score = _values_found_in(pred_nums, gold_nums, tol)
        return 1.0 if subset_score >= 0.99 else subset_score
    return _values_found_in(gold_nums, pred_nums, tol)


def _match_result_sets(pred_rows: list, gold_rows: list) -> float:
    if not gold_rows:
        return 1.0 if not pred_rows else 0.5
    pred_nums_list = [_extract_row_numbers(r) for r in pred_rows]
    gold_nums_list = [_extract_row_numbers(r) for r in gold_rows]
    used_pred = set()
    total_score = 0.0
    for gn in gold_nums_list:
        best_score, best_pi = 0.0, -1
        for pi, pn in enumerate(pred_nums_list):
            if pi in used_pred:
                continue
            score = _row_values_match(pn, gn)
            if score > best_score:
                best_score, best_pi = score, pi
        if best_pi >= 0:
            used_pred.add(best_pi)
        total_score += best_score
    return total_score / len(gold_rows)


# ── Public API ──────────────────────────────────────────────────────────────

def execution_accuracy(
    generated_sql: str,
    gold_sql: str,
    db_path: str | None = None,
    execute_fn: ExecuteFn | None = None,
) -> tuple[float, dict]:
    """
    Semantic execution accuracy.

    Formula: 60% Output Match + 20% Structure Match + 20% Efficiency

    Output Match:  Row-by-row numeric comparison. Ignores label differences
                   (0 vs 'Male'), tolerates ROUND, handles extra columns.
    Structure:     Same row count.
    Efficiency:    Generated query speed vs gold query speed.

    Args:
        generated_sql: SQL produced by the agent
        gold_sql: Ground-truth SQL
        db_path: Path to SQLite database (backward-compatible)
        execute_fn: Optional callable (sql: str) -> list[tuple].
                    When provided, takes precedence over db_path and enables
                    evaluation against any database (Postgres, MySQL, Snowflake, etc.)

    Returns:
        (score, details) where score is 0.0–1.0
    """
    if execute_fn is not None:
        try:
            start = time.perf_counter()
            gold_result = list(execute_fn(gold_sql))
            gold_time = max((time.perf_counter() - start) * 1000, 0.01)

            start = time.perf_counter()
            pred_result = list(execute_fn(generated_sql))
            pred_time = max((time.perf_counter() - start) * 1000, 0.01)
        except Exception as e:
            logger.warning("execute_fn failed in execution_accuracy: %s", e)
            return 0.0, {"error": str(e)}
    elif db_path is not None:
        try:
            conn = _connect_readonly(db_path)
        except Exception as e:
            return 0.0, {"error": f"db_connect_failed: {e}"}
        try:
            start = time.perf_counter()
            gold_result = conn.execute(gold_sql).fetchall()
            gold_time = max((time.perf_counter() - start) * 1000, 0.01)

            start = time.perf_counter()
            pred_result = conn.execute(generated_sql).fetchall()
            pred_time = max((time.perf_counter() - start) * 1000, 0.01)
        except Exception as e:
            return 0.0, {"error": str(e)}
        finally:
            conn.close()
    else:
        return 0.0, {"error": "db_path or execute_fn required for execution_accuracy"}

    output_score = _match_result_sets(pred_result, gold_result)

    struct_score = 0.0
    if len(pred_result) == len(gold_result):
        struct_score = 1.0
    elif pred_result and gold_result:
        struct_score = min(len(pred_result), len(gold_result)) / max(len(pred_result), len(gold_result))

    time_ratio = gold_time / pred_time if pred_time > 0 else 1.0
    efficiency = min(time_ratio, 1.0)

    final = round(0.6 * output_score + 0.2 * struct_score + 0.2 * efficiency, 4)

    return final, {
        "output_score": round(output_score, 4),
        "structural_score": round(struct_score, 4),
        "efficiency_score": round(efficiency, 4),
        "predicted_rows": len(pred_result),
        "gold_rows": len(gold_result),
    }


def syntax_valid(sql: str, dialect: str = "sqlite") -> float:
    """Check if SQL parses without errors. Returns 1.0 or 0.0."""
    try:
        results = sqlglot.parse(sql, dialect=dialect)
        return 1.0 if results and results[0] is not None else 0.0
    except Exception:
        return 0.0


def semantic_equivalence(
    question: str,
    generated_sql: str,
    llm_judge: LLMJudge,
    gold_sql: str | None = None,
) -> tuple[float, dict]:
    """
    LLM judges whether the SQL correctly answers the user's question.
    Handles alias differences, join variations, CASE WHEN labels.

    Args:
        question: User's natural language question
        generated_sql: SQL produced by the agent
        llm_judge: Function (prompt: str) -> str
        gold_sql: Optional reference SQL for comparison

    Returns:
        (score, details) where score is 0.0–1.0
    """
    gold_section = f"\n**Reference SQL:**\n```sql\n{gold_sql}\n```" if gold_sql else ""

    prompt = f"""You are a SQL expert judge. Evaluate if the Generated SQL correctly answers the User Question.
{gold_section}

**User Question:** {question}

**Generated SQL:**
```sql
{generated_sql}
```

Evaluate:
1. Does the SQL retrieve the correct data to answer the question?
2. Are the right tables, columns, and filters used?
3. Are aggregations applied correctly?
4. Are JOINs correct and necessary?

Score 0.0 to 1.0:
- 1.0: Perfectly answers the question
- 0.7-0.9: Minor issues not affecting the core answer
- 0.4-0.6: Partially correct, missing key elements
- 0.0-0.3: Wrong approach or major errors

Respond EXACTLY:
Semantic_Score: [score]
Reasoning: [one sentence]"""

    try:
        result = llm_judge(prompt)
    except Exception as e:
        logger.warning("LLM judge failed in semantic_equivalence: %s", e)
        return 0.0, {"error": str(e)}

    score, reasoning = _parse_score(result, "Semantic_Score")
    return score, {"reasoning": reasoning}


def result_set_similarity(
    generated_sql: str,
    gold_sql: str,
    db_path: str | None = None,
    execute_fn: ExecuteFn | None = None,
) -> tuple[float, dict]:
    """
    RAGAS Answer Similarity for SQL agents.

    Computes Jaccard similarity on normalized result sets between
    generated and gold SQL execution outputs.

    Args:
        generated_sql: SQL produced by the agent
        gold_sql: Ground-truth SQL
        db_path: Path to SQLite database (backward-compatible)
        execute_fn: Optional callable (sql: str) -> list[tuple].
                    When provided, takes precedence over db_path.

    Returns:
        (similarity score 0.0–1.0, details dict)
    """
    if execute_fn is not None:
        try:
            gold_rows = list(execute_fn(gold_sql))
            pred_rows = list(execute_fn(generated_sql))
        except Exception as e:
            logger.warning("execute_fn failed in result_set_similarity: %s", e)
            return 0.0, {"error": str(e)}
        # Infer column count from rows; 0 if result is empty
        gold_cols = len(gold_rows[0]) if gold_rows else 0
        pred_cols = len(pred_rows[0]) if pred_rows else 0
    elif db_path is not None:
        try:
            conn = _connect_readonly(db_path)
        except Exception as e:
            return 0.0, {"error": f"db_connect_failed: {e}"}
        try:
            gold_rows = conn.execute(gold_sql).fetchall()
            gold_desc = conn.execute(gold_sql).description
            pred_rows = conn.execute(generated_sql).fetchall()
            pred_desc = conn.execute(generated_sql).description
        except Exception as e:
            return 0.0, {"error": str(e)}
        finally:
            conn.close()
        gold_cols = len(gold_desc) if gold_desc else 0
        pred_cols = len(pred_desc) if pred_desc else 0
    else:
        return 0.0, {"error": "db_path or execute_fn required for result_set_similarity"}

    def _normalize_row(row):
        cells = []
        for v in row:
            if isinstance(v, float):
                cells.append(round(v, 2))
            elif isinstance(v, str):
                cells.append(v.strip().lower())
            else:
                cells.append(v)
        return tuple(cells)

    gold_set = {_normalize_row(r) for r in gold_rows}
    pred_set = {_normalize_row(r) for r in pred_rows}

    union = gold_set | pred_set
    intersection = gold_set & pred_set

    jaccard = len(intersection) / len(union) if union else 1.0

    col_match = 1.0 if gold_cols == pred_cols else (
        min(gold_cols, pred_cols) / max(gold_cols, pred_cols) if max(gold_cols, pred_cols) > 0 else 1.0
    )

    score = round(0.8 * jaccard + 0.2 * col_match, 4)

    return score, {
        "jaccard": round(jaccard, 4),
        "column_match": round(col_match, 4),
        "generated_row_count": len(pred_rows),
        "gold_row_count": len(gold_rows),
        "intersection_size": len(intersection),
        "union_size": len(union),
    }

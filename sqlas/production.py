"""
Production Execution Metrics.
- Data Scan Efficiency (full scan detection)
- Execution Result (success, empty result, row explosion)

Author: SQLAS Contributors
"""

import re


def data_scan_efficiency(
    sql: str,
    result_row_count: int = 0,
    truncated: bool = False,
) -> tuple[float, dict]:
    """
    Detect inefficient data access patterns.

    v2.1.1 fix: result_row_count is capped at max_result_rows (default 500) by the agent,
    so checking row_count alone misses row explosions on large tables.
    We now use the ``truncated`` flag as the authoritative signal:
    - truncated=True with JOIN + no LIMIT  → very likely row explosion (100K+ rows)
    - truncated=True without GROUP BY/LIMIT → query returns far too many rows
    """
    upper = sql.upper()
    issues = []
    score = 1.0

    if "SELECT *" in upper or "SELECT  *" in upper:
        issues.append("SELECT * — should specify columns")
        score -= 0.2

    has_where = "WHERE" in upper
    has_group = "GROUP BY" in upper
    has_limit = "LIMIT" in upper

    if not has_where and not has_group and not has_limit:
        issues.append("No WHERE, GROUP BY, or LIMIT — potential full table scan")
        score -= 0.3

    # Use truncated flag (reliable) instead of raw row count (capped at 500)
    if truncated and has_group:
        issues.append("Aggregation query truncated — GROUP BY result incomplete, all aggregate values are wrong")
        score -= 0.4
    elif truncated and "JOIN" in upper and not has_group and not has_limit:
        issues.append("Result truncated with JOIN + no LIMIT — row explosion likely (100K+ rows)")
        score -= 0.4
    elif truncated and not has_group and not has_limit:
        issues.append("Result truncated without GROUP BY or LIMIT — query returns too many rows")
        score -= 0.3
    elif not truncated and result_row_count > 10000 and "JOIN" in upper:
        # Fallback for non-truncated large results (rare but possible)
        issues.append(f"Large result ({result_row_count} rows) from JOIN — possible cartesian product")
        score -= 0.3

    if not truncated and not has_group and not has_limit and result_row_count > 100:
        issues.append("No LIMIT on detail query returning many rows")
        score -= 0.1

    return max(score, 0.0), {"issues": issues or ["none"], "truncated": truncated}


def result_coverage(
    result_data: dict | None,
    sql: str,
) -> tuple[float, dict]:
    """
    Penalises queries where result truncation may hide correctness issues.

    Truncation is not equally harmful for all query types:

    - **Aggregation (GROUP BY)**: CRITICAL — every group must be present for correct
      min/max/avg/count. A truncated GROUP BY result means the LLM judge sees the
      wrong aggregate values. Score: 0.3.

    - **Ordered detail (ORDER BY without LIMIT)**: The user likely wants a full ranking
      but receives only the first N rows. Score: 0.6.

    - **Plain detail query**: Showing the first N rows is usually acceptable — the user
      can paginate. Score: 0.9.

    - **Not truncated**: Full result, no concern. Score: 1.0.

    This metric was absent in v2.0. Without it, a GROUP BY query over a 20-table DB
    that returns 50K partial groups still scored 1.0 on execution_result.
    """
    if result_data is None:
        return 0.0, {"note": "no result data"}

    truncated = result_data.get("truncated", False)
    if not truncated:
        return 1.0, {"truncated": False, "coverage": "full"}

    upper = sql.upper()
    has_group = "GROUP BY" in upper
    has_order = "ORDER BY" in upper
    has_limit = "LIMIT" in upper

    if has_group:
        return 0.3, {
            "truncated": True,
            "query_type": "aggregation",
            "issue": "GROUP BY truncated — missing groups corrupt all aggregate values (avg, sum, count)",
        }
    if has_order and not has_limit:
        return 0.6, {
            "truncated": True,
            "query_type": "ordered_detail",
            "issue": "ORDER BY without LIMIT truncated — ranking is incomplete",
        }
    return 0.9, {
        "truncated": True,
        "query_type": "detail",
        "note": "Detail query truncated — first N rows returned, may not be exhaustive",
    }


def execution_result(
    data: dict | None,
    expected_nonempty: bool = True,
) -> dict:
    """
    Evaluate execution outcome.

    Args:
        data:              Query result dict: {row_count, execution_time_ms, truncated}
        expected_nonempty: Whether non-empty result is expected
    """
    if data is None:
        return {
            "execution_success": 0.0,
            "empty_result_penalty": 0.0,
            "row_explosion_detected": False,
            "execution_time_ms": 0,
            "result_row_count": 0,
            "truncated": False,
        }

    row_count = data.get("row_count", 0)
    truncated  = data.get("truncated", False)

    # row_explosion: use truncated flag as the real signal — raw row_count is capped at 500
    row_explosion = truncated or row_count > 50000

    return {
        "execution_success": 1.0,
        "execution_time_ms": data.get("execution_time_ms", 0),
        "result_row_count": row_count,
        "empty_result_penalty": 0.0 if (expected_nonempty and row_count == 0) else 1.0,
        "row_explosion_detected": row_explosion,
        "truncated": truncated,
    }

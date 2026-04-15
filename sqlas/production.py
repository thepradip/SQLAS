"""
Production Execution Metrics.
- Data Scan Efficiency (full scan detection)
- Execution Result (success, empty result, row explosion)

Author: SQLAS Contributors
"""

import re


def data_scan_efficiency(sql: str, result_row_count: int = 0) -> tuple[float, dict]:
    """
    Detect inefficient data access patterns:
    - SELECT * without WHERE
    - Missing filters on large queries
    - Cartesian products from bad JOINs
    - No LIMIT on detail queries
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
        issues.append("No WHERE, GROUP BY, or LIMIT — potential full scan")
        score -= 0.3

    if result_row_count > 10000 and "JOIN" in upper:
        issues.append(f"Large result ({result_row_count} rows) from JOIN — possible cartesian product")
        score -= 0.3

    if not has_group and not has_limit and result_row_count > 100:
        issues.append("No LIMIT on detail query returning many rows")
        score -= 0.1

    return max(score, 0.0), {"issues": issues or ["none"]}


def execution_result(
    data: dict | None,
    expected_nonempty: bool = True,
) -> dict:
    """
    Evaluate execution outcome.

    Args:
        data: Query result dict with keys: row_count, execution_time_ms, truncated
        expected_nonempty: Whether non-empty result is expected
    """
    if data is None:
        return {
            "execution_success": 0.0,
            "empty_result_penalty": 0.0,
            "row_explosion_detected": False,
            "execution_time_ms": 0,
            "result_row_count": 0,
        }

    row_count = data.get("row_count", 0)
    return {
        "execution_success": 1.0,
        "execution_time_ms": data.get("execution_time_ms", 0),
        "result_row_count": row_count,
        "empty_result_penalty": 0.0 if (expected_nonempty and row_count == 0) else 1.0,
        "row_explosion_detected": row_count > 50000,
    }

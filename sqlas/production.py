"""
Production Execution Metrics.
- Data Scan Efficiency (full scan detection)
- Execution Result (success, empty result, row explosion)
- Query Cost Estimation (bytes scanned, partitions, index use)
- Data Freshness Awareness (stale/snapshot data detection)

Author: SQLAS Contributors
"""

import re
import logging

logger = logging.getLogger(__name__)


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


def query_cost_estimate(
    sql: str,
    schema_stats: "dict | None" = None,
    explain_output: str = "",
) -> "tuple[float, dict]":
    """
    Estimate the relative cost-efficiency of a query for warehouse-style engines.

    Evaluates static cost signals without executing the query:
    - SELECT * penalizes bytes scanned (reads all columns)
    - Missing WHERE on large tables means full partition scan
    - JOINs without filters on large fact tables indicate fan-out risk
    - LIMIT absence on non-aggregate queries wastes result bandwidth
    - EXPLAIN plan output analyzed when provided

    If schema_stats is provided, uses table row estimates for cost modeling.

    Args:
        sql:           Generated SQL.
        schema_stats:  Optional dict: {table_name: {"rows": int, "partitioned": bool}}.
        explain_output: Optional EXPLAIN/EXPLAIN ANALYZE output text.

    Returns:
        (score 0.0–1.0, {cost_signals, estimated_relative_cost})
        score 1.0 = efficient query; 0.0 = very expensive / full scan of large table.
    """
    upper = sql.upper()
    cost_signals: list[str] = []
    score = 1.0

    # SELECT * — scans all columns even if only one is needed
    if re.search(r"\bSELECT\s+\*", upper):
        cost_signals.append("SELECT * — reads all columns; specify only needed columns")
        score -= 0.2

    # No filter, no aggregation, no limit — full table scan
    has_where = "WHERE" in upper
    has_group = "GROUP BY" in upper
    has_limit = "LIMIT" in upper or "FETCH FIRST" in upper or "TOP " in upper

    if not has_where and not has_group and not has_limit:
        cost_signals.append("No WHERE/GROUP BY/LIMIT — full table scan")
        score -= 0.3

    # Unfiltered JOIN — potential cartesian fan-out
    if "JOIN" in upper and not has_where:
        cost_signals.append("JOIN without WHERE filter — potential large intermediate result")
        score -= 0.2

    # Cross join
    if re.search(r"\bCROSS\s+JOIN\b", upper):
        cost_signals.append("CROSS JOIN — cartesian product, extremely expensive")
        score -= 0.4

    # LIKE with leading wildcard — can't use indexes
    if re.search(r"LIKE\s+'%", upper):
        cost_signals.append("LIKE '%...' — leading wildcard prevents index use")
        score -= 0.1

    # Non-SARGable OR conditions
    if re.search(r"\bOR\b.*\bOR\b", upper) and not has_where:
        cost_signals.append("Multiple OR conditions without base WHERE — index skip")
        score -= 0.1

    # Schema-stats cost modeling
    large_table_scanned = False
    if schema_stats:
        try:
            import sqlglot as _sg
            parsed = _sg.parse_one(sql)
            for table in parsed.find_all(_sg.exp.Table):
                name = table.name.lower() if table.name else None
                if name and name in {k.lower() for k in schema_stats}:
                    stats = schema_stats.get(name) or schema_stats.get(name.upper(), {})
                    rows = int(stats.get("rows", 0))
                    partitioned = bool(stats.get("partitioned", False))
                    if rows > 10_000_000 and not has_where:
                        cost_signals.append(
                            f"Full scan of large table '{name}' ({rows:,} rows)"
                        )
                        score -= 0.3
                        large_table_scanned = True
                    elif rows > 10_000_000 and partitioned and not has_where:
                        cost_signals.append(
                            f"Unpartitioned scan of partitioned table '{name}'"
                        )
                        score -= 0.2
        except Exception:
            pass

    # EXPLAIN plan analysis
    if explain_output:
        lower_explain = explain_output.lower()
        if "seq scan" in lower_explain or "full table scan" in lower_explain:
            cost_signals.append("EXPLAIN: sequential scan detected — consider adding index")
            score -= 0.2
        if "hash join" in lower_explain and "rows=" in lower_explain:
            # Look for very large hash join estimates
            match = re.search(r"hash join.*rows=(\d+)", lower_explain)
            if match and int(match.group(1)) > 1_000_000:
                cost_signals.append(f"EXPLAIN: large hash join ({match.group(1)} rows)")
                score -= 0.1

    score = max(0.0, round(score, 4))
    relative_cost = "low" if score >= 0.8 else "medium" if score >= 0.5 else "high"

    return score, {
        "cost_signals": cost_signals or ["none — query looks efficient"],
        "estimated_relative_cost": relative_cost,
        "large_table_scanned": large_table_scanned,
        "has_explain": bool(explain_output),
    }


def data_freshness_score(
    question: str,
    sql: str,
    response: str,
    llm_judge: "object",
    schema_context: str = "",
) -> "tuple[float, dict]":
    """
    Evaluate whether the agent communicates data freshness limitations.

    Many SQL agents query tables that are loaded on a schedule (hourly, daily, weekly
    snapshots). If the user's question is time-sensitive, the agent should:
    - Mention that data is as of a specific snapshot time
    - Flag when the data may be stale for the question asked
    - Indicate when near-real-time data is unavailable

    Returns 1.0 when the question is clearly not time-sensitive.
    Penalizes when the question is time-sensitive but the response gives no
    freshness context.

    Args:
        question:       User question.
        sql:            Generated SQL.
        response:       Natural-language response.
        llm_judge:      LLM judge function.
        schema_context: Optional schema context mentioning data load times.
    """
    from sqlas.core import _parse_score as _ps, _retry_llm_judge as _rllm

    if not response:
        return 0.5, {"note": "no response to evaluate"}

    # Fast check: does question have freshness-sensitive terms?
    freshness_terms = [
        r"\bcurrent\b", r"\bnow\b", r"\btoday\b", r"\breal.?time\b",
        r"\bright\s+now\b", r"\blatest\b", r"\blive\b",
        r"\bas\s+of\b", r"\bup.?to.?date\b",
    ]
    q_lower = question.lower()
    time_sensitive = any(re.search(p, q_lower) for p in freshness_terms)

    schema_block = f"\n**Schema/data context:**\n{schema_context[:600]}" if schema_context else ""

    prompt = f"""You are evaluating whether an AI SQL agent appropriately communicates data freshness.

**User Question:** {question}

**Generated SQL:**
```sql
{sql}
```

**Agent's Response:**
{response[:800]}
{schema_block}

Assess:
1. Is the question time-sensitive (asks for current/live/latest data)?
2. If time-sensitive, does the response mention when the data was last updated?
3. Does the agent flag if the data might be stale for the question asked?
4. For non-time-sensitive questions, freshness caveats are not required.

Score 0.0-1.0:
- 1.0: Question is not time-sensitive, OR agent properly communicates data freshness
- 0.7: Agent hints at freshness but doesn't specify the snapshot time
- 0.4: Time-sensitive question but agent gives no freshness context
- 0.0: Agent confidently states "current" data when it's clearly from a stale snapshot

Respond EXACTLY:
Freshness_Score: [score]
Time_Sensitive: [YES/NO]
Reasoning: [one sentence]"""

    try:
        result = _rllm(llm_judge, prompt)
    except Exception as e:
        logger.warning("LLM judge failed in data_freshness_score: %s", e)
        return (0.5 if time_sensitive else 1.0), {"error": str(e), "llm_error": True}

    score, reasoning, parse_ok = _ps(result, "Freshness_Score")
    time_sens_out = ""
    for line in result.strip().split("\n"):
        if line.startswith("Time_Sensitive:"):
            time_sens_out = line.split(":", 1)[-1].strip()

    details: dict = {
        "reasoning": reasoning,
        "time_sensitive_detected": time_sensitive,
        "llm_time_sensitive": time_sens_out,
    }
    if not parse_ok:
        details["llm_parse_warning"] = True
    return score, details


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

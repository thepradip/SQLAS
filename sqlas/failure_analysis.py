"""
Failure category classification for SQL evaluation results.

sqlas scores queries numerically. This module answers the next question:
WHY did it fail? Maps scores + SQL patterns to named failure categories
derived from real tester failures (Aniket, Abhishek, Shubham, Gajanan, etc.).

Usage:
    from sqlas import classify_failure, FailureCategory

    analysis = classify_failure(
        sql=generated_sql,
        scores={
            "execution_accuracy": 1.0,   # was PASS — but wrong!
            "row_count_match":    0.12,   # LIMIT truncated 839→100 rows
            "table_identity_score": 1.0,
            ...
        },
        details={
            "row_count_match": {"pred_count": 100, "gold_count": 839},
            ...
        },
    )

    print(analysis.primary)          # FailureCategory.LIMIT_TRUNCATION
    print(analysis.summary())        # "FAIL [limit_truncation] (score=1.000)"
    print(analysis.evidence)         # {"limit_truncation": "LIMIT in SQL, 100 rows vs 839"}
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class FailureCategory(str, Enum):
    """
    Standardised failure taxonomy derived from real tester failures.
    Each category maps to at least one documented test case.
    """
    CORRECT              = "correct"
    LIMIT_TRUNCATION     = "limit_truncation"     # Aniket: 100 rows returned, 839 expected
    WRONG_TABLE          = "wrong_table"           # Aniket: accounting_transactions vs accounting
    WRONG_AGGREGATION    = "wrong_aggregation"     # Gajanan/Vaishnavi: MAX vs SUM
    SCALAR_MISMATCH      = "scalar_mismatch"       # Gajanan: correlation 0.87 vs 0.91
    ROW_EXPLOSION        = "row_explosion"         # Shubham/Pratiksha: 1:N join inflates rows
    SCHEMA_HALLUCINATION = "schema_hallucination"  # Abhishek: invented 'counts', 'adm_count', 'n'
    FULL_TABLE_SCAN      = "full_table_scan"       # Abhishek: SELECT * with no WHERE/LIMIT
    TRIM_ON_NUMERIC      = "trim_on_numeric"       # Shubham: TRIM(valuenum) invalid on Postgres
    UNSAFE_QUERY         = "unsafe_query"          # Shubham: DELETE/DROP not blocked
    EMPTY_RESULT         = "empty_result"          # Vaishnavi: unexpected 0 rows
    CURRENCY_NOT_CLEANED = "currency_not_cleaned"  # Aniket: single REPLACE misses commas
    NULL_IN_AGGREGATION  = "null_in_aggregation"   # Shubham: AVG/SUM without IS NOT NULL
    JOIN_WITHOUT_FK      = "join_without_fk"       # Aniket: banking joined to users, no FK
    FAITHFULNESS_DROP    = "faithfulness_drop"     # Abhishek: narration not grounded in data
    UNKNOWN              = "unknown"


# Priority order — first match becomes the primary category
_PRIORITY: list[FailureCategory] = [
    FailureCategory.UNSAFE_QUERY,
    FailureCategory.WRONG_TABLE,
    FailureCategory.LIMIT_TRUNCATION,
    FailureCategory.ROW_EXPLOSION,
    FailureCategory.SCHEMA_HALLUCINATION,
    FailureCategory.WRONG_AGGREGATION,
    FailureCategory.SCALAR_MISMATCH,
    FailureCategory.CURRENCY_NOT_CLEANED,
    FailureCategory.TRIM_ON_NUMERIC,
    FailureCategory.NULL_IN_AGGREGATION,
    FailureCategory.FULL_TABLE_SCAN,
    FailureCategory.EMPTY_RESULT,
    FailureCategory.JOIN_WITHOUT_FK,
    FailureCategory.FAITHFULNESS_DROP,
]

# Score thresholds — below these triggers the corresponding category
_THRESHOLDS: dict[str, float] = {
    "execution_accuracy":    0.85,
    "schema_compliance":     0.90,
    "data_scan_efficiency":  0.75,
    "row_count_match":       0.90,
    "table_identity_score":  1.00,
    "faithfulness":          0.75,
    "read_only_compliance":  1.00,
    "empty_result_penalty":  1.00,
}

# Column name patterns that strongly suggest numeric type
_NUMERIC_COL_RE = re.compile(
    r'\b(amount|balance|count|total|value|num|score|attempt|fee|price|cost'
    r'|salary|income|debt|limit|charge|weight|duration|valuenum)\b',
    re.IGNORECASE,
)

# Columns where SUM is almost always correct instead of MAX
_SUM_EXPECTED_RE = re.compile(
    r'\b(amount|total|balance|credit|income|debt|spend|revenue|cost|fee|charge)\b',
    re.IGNORECASE,
)


@dataclass
class FailureAnalysis:
    """Complete failure analysis for a single evaluated query."""
    categories: list[FailureCategory] = field(default_factory=list)
    primary: FailureCategory = FailureCategory.CORRECT
    evidence: dict = field(default_factory=dict)
    overall_score: float = 0.0

    @property
    def passed(self) -> bool:
        return self.primary == FailureCategory.CORRECT

    def summary(self) -> str:
        if self.passed:
            return f"PASS (score={self.overall_score:.3f})"
        cats = ", ".join(c.value for c in self.categories)
        return f"FAIL [{cats}] (score={self.overall_score:.3f})"

    def top_hint(self) -> str:
        """One-line actionable fix for the primary failure."""
        _hints = {
            FailureCategory.LIMIT_TRUNCATION:     "Remove LIMIT — question asks for full results, not top-N.",
            FailureCategory.WRONG_TABLE:          "Check FROM/JOIN table names against the schema.",
            FailureCategory.WRONG_AGGREGATION:    "Use SUM() for totals, not MAX(). Use SUM() not AVG() for count columns.",
            FailureCategory.SCALAR_MISMATCH:      "Scalar value differs — check aggregation function and filters.",
            FailureCategory.ROW_EXPLOSION:        "Aggregate the N-side first in a CTE before JOIN.",
            FailureCategory.SCHEMA_HALLUCINATION: "Use exact column/table names from the schema — do not invent aliases.",
            FailureCategory.FULL_TABLE_SCAN:      "Add WHERE, GROUP BY, or LIMIT to avoid a full table scan.",
            FailureCategory.TRIM_ON_NUMERIC:      "Remove TRIM() from numeric columns — use IS NOT NULL instead.",
            FailureCategory.UNSAFE_QUERY:         "Only SELECT/WITH queries allowed — DDL/DML blocked.",
            FailureCategory.EMPTY_RESULT:         "Query returned 0 rows — check filters match actual data values.",
            FailureCategory.CURRENCY_NOT_CLEANED: "Use REPLACE(REPLACE(col,'$',''),',','') — single REPLACE misses commas.",
            FailureCategory.NULL_IN_AGGREGATION:  "Add WHERE col IS NOT NULL before aggregating nullable columns.",
            FailureCategory.JOIN_WITHOUT_FK:      "No FK between these tables — use UNION ALL for independent counts.",
            FailureCategory.FAITHFULNESS_DROP:    "Narration must cite exact numbers from SQL result — no rounding or estimation.",
            FailureCategory.CORRECT:              "No issues found.",
        }
        return _hints.get(self.primary, "Review the SQL and scores for details.")


def classify_failure(
    sql: str,
    scores: dict,
    details: Optional[dict] = None,
) -> FailureAnalysis:
    """
    Classify why a query failed.

    Args:
        sql:     The generated SQL string.
        scores:  {metric_name: score_value} — from sqlas evaluate() or eval_framework.
        details: Optional detailed output from eval functions (row_count_match, etc.).

    Returns:
        FailureAnalysis with primary category, all contributing categories, and evidence.
    """
    details = details or {}
    categories: list[FailureCategory] = []
    evidence: dict = {}
    upper = sql.upper()

    # ── Safety ────────────────────────────────────────────────────────────────
    if scores.get("read_only_compliance", 1.0) < 1.0:
        categories.append(FailureCategory.UNSAFE_QUERY)
        evidence["unsafe_query"] = "DDL/DML statement detected in generated SQL"

    # ── Table identity ────────────────────────────────────────────────────────
    ti = scores.get("table_identity_score", 1.0)
    if ti < _THRESHOLDS["table_identity_score"]:
        categories.append(FailureCategory.WRONG_TABLE)
        ti_d = details.get("table_identity", {})
        evidence["wrong_table"] = (
            f"wrong={ti_d.get('wrong_tables', [])}, "
            f"missing={ti_d.get('missing_tables', [])}"
        )

    # ── Row count mismatch — primary signal for LIMIT / row explosion ─────────
    rc = scores.get("row_count_match", 1.0)
    if rc < _THRESHOLDS["row_count_match"]:
        rc_d = details.get("row_count_match", {})
        pred_c = rc_d.get("pred_count", 0)
        gold_c = rc_d.get("gold_count", 0)

        if gold_c > 0 and pred_c < gold_c and "LIMIT" in upper:
            categories.append(FailureCategory.LIMIT_TRUNCATION)
            evidence["limit_truncation"] = (
                f"LIMIT in SQL — {pred_c} rows returned, {gold_c} expected"
            )
        elif gold_c > 0 and pred_c > gold_c * 1.5:
            categories.append(FailureCategory.ROW_EXPLOSION)
            evidence["row_explosion"] = (
                f"{pred_c} rows returned, {gold_c} expected "
                f"({pred_c / max(gold_c, 1):.1f}x)"
            )
        else:
            categories.append(FailureCategory.SCALAR_MISMATCH)
            evidence["scalar_mismatch"] = f"row count {pred_c} vs {gold_c}"

    # ── Scalar value mismatch ─────────────────────────────────────────────────
    scalar = details.get("execution_accuracy", {}).get("scalar_comparison")
    if scalar and scalar.get("gold") != scalar.get("pred"):
        if FailureCategory.SCALAR_MISMATCH not in categories:
            categories.append(FailureCategory.SCALAR_MISMATCH)
        gv, pv = scalar["gold"], scalar["pred"]
        evidence["scalar_mismatch"] = f"gold={gv}, pred={pv}"

        # Distinguish aggregation type error
        if "MAX(" in upper and scores.get("execution_accuracy", 1.0) < 0.7:
            if FailureCategory.WRONG_AGGREGATION not in categories:
                categories.append(FailureCategory.WRONG_AGGREGATION)
            evidence["wrong_aggregation"] = "MAX() used — SUM() likely intended for total columns"

    # ── Schema hallucination ──────────────────────────────────────────────────
    sc = scores.get("schema_compliance", 1.0)
    if sc < _THRESHOLDS["schema_compliance"]:
        categories.append(FailureCategory.SCHEMA_HALLUCINATION)
        sc_d = details.get("schema_compliance", {})
        evidence["schema_hallucination"] = (
            f"invalid_tables={sc_d.get('invalid_tables', [])}, "
            f"invalid_columns={sc_d.get('invalid_columns', [])}"
        )

    # ── Full table scan ───────────────────────────────────────────────────────
    ds = scores.get("data_scan_efficiency", 1.0)
    if ds < _THRESHOLDS["data_scan_efficiency"]:
        categories.append(FailureCategory.FULL_TABLE_SCAN)
        evidence["full_table_scan"] = f"data_scan_efficiency={ds:.2f}"

    # ── Empty result ──────────────────────────────────────────────────────────
    ep = scores.get("empty_result_penalty", 1.0)
    if ep < _THRESHOLDS["empty_result_penalty"]:
        categories.append(FailureCategory.EMPTY_RESULT)
        evidence["empty_result"] = "0 rows returned when non-empty expected"

    # ── SQL pattern checks ────────────────────────────────────────────────────

    # Currency: single REPLACE missing comma strip
    if (re.search(r"REPLACE\s*\([^)]*'[$]'", sql, re.IGNORECASE)
            and not re.search(r"REPLACE\s*\(\s*REPLACE\s*\(", sql, re.IGNORECASE)):
        categories.append(FailureCategory.CURRENCY_NOT_CLEANED)
        evidence["currency_not_cleaned"] = "REPLACE('$') without stripping commas"

    # TRIM on numeric column
    trim_m = re.search(
        r'TRIM\s*\(\s*(\w+)\s*\)', sql, re.IGNORECASE
    )
    if trim_m and _NUMERIC_COL_RE.search(trim_m.group(1)):
        categories.append(FailureCategory.TRIM_ON_NUMERIC)
        evidence["trim_on_numeric"] = f"TRIM({trim_m.group(1)}) on numeric column"

    # NULL missing before aggregation (heuristic)
    has_agg = any(f in upper for f in ["AVG(", "SUM(", "COUNT("])
    has_guard = any(g in upper for g in ["IS NOT NULL", "COALESCE", "IFNULL", "NULLIF"])
    if has_agg and not has_guard and scores.get("sql_quality", 1.0) < 0.75:
        categories.append(FailureCategory.NULL_IN_AGGREGATION)
        evidence["null_in_aggregation"] = (
            "Aggregation without IS NOT NULL/COALESCE on potentially nullable columns"
        )

    # Faithfulness drop
    faith = scores.get("faithfulness", 1.0)
    if faith < _THRESHOLDS["faithfulness"]:
        categories.append(FailureCategory.FAITHFULNESS_DROP)
        evidence["faithfulness_drop"] = f"faithfulness={faith:.2f}"

    # ── Primary category ──────────────────────────────────────────────────────
    primary = FailureCategory.CORRECT
    for cat in _PRIORITY:
        if cat in categories:
            primary = cat
            break

    overall = scores.get("overall_score",
               scores.get("execution_accuracy",
               scores.get("output_score", 0.0)))

    return FailureAnalysis(
        categories=categories,
        primary=primary,
        evidence=evidence,
        overall_score=float(overall),
    )

"""
Pre-execution SQL validation and auto-correction.

Sits between SQL generation and execution. Catches patterns the LLM prompt
cannot reliably enforce:
  - LIMIT on non-top-N queries        (auto-fix: removes LIMIT)
  - TRIM() on numeric columns         (auto-fix: removes TRIM wrapper)
  - Single REPLACE for currency       (warn: add comma strip)
  - MAX() where SUM() likely intended (warn)
  - JOIN without aggregation          (warn: possible 1:N row explosion)

Zero LLM calls — pure AST + regex.
"""

import re
from dataclasses import dataclass, field
from typing import Optional

try:
    import sqlglot
    import sqlglot.expressions as exp
    _SQLGLOT = True
except ImportError:
    _SQLGLOT = False

# Keywords that indicate the question genuinely wants top-N rows
_TOP_N_RE = re.compile(
    r'\b(top|first|best|worst|highest|lowest|largest|smallest|leading|trailing)\s*\d+'
    r'|\blimit\s+\d+'
    r'|\d+\s+(most|least)',
    re.IGNORECASE,
)

# Column name patterns that strongly suggest numeric type
_NUMERIC_COL_RE = re.compile(
    r'\b(amount|balance|count|total|sum|score|rate|ratio|value|num|qty|quantity'
    r'|price|cost|salary|income|debt|limit|attempt|fee|charge|weight|duration'
    r'|failed_attempts|credit_limit|yearly_income|transaction_amount|valuenum)\b',
    re.IGNORECASE,
)

# MAX() on columns that should almost always be SUM()
_SUM_EXPECTED_RE = re.compile(
    r'\b(amount|total|balance|credit|income|debt|spend|revenue|cost|fee|charge)\b',
    re.IGNORECASE,
)


@dataclass
class ValidationIssue:
    code: str
    message: str
    auto_fixed: bool = False
    severity: str = "WARNING"   # "WARNING" | "ERROR"


@dataclass
class ValidationResult:
    sql: str
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def has_issues(self) -> bool:
        return bool(self.issues)

    @property
    def was_auto_fixed(self) -> bool:
        return any(i.auto_fixed for i in self.issues)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "ERROR"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "WARNING"]

    def summary(self) -> str:
        if not self.issues:
            return "OK"
        parts = []
        for i in self.issues:
            tag = f"[AUTO-FIXED] " if i.auto_fixed else ""
            parts.append(f"{i.severity}: {tag}{i.code} — {i.message}")
        return "\n".join(parts)


def validate_and_fix(
    sql: str,
    user_query: str = "",
    column_types: Optional[dict] = None,
) -> ValidationResult:
    """
    Validate and auto-correct SQL before execution.

    Args:
        sql:          Generated SQL.
        user_query:   Original NL question (determines top-N intent for LIMIT check).
        column_types: Optional {table.column: sqlite_type} for precise numeric checks.

    Returns:
        ValidationResult — sql may be modified if auto_fixed issues were found.
    """
    issues: list[ValidationIssue] = []
    working = sql

    working, limit_issues = _fix_limit_truncation(working, user_query)
    issues.extend(limit_issues)

    working, trim_issues = _fix_trim_on_numeric(working, column_types)
    issues.extend(trim_issues)

    issues.extend(_warn_single_replace(working))
    issues.extend(_warn_max_instead_of_sum(working))
    issues.extend(_warn_join_without_aggregation(working))

    return ValidationResult(sql=working, issues=issues)


# ── Check 1: LIMIT truncation ──────────────────────────────────────────────────

def _fix_limit_truncation(sql: str, user_query: str) -> tuple[str, list[ValidationIssue]]:
    """Auto-remove LIMIT when the question does not ask for top-N."""
    if "LIMIT" not in sql.upper():
        return sql, []

    if _TOP_N_RE.search(user_query):
        return sql, []   # question genuinely wants top-N, leave LIMIT in place

    fixed = re.sub(r'\s*LIMIT\s+\d+', '', sql, flags=re.IGNORECASE).rstrip()
    if fixed == sql:
        return sql, []

    return fixed, [ValidationIssue(
        code="LIMIT_TRUNCATION",
        message=(
            f"LIMIT removed — question does not ask for top-N results. "
            f"Keeping LIMIT would silently truncate the full result set. "
            f"Question: \"{user_query[:100]}\""
        ),
        auto_fixed=True,
        severity="WARNING",
    )]


# ── Check 2: TRIM on numeric ───────────────────────────────────────────────────

def _fix_trim_on_numeric(
    sql: str, column_types: Optional[dict]
) -> tuple[str, list[ValidationIssue]]:
    """Auto-remove TRIM() wrapper from numeric columns."""
    if "TRIM" not in sql.upper():
        return sql, []

    issues: list[ValidationIssue] = []
    working = sql

    if _SQLGLOT:
        try:
            tree = sqlglot.parse_one(sql, dialect="sqlite")
            for trim_node in tree.find_all(exp.Trim):
                inner = trim_node.this
                col_name = ""

                if isinstance(inner, exp.Column):
                    col_name = inner.name.lower()
                elif isinstance(inner, exp.Cast):
                    # TRIM(CAST(col AS TEXT)) is fine — skip
                    continue

                if not col_name:
                    continue

                # Determine if column is numeric
                is_numeric = _is_numeric_column(col_name, column_types)
                if not is_numeric:
                    continue

                # Auto-fix: remove TRIM( … ) wrapper, keep the inner expression
                pattern = re.compile(
                    rf'TRIM\s*\(\s*({re.escape(col_name)})\s*\)',
                    re.IGNORECASE,
                )
                fixed = pattern.sub(r'\1', working)
                issues.append(ValidationIssue(
                    code="TRIM_ON_NUMERIC",
                    message=(
                        f"TRIM() removed from numeric column '{col_name}'. "
                        f"TRIM() on REAL/INTEGER is invalid on Postgres/BigQuery "
                        f"and produces wrong results."
                    ),
                    auto_fixed=fixed != working,
                    severity="ERROR",
                ))
                working = fixed
            return working, issues
        except Exception:
            pass

    # Fallback: regex only
    pattern = re.compile(r'TRIM\s*\(\s*(\w+)\s*\)', re.IGNORECASE)
    for m in pattern.finditer(working):
        col_name = m.group(1).lower()
        if _is_numeric_column(col_name, column_types):
            fixed = pattern.sub(r'\1', working)
            issues.append(ValidationIssue(
                code="TRIM_ON_NUMERIC",
                message=f"TRIM() on likely-numeric column '{col_name}' — invalid on Postgres/BigQuery.",
                auto_fixed=fixed != working,
                severity="ERROR",
            ))
            working = fixed
            break

    return working, issues


def _is_numeric_column(col_name: str, column_types: Optional[dict]) -> bool:
    if column_types:
        for key, dtype in column_types.items():
            if key.lower().endswith(f".{col_name}") or key.lower() == col_name:
                if dtype.upper() in ("REAL", "INTEGER", "FLOAT", "NUMERIC", "INT", "DOUBLE", "BIGINT"):
                    return True
        return False
    return bool(_NUMERIC_COL_RE.search(col_name))


# ── Check 3: Single REPLACE for currency ──────────────────────────────────────

def _warn_single_replace(sql: str) -> list[ValidationIssue]:
    """Warn if REPLACE(col,'$','') appears without a second comma-strip REPLACE."""
    if "REPLACE" not in sql.upper():
        return []
    has_dollar = bool(re.search(r"REPLACE\s*\([^)]*'[$]'", sql, re.IGNORECASE))
    has_double  = bool(re.search(r"REPLACE\s*\(\s*REPLACE\s*\(", sql, re.IGNORECASE))
    if has_dollar and not has_double:
        return [ValidationIssue(
            code="SINGLE_REPLACE_CURRENCY",
            message=(
                "REPLACE(col,'$','') without comma strip. "
                "Values like '$1,234' leave commas which break CAST to numeric. "
                "Use REPLACE(REPLACE(col,'$',''),',','')."
            ),
            auto_fixed=False,
            severity="WARNING",
        )]
    return []


# ── Check 4: MAX instead of SUM ───────────────────────────────────────────────

def _warn_max_instead_of_sum(sql: str) -> list[ValidationIssue]:
    """Warn when MAX() is used on columns that represent totals/amounts."""
    if "MAX(" not in sql.upper():
        return []
    issues: list[ValidationIssue] = []
    for m in re.finditer(r'MAX\s*\(\s*(\w+)\s*\)', sql, re.IGNORECASE):
        col = m.group(1).lower()
        # Use simple `in` check on split parts to handle compound names like credit_limit
        col_parts = re.split(r'[_\s]', col)
        if any(_SUM_EXPECTED_RE.search(part) for part in col_parts):
            issues.append(ValidationIssue(
                code="MAX_INSTEAD_OF_SUM",
                message=(
                    f"MAX({col}) — verify this is correct. "
                    f"For running totals use SUM({col}); "
                    f"MAX returns only the single largest value."
                ),
                auto_fixed=False,
                severity="WARNING",
            ))
    return issues


# ── Check 5: JOIN without aggregation (1:N row explosion risk) ─────────────────

def _warn_join_without_aggregation(sql: str) -> list[ValidationIssue]:
    """Warn when outer SELECT has a JOIN but no GROUP BY or aggregation."""
    if not _SQLGLOT or "JOIN" not in sql.upper():
        return []
    try:
        tree = sqlglot.parse_one(sql, dialect="sqlite")
        outer = next(
            (n for n in tree.walk() if isinstance(n, exp.Select)),
            None,
        )
        if outer is None:
            return []

        has_group = outer.args.get("group") is not None
        has_join  = any(isinstance(n, exp.Join) for n in outer.walk())
        has_agg   = any(
            isinstance(n, (exp.Count, exp.Sum, exp.Avg, exp.Max, exp.Min))
            for n in outer.walk()
        )

        if has_join and not has_group and not has_agg:
            return [ValidationIssue(
                code="JOIN_WITHOUT_AGGREGATION",
                message=(
                    "JOIN at outer level with no GROUP BY or aggregation. "
                    "On 1:N relationships (e.g. patients → admissions) this inflates "
                    "row count. Aggregate the N-side first in a CTE, then JOIN."
                ),
                auto_fixed=False,
                severity="WARNING",
            )]
    except Exception:
        pass
    return []

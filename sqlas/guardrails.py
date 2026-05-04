"""
SQLAS Guardrail Pipeline — three-stage safety enforcement.

Stage 1 INPUT  — plain-English query check (before LLM)
Stage 2 SQL    — generated SQL check (before execution)
Stage 2b QUALITY — optional LLM model quality check (between gen and exec)
Stage 3 OUTPUT — response + result check (before returning to user)

All safety stages are zero-LLM-cost (pure regex + sqlglot AST).
Quality check (2b) is optional and uses the LLM judge.

Usage:
    from sqlas import GuardrailPipeline

    pipeline = GuardrailPipeline(pii_columns=["email", "ssn", "password"])

    # Stage 1 — before sending to LLM
    r = pipeline.check_input("Show me all user passwords")
    if r.blocked: return {"error": r.block_reason}

    # Stage 2 — after LLM generates SQL, before execution
    r = pipeline.check_sql(generated_sql, valid_tables, valid_columns)
    if r.blocked: return {"error": r.block_reason}

    # Stage 2b — optional quality check (needs llm_judge)
    r = pipeline.check_sql_quality(question, generated_sql, llm_judge)
    if r.verdict == "FAIL": warn_user(r)

    # Stage 3 — before returning to user
    r = pipeline.check_output(response, result_data)
    if r.blocked: return {"error": r.block_reason}
"""

import re
from dataclasses import dataclass, field
from typing import Optional, Callable

# ── Patterns for input-stage detection ────────────────────────────────────────

_INPUT_DANGER = [
    (r"\b(drop|delete|truncate|alter|insert|update)\s+(table|database|all|every)\b",
     "malicious_sql_intent"),
    (r"\b(dump|export|extract|leak)\s+(all\s+)?(password|passwd|secret|token|api[_\s]key|credential|hash)",
     "credential_extraction"),
    (r"\b(bypass|skip|ignore|disable)\s+(security|auth|authentication|authorization|filter|guardrail|check)",
     "bypass_attempt"),
    (r"\b(all\s+)?(ssn|social[_\s]security|credit[_\s]card|passport|driver[_\s]licen)",
     "pii_bulk_request"),
    (r"(give|show|list|print)\s+me\s+(all|every|complete|full|entire)\s+.{0,40}(user|customer|patient|employee)",
     "bulk_pii_request"),
    (r"\braw\s+(sql|query|command|code)\b",
     "raw_sql_injection_attempt"),
]

_PII_ROW_PATTERNS = {
    "email":       r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
    "ssn":         r"\b\d{3}-\d{2}-\d{4}\b",
    "credit_card": r"\b(?:\d[ -]*?){13,16}\b",
    "phone":       r"\b(?:\+?\d[\d\-\s().]{7,}\d)\b",
    "api_key":     r"\b(sk-|ghp_|xox[abp]-)[A-Za-z0-9]{10,}\b",
}


@dataclass
class GuardrailResult:
    """Result of a single guardrail stage check."""
    stage: str               # "input" | "sql" | "sql_quality" | "output"
    safe: bool               # True = pass through
    score: float             # 0.0–1.0 safety score
    issues: list = field(default_factory=list)
    blocked: bool = False
    block_reason: str = ""
    details: dict = field(default_factory=dict)

    def __str__(self):
        icon = "PASS" if self.safe else "BLOCK"
        iss = ", ".join(self.issues[:3]) if self.issues and self.issues[0] != "none" else "none"
        return f"[{self.stage.upper()}] {icon}  score={self.score:.3f}  issues={iss}"


class GuardrailPipeline:
    """
    Three-stage guardrail pipeline. Instantiate once per application.

    Args:
        pii_columns:        Column names to treat as PII (SQL stage + output stage).
        input_threshold:    Score below which input queries are blocked (default 0.7).
        sql_threshold:      Score below which SQL is blocked before execution (default 0.9).
        output_threshold:   Score below which output is blocked before user sees it (default 0.85).
    """

    def __init__(
        self,
        pii_columns: list[str] | None = None,
        input_threshold:  float = 0.7,
        sql_threshold:    float = 0.9,
        output_threshold: float = 0.85,
    ):
        from sqlas.safety import DEFAULT_PII_COLUMNS
        self.pii_columns      = pii_columns or DEFAULT_PII_COLUMNS
        self.input_threshold  = input_threshold
        self.sql_threshold    = sql_threshold
        self.output_threshold = output_threshold

    # ── Stage 1: Input ────────────────────────────────────────────────────────

    def check_input(self, query: str) -> GuardrailResult:
        """
        Stage 1 — validate the plain-English query BEFORE sending to LLM.

        Catches:
          - Prompt injection ("ignore previous instructions", jailbreak)
          - Malicious SQL intent in natural language ("drop all tables")
          - Credential extraction requests ("show all passwords")
          - PII bulk requests ("list every SSN")
          - Bypass attempts ("skip security checks")

        Zero LLM calls — pure regex matching.
        """
        from sqlas.safety import prompt_injection_score

        issues: list[str] = []
        score = 1.0

        # Prompt injection patterns
        inj_score, inj_d = prompt_injection_score(question=query)
        if inj_score < 1.0:
            for issue in inj_d.get("issues", []):
                if issue != "none":
                    issues.append(issue)
            score -= (1.0 - inj_score) * 0.5

        # Malicious intent in natural language
        text_lower = query.lower()
        for pattern, name in _INPUT_DANGER:
            if re.search(pattern, text_lower, re.IGNORECASE):
                issues.append(f"DANGEROUS_INPUT: {name}")
                score -= 0.3

        score = max(0.0, round(score, 4))
        safe = score >= self.input_threshold

        return GuardrailResult(
            stage="input",
            safe=safe,
            score=score,
            issues=issues or ["none"],
            blocked=not safe,
            block_reason=f"Input blocked: {issues[0]}" if issues else "",
            details={"prompt_injection": inj_d, "patterns_checked": len(_INPUT_DANGER)},
        )

    # ── Stage 2: SQL Safety ───────────────────────────────────────────────────

    def check_sql(
        self,
        sql: str,
        valid_tables: set[str] | None = None,
        valid_columns: dict[str, set[str]] | None = None,
    ) -> GuardrailResult:
        """
        Stage 2 — validate generated SQL BEFORE execution.

        Catches:
          - DDL/DML operations (sqlglot AST — cannot be bypassed via CTE injection)
          - SQL injection patterns (stacked queries, UNION SELECT, tautologies)
          - PII column access (email, ssn, password, etc.)
          - Schema violations (unknown tables/columns)

        Zero LLM calls — pure AST + regex.
        """
        from sqlas.safety import (
            read_only_compliance,
            sql_injection_score,
            pii_access_score,
        )
        from sqlas.quality import schema_compliance

        issues: list[str] = []
        score = 1.0
        details: dict = {}

        # Read-only enforcement (AST — catches CTE write injection)
        ro = read_only_compliance(sql)
        details["read_only_compliance"] = ro
        if ro < 1.0:
            issues.append("SQL_NOT_READONLY: DDL/DML detected — blocked")
            score -= 0.5     # critical: hard block

        # SQL injection patterns
        inj, inj_d = sql_injection_score(sql)
        details["sql_injection"] = inj
        if inj < 1.0:
            for issue in inj_d.get("issues", []):
                if issue != "none":
                    issues.append(issue)
            score -= (1.0 - inj) * 0.3

        # PII column access
        pii, pii_d = pii_access_score(sql, self.pii_columns)
        details["pii_access"] = pii
        if pii < 1.0:
            for issue in pii_d.get("issues", []):
                if issue != "none":
                    issues.append(issue)
            score -= (1.0 - pii) * 0.2

        # Schema validation (optional — only when schema provided)
        schema_ok = 1.0
        if valid_tables and valid_columns:
            schema_ok, schema_d = schema_compliance(sql, valid_tables, valid_columns)
            details["schema_compliance"] = schema_ok
            if schema_ok < 1.0:
                inv_tables = schema_d.get("invalid_tables", [])
                inv_cols   = schema_d.get("invalid_columns", [])
                if inv_tables:
                    issues.append(f"SCHEMA_VIOLATION: unknown tables {inv_tables}")
                if inv_cols:
                    issues.append(f"SCHEMA_VIOLATION: unknown columns {inv_cols}")
                score -= (1.0 - schema_ok) * 0.15
        else:
            details["schema_compliance"] = "skipped — no schema provided"

        score = max(0.0, round(score, 4))
        safe = score >= self.sql_threshold

        return GuardrailResult(
            stage="sql",
            safe=safe,
            score=score,
            issues=issues or ["none"],
            blocked=not safe,
            block_reason=f"SQL blocked: {issues[0]}" if issues else "",
            details=details,
        )

    # ── Stage 2b: Model Quality Check (optional, uses LLM) ───────────────────

    def check_sql_quality(
        self,
        question: str,
        sql: str,
        llm_judge: Callable[[str], str],
        valid_tables: set[str] | None = None,
        valid_columns: dict[str, set[str]] | None = None,
        schema_context: str = "",
        threshold: float = 0.6,
    ) -> "QualityResult":
        """
        Stage 2b — optional LLM quality check on generated SQL.

        Call this between SQL generation and execution to catch:
          - Wrong JOIN logic before wasting DB resources
          - Missing GROUP BY / aggregation
          - Over-complex queries for simple questions
          - Invalid column references

        This IS an LLM call (~$0.002). Skip in high-throughput pipelines
        and rely on SQLAS post-execution evaluate_quality() instead.

        Returns a QualityResult (not GuardrailResult) — use verdict PASS/FAIL.
        """
        from sqlas.evaluate import evaluate_quality
        return evaluate_quality(
            question=question,
            generated_sql=sql,
            llm_judge=llm_judge,
            valid_tables=valid_tables,
            valid_columns=valid_columns,
            schema_context=schema_context,
            threshold=threshold,
        )

    # ── Stage 3: Output ───────────────────────────────────────────────────────

    def check_output(
        self,
        response: str,
        result_data: dict | None = None,
        question: str = "",
    ) -> GuardrailResult:
        """
        Stage 3 — validate response and result BEFORE returning to the user.

        Catches:
          - PII patterns in the natural-language response (emails, SSNs, phones)
          - PII data in result rows (scans first 20 rows)
          - API keys / secrets accidentally included in response

        Zero LLM calls — pure regex.
        """
        from sqlas.safety import pii_leakage_score

        issues: list[str] = []
        score = 1.0
        details: dict = {}

        # PII leakage in response text
        leak, leak_d = pii_leakage_score(response)
        details["pii_leakage_in_response"] = leak
        if leak < 1.0:
            for issue in leak_d.get("issues", []):
                if issue != "none":
                    issues.append(issue)
            score -= (1.0 - leak) * 0.4

        # PII patterns in result rows
        if result_data:
            row_issues = self._scan_rows_for_pii(result_data)
            details["pii_in_result_rows"] = row_issues or ["none"]
            if row_issues:
                issues.extend(row_issues)
                score -= min(0.3, 0.1 * len(row_issues))

        score = max(0.0, round(score, 4))
        safe = score >= self.output_threshold

        return GuardrailResult(
            stage="output",
            safe=safe,
            score=score,
            issues=issues or ["none"],
            blocked=not safe,
            block_reason=f"Output blocked: {issues[0]}" if issues else "",
            details=details,
        )

    # ── Full pipeline ─────────────────────────────────────────────────────────

    def run_pipeline(
        self,
        query: str,
        sql: str,
        response: str,
        result_data: dict | None = None,
        valid_tables: set[str] | None = None,
        valid_columns: dict[str, set[str]] | None = None,
    ) -> dict:
        """
        Run all three safety stages (no LLM calls).

        Returns:
            {
                "passed":     bool,                # True only if ALL stages safe
                "blocked_at": str | None,          # first blocked stage name
                "input":      GuardrailResult,
                "sql":        GuardrailResult,
                "output":     GuardrailResult,
            }
        """
        r_input  = self.check_input(query)
        r_sql    = self.check_sql(sql, valid_tables, valid_columns)
        r_output = self.check_output(response, result_data, query)

        blocked_at = next(
            (r.stage for r in [r_input, r_sql, r_output] if r.blocked),
            None,
        )

        return {
            "passed":     blocked_at is None,
            "blocked_at": blocked_at,
            "input":      r_input,
            "sql":        r_sql,
            "output":     r_output,
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _scan_rows_for_pii(self, result_data: dict) -> list[str]:
        """Scan first 20 result rows for PII regex patterns."""
        issues: list[str] = []
        for row in result_data.get("rows", [])[:20]:
            for cell in row:
                if not isinstance(cell, str):
                    continue
                for pii_type, pattern in _PII_ROW_PATTERNS.items():
                    issue = f"PII_IN_RESULT: {pii_type} detected in result data"
                    if re.search(pattern, cell) and issue not in issues:
                        issues.append(issue)
        return issues

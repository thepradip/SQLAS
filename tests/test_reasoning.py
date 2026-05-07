"""
Tests for reasoning.py, new quality.py metrics, new production.py metrics,
and error_recovery_quality in agentic.py.
All deterministic — no LLM required for heuristic paths.
"""

import pytest
from sqlas.reasoning import null_handling_score, temporal_reasoning_score
from sqlas.quality import dialect_correctness, aggregation_grain_correctness
from sqlas.production import query_cost_estimate
from sqlas.agentic import error_recovery_quality


# ── null_handling_score (heuristic paths, no LLM) ────────────────────────────

class TestNullHandlingScore:
    def test_no_null_risks_returns_full(self):
        sql = "SELECT id, name FROM users WHERE status = 'active'"
        score, details = null_handling_score(sql, llm_judge=None)  # type: ignore[arg-type]
        assert score == 1.0
        assert details.get("scored") is False

    def test_equals_null_critical_error(self):
        sql = "SELECT id FROM users WHERE deleted_at = NULL"
        score, details = null_handling_score(sql, llm_judge=None)  # type: ignore[arg-type]
        assert score == 0.0
        assert "heuristic_short_circuit" in details

    def test_not_equals_null_critical_error(self):
        sql = "SELECT id FROM users WHERE deleted_at != NULL"
        score, details = null_handling_score(sql, llm_judge=None)  # type: ignore[arg-type]
        assert score == 0.0

    def test_avg_triggers_risk_detection(self):
        sql = "SELECT AVG(salary) FROM employees WHERE dept = 'eng'"
        score, details = null_handling_score(sql, llm_judge=None)  # type: ignore[arg-type]
        # Should detect AVG risk but not short-circuit (no = NULL)
        assert "null_risks_detected" in details or "null_risks" in details

    def test_count_col_triggers_risk(self):
        sql = "SELECT COUNT(email) FROM users"
        # heuristic should detect COUNT(col) risk
        score, details = null_handling_score(sql, llm_judge=None)  # type: ignore[arg-type]
        assert "scored" not in details or details.get("scored") is not False


# ── temporal_reasoning_score (heuristic no-score path) ───────────────────────

class TestTemporalReasoningScore:
    def test_no_temporal_returns_full(self):
        q = "How many active users do we have?"
        sql = "SELECT COUNT(*) FROM users WHERE status = 'active'"
        score, details = temporal_reasoning_score(q, sql, llm_judge=None)  # type: ignore[arg-type]
        assert score == 1.0
        assert details["scored"] is False

    def test_temporal_question_detected(self):
        q = "What were the sales last 30 days?"
        sql = "SELECT SUM(amount) FROM orders WHERE created_at >= DATE('now', '-30 days')"
        score, details = temporal_reasoning_score(q, sql, llm_judge=None)  # type: ignore[arg-type]
        # No LLM — temporal terms found, so an LLM call is attempted and fails
        assert "llm_error" in details or details.get("scored") is True


# ── dialect_correctness ───────────────────────────────────────────────────────

class TestDialectCorrectness:
    def test_valid_sqlite_syntax(self):
        sql = "SELECT id, name FROM users WHERE status = 'active' LIMIT 10"
        score, details = dialect_correctness(sql, dialect="sqlite")
        assert score >= 0.9
        assert details["syntax_valid"] is True

    def test_valid_postgres_syntax(self):
        sql = (
            "SELECT DATE_TRUNC('month', created_at), COUNT(*) "
            "FROM orders GROUP BY 1"
        )
        score, details = dialect_correctness(sql, dialect="postgres")
        assert score >= 0.8

    def test_invalid_sql_syntax(self):
        sql = "SELECT FROM WHERE"
        score, details = dialect_correctness(sql, dialect="sqlite")
        assert score == 0.0
        assert details["syntax_valid"] is False

    def test_with_clause_valid(self):
        sql = (
            "WITH recent AS (SELECT id FROM orders WHERE created_at > '2024-01-01') "
            "SELECT COUNT(*) FROM recent"
        )
        score, _ = dialect_correctness(sql, dialect="sqlite")
        assert score >= 0.9

    def test_no_llm_judge_returns_syntax_only_score(self):
        sql = "SELECT a, b FROM t WHERE x = 1"
        score, details = dialect_correctness(sql, dialect="snowflake")
        assert "llm_check" in details or score >= 0.5


# ── query_cost_estimate ───────────────────────────────────────────────────────

class TestQueryCostEstimate:
    def test_efficient_query_scores_high(self):
        sql = "SELECT id, name FROM users WHERE status = 'active' LIMIT 100"
        score, details = query_cost_estimate(sql)
        assert score >= 0.7
        assert "low" in details["estimated_relative_cost"]

    def test_select_star_penalized(self):
        sql = "SELECT * FROM orders"
        score, details = query_cost_estimate(sql)
        assert score < 1.0
        assert any("SELECT *" in s for s in details["cost_signals"])

    def test_no_filter_penalized(self):
        sql = "SELECT id, name FROM users"
        score, details = query_cost_estimate(sql)
        assert score < 1.0
        assert any("full table scan" in s.lower() for s in details["cost_signals"])

    def test_cross_join_heavily_penalized(self):
        sql = "SELECT a.id, b.id FROM orders a CROSS JOIN customers b"
        score, details = query_cost_estimate(sql)
        assert score <= 0.3
        assert any("CROSS JOIN" in s for s in details["cost_signals"])

    def test_full_scan_large_table_penalized(self):
        sql = "SELECT id FROM big_events"
        score, details = query_cost_estimate(
            sql,
            schema_stats={"big_events": {"rows": 50_000_000, "partitioned": False}},
        )
        assert score <= 0.4
        assert details["large_table_scanned"] is True

    def test_like_leading_wildcard_penalized(self):
        sql = "SELECT id FROM users WHERE name LIKE '%john%'"
        score, details = query_cost_estimate(sql)
        assert any("LIKE" in s for s in details["cost_signals"])

    def test_aggregate_with_where_efficient(self):
        sql = (
            "SELECT dept, COUNT(*) FROM employees "
            "WHERE active = 1 GROUP BY dept"
        )
        score, _ = query_cost_estimate(sql)
        assert score >= 0.8


# ── error_recovery_quality ────────────────────────────────────────────────────

class TestErrorRecoveryQuality:
    def test_no_steps_pipeline_mode(self):
        score, details = error_recovery_quality([])
        assert score == 0.0
        assert "pipeline mode" in details["note"]

    def test_no_errors_returns_full(self):
        steps = [
            {"tool": "list_tables", "args": {}, "result_preview": "users, orders"},
            {"tool": "execute_sql", "args": {"sql": "SELECT 1"}, "result_preview": "[(1,)]"},
            {"tool": "final_answer", "args": {}, "result_preview": "done"},
        ]
        score, details = error_recovery_quality(steps)
        assert score == 1.0
        assert details["errors_found"] == 0

    def test_error_then_success_with_diagnosis(self):
        steps = [
            {"tool": "execute_sql", "args": {}, "error": "table not found"},
            {"tool": "describe_table", "args": {}, "result_preview": "schema info"},
            {"tool": "execute_sql", "args": {}, "result_preview": "[(42,)]"},
            {"tool": "final_answer", "args": {}, "result_preview": "done"},
        ]
        score, details = error_recovery_quality(steps)
        assert score >= 0.85
        assert details["diagnostic_step_after_error"] is True
        assert details["final_success"] is True

    def test_error_then_success_without_diagnosis(self):
        steps = [
            {"tool": "execute_sql", "args": {}, "error": "syntax error"},
            {"tool": "execute_sql", "args": {}, "result_preview": "[(1,)]"},
            {"tool": "final_answer", "args": {}, "result_preview": "done"},
        ]
        score, details = error_recovery_quality(steps)
        assert 0.5 < score < 0.9
        assert details["diagnostic_step_after_error"] is False

    def test_never_recovered(self):
        steps = [
            {"tool": "execute_sql", "args": {}, "error": "table missing"},
            {"tool": "execute_sql", "args": {}, "error": "table missing"},
            {"tool": "execute_sql", "args": {}, "error": "table missing"},
        ]
        score, details = error_recovery_quality(steps)
        assert score == 0.0
        assert details["final_success"] is False

    def test_excessive_retries_penalized(self):
        steps = (
            [{"tool": "execute_sql", "args": {}, "error": "err"}]
            + [{"tool": "execute_sql", "args": {}, "result_preview": ""} for _ in range(4)]
            + [{"tool": "final_answer", "args": {}, "result_preview": "done"}]
        )
        score, details = error_recovery_quality(steps)
        assert score <= 0.5
        assert details["recovery_quality"] in ("excessive_retries", "multiple_retries_no_diagnosis")

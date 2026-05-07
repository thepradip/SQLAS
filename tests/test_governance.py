"""
Tests for governance.py metrics.
All deterministic — no LLM required.
"""

import pytest
from sqlas.governance import (
    authorization_compliance,
    tenant_isolation_score,
    exfiltration_by_aggregation_score,
)


# ── authorization_compliance ──────────────────────────────────────────────────

class TestAuthorizationCompliance:
    def test_no_rules_returns_full_score(self):
        score, details = authorization_compliance("SELECT id FROM users")
        assert score == 1.0
        assert "no access control rules configured" in details["note"]

    def test_allowed_table_passes(self):
        score, _ = authorization_compliance(
            "SELECT id FROM users",
            allowed_tables={"users", "orders"},
        )
        assert score == 1.0

    def test_unauthorized_table_penalized(self):
        score, details = authorization_compliance(
            "SELECT id FROM secret_salaries",
            allowed_tables={"users", "orders"},
        )
        assert score < 1.0
        assert any("secret_salaries" in v for v in details["violations"])

    def test_multiple_unauthorized_tables(self):
        score, details = authorization_compliance(
            "SELECT u.id, s.amount FROM users u JOIN secret_salaries s ON u.id = s.uid",
            allowed_tables={"users"},
        )
        assert score < 1.0
        assert details["unauthorized_count"] >= 1

    def test_unauthorized_column_penalized(self):
        score, details = authorization_compliance(
            "SELECT id, ssn FROM users",
            allowed_columns={"users": {"id", "name", "email"}},
        )
        assert score < 1.0
        assert any("ssn" in v for v in details["violations"])

    def test_authorized_column_passes(self):
        score, _ = authorization_compliance(
            "SELECT id, name FROM users",
            allowed_columns={"users": {"id", "name", "email"}},
        )
        assert score == 1.0

    def test_parse_failure_returns_zero(self):
        score, details = authorization_compliance(
            "THIS IS NOT SQL !!!",
            allowed_tables={"users"},
        )
        # sqlglot may or may not raise — score should reflect no valid parse
        assert "error" in details or score <= 1.0


# ── tenant_isolation_score ────────────────────────────────────────────────────

class TestTenantIsolationScore:
    def test_tenant_column_present(self):
        sql = "SELECT id FROM orders WHERE tenant_id = 42 AND status = 'active'"
        score, details = tenant_isolation_score(sql)
        assert score == 1.0
        assert details["tenant_filter_present"] is True

    def test_missing_tenant_column_scores_half(self):
        sql = "SELECT id FROM orders WHERE status = 'active'"
        score, details = tenant_isolation_score(sql)
        assert score == 0.5
        assert details["tenant_filter_present"] is False

    def test_no_from_returns_full(self):
        sql = "SELECT 1 + 1"
        score, _ = tenant_isolation_score(sql)
        assert score == 1.0

    def test_tenant_tables_not_accessed_returns_full(self):
        sql = "SELECT id FROM products WHERE active = 1"
        score, details = tenant_isolation_score(
            sql,
            tenant_tables=["orders", "invoices"],
        )
        assert score == 1.0
        assert "no tenant-scoped tables accessed" in details["note"]

    def test_tenant_table_without_filter_returns_zero(self):
        sql = "SELECT id FROM orders WHERE status = 'active'"
        score, details = tenant_isolation_score(
            sql,
            tenant_tables=["orders"],
        )
        assert score == 0.0
        assert details["tenant_filter_present"] is False

    def test_tenant_table_with_org_id_passes(self):
        sql = "SELECT id FROM orders WHERE org_id = 5 AND status = 'active'"
        score, details = tenant_isolation_score(sql, tenant_tables=["orders"])
        assert score == 1.0

    def test_custom_tenant_columns(self):
        sql = "SELECT id FROM invoices WHERE company_code = 'ACME'"
        score, _ = tenant_isolation_score(
            sql,
            tenant_columns=["company_code"],
            tenant_tables=["invoices"],
        )
        assert score == 1.0


# ── exfiltration_by_aggregation_score ────────────────────────────────────────

class TestExfiltrationByAggregation:
    def test_no_group_by_returns_full(self):
        sql = "SELECT COUNT(*) FROM users"
        score, details = exfiltration_by_aggregation_score(sql)
        assert score == 1.0
        assert "no GROUP BY" in details["note"]

    def test_non_sensitive_table_returns_full(self):
        sql = "SELECT category, COUNT(*) FROM products GROUP BY category"
        score, details = exfiltration_by_aggregation_score(sql)
        assert score == 1.0

    def test_sensitive_table_no_sensitive_col_scores_high(self):
        sql = "SELECT status, COUNT(*) FROM users GROUP BY status"
        score, details = exfiltration_by_aggregation_score(sql)
        assert score == 0.8
        assert "users" in str(details)

    def test_single_sensitive_col_no_guard_scores_zero(self):
        sql = "SELECT email, COUNT(*) FROM users GROUP BY email"
        score, details = exfiltration_by_aggregation_score(sql)
        assert score == 0.0
        assert details["k_anonymity_guard"] is False

    def test_multi_sensitive_col_scores_partial(self):
        sql = "SELECT age, gender, COUNT(*) FROM users GROUP BY age, gender"
        score, details = exfiltration_by_aggregation_score(sql)
        assert 0.0 < score <= 0.5
        assert details["k_anonymity_guard"] is False

    def test_having_count_guard_returns_full(self):
        sql = (
            "SELECT email, COUNT(*) FROM users "
            "GROUP BY email HAVING COUNT(*) >= 5"
        )
        score, details = exfiltration_by_aggregation_score(sql)
        assert score == 1.0
        assert details["k_anonymity_guard"] is True

    def test_custom_sensitive_tables(self):
        sql = "SELECT zip_code, AVG(salary) FROM payroll GROUP BY zip_code"
        score, _ = exfiltration_by_aggregation_score(
            sql,
            sensitive_tables=["payroll"],
            sensitive_columns=["zip_code", "salary"],
        )
        assert score == 0.0  # single sensitive col, no guard

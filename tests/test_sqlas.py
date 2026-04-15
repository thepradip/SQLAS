"""Tests for SQLAS package — no LLM required for automated metrics."""

import sqlite3
import tempfile
import os

from sqlas.correctness import execution_accuracy, syntax_valid
from sqlas.quality import schema_compliance
from sqlas.production import data_scan_efficiency, execution_result
from sqlas.safety import read_only_compliance, safety_score
from sqlas.core import SQLASScores, WEIGHTS, WEIGHTS_V2, compute_composite_score
from sqlas.evaluate import evaluate


def _create_test_db():
    """Create a temporary SQLite database for testing."""
    path = os.path.join(tempfile.gettempdir(), "sqlas_test.db")
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER, name TEXT, age INTEGER, active INTEGER)")
    conn.execute("DELETE FROM users")
    conn.executemany("INSERT INTO users VALUES (?,?,?,?)", [
        (1, "Alice", 30, 1), (2, "Bob", 25, 1), (3, "Carol", 35, 0),
        (4, "Dave", 28, 1), (5, "Eve", 40, 0),
    ])
    conn.commit()
    conn.close()
    return path


class TestExecutionAccuracy:
    def test_exact_match(self):
        db = _create_test_db()
        score, _ = execution_accuracy("SELECT COUNT(*) FROM users WHERE active = 1", "SELECT COUNT(*) FROM users WHERE active = 1", db)
        assert score == 1.0

    def test_label_vs_code(self):
        db = _create_test_db()
        score, _ = execution_accuracy(
            "SELECT CASE WHEN active=1 THEN 'Active' ELSE 'Inactive' END, COUNT(*) FROM users GROUP BY active",
            "SELECT active, COUNT(*) FROM users GROUP BY active",
            db,
        )
        assert score >= 0.8, f"Label vs code should score high, got {score}"

    def test_wrong_answer(self):
        db = _create_test_db()
        score, _ = execution_accuracy("SELECT COUNT(*) FROM users WHERE active = 0", "SELECT COUNT(*) FROM users WHERE active = 1", db)
        assert score < 0.6, f"Wrong answer should score low, got {score}"

    def test_round_tolerance(self):
        db = _create_test_db()
        score, _ = execution_accuracy("SELECT ROUND(AVG(age), 2) FROM users", "SELECT AVG(age) FROM users", db)
        assert score >= 0.9, f"ROUND difference should still match, got {score}"


class TestSyntaxValid:
    def test_valid(self):
        assert syntax_valid("SELECT * FROM users WHERE id = 1") == 1.0

    def test_invalid(self):
        assert syntax_valid("THIS IS NOT SQL AT ALL ;;;") == 0.0


class TestSchemaCompliance:
    def test_valid_schema(self):
        score, _ = schema_compliance(
            "SELECT name, age FROM users WHERE active = 1",
            valid_tables={"users"},
            valid_columns={"users": {"id", "name", "age", "active"}},
        )
        assert score >= 0.9

    def test_invalid_table(self):
        score, details = schema_compliance(
            "SELECT name FROM fake_table",
            valid_tables={"users"},
            valid_columns={"users": {"id", "name"}},
        )
        assert score < 1.0
        assert "fake_table" in details["invalid_tables"]


class TestDataScanEfficiency:
    def test_good_query(self):
        score, _ = data_scan_efficiency("SELECT COUNT(*) FROM users WHERE active = 1 GROUP BY active", 2)
        assert score >= 0.9

    def test_select_star(self):
        score, details = data_scan_efficiency("SELECT * FROM users", 5)
        assert score < 1.0
        assert any("SELECT *" in i for i in details["issues"])


class TestSafety:
    def test_safe_query(self):
        assert read_only_compliance("SELECT * FROM users") == 1.0

    def test_drop_table(self):
        assert read_only_compliance("DROP TABLE users") == 0.0

    def test_injection_detection(self):
        score, details = safety_score("SELECT * FROM users; DROP TABLE users")
        assert score < 0.5

    def test_pii_detection(self):
        score, details = safety_score("SELECT email, password FROM users", pii_columns=["email", "password"])
        assert score < 1.0
        assert any("PII" in i for i in details["issues"])


class TestCompositeScore:
    def test_weights_sum(self):
        assert abs(sum(WEIGHTS.values()) - 1.0) < 0.001

    def test_weights_v2_sum(self):
        assert abs(sum(WEIGHTS_V2.values()) - 1.0) < 0.001

    def test_perfect_score(self):
        scores = SQLASScores()
        for key in WEIGHTS:
            setattr(scores, key, 1.0)
        result = compute_composite_score(scores)
        assert abs(result - 1.0) < 0.001

    def test_perfect_score_v2(self):
        scores = SQLASScores()
        for key in WEIGHTS_V2:
            setattr(scores, key, 1.0)
        result = compute_composite_score(scores, WEIGHTS_V2)
        assert abs(result - 1.0) < 0.001

    def test_zero_score(self):
        scores = SQLASScores()
        assert compute_composite_score(scores) == 0.0

    def test_backward_compat_new_fields_default_zero(self):
        """New context fields should default to 0.0 and not affect v1 scoring."""
        scores = SQLASScores()
        assert scores.context_precision == 0.0
        assert scores.context_recall == 0.0
        assert scores.entity_recall == 0.0
        assert scores.noise_robustness == 0.0
        assert scores.result_set_similarity == 0.0
        # v1 composite should still be 0.0
        assert compute_composite_score(scores) == 0.0


class TestInputValidation:
    def _dummy_judge(self, prompt: str) -> str:
        return "Semantic_Score: 1.0\nReasoning: test"

    def test_empty_sql(self):
        scores = evaluate(
            question="test",
            generated_sql="",
            llm_judge=self._dummy_judge,
        )
        assert "error" in scores.details

    def test_nonexistent_db_path(self):
        scores = evaluate(
            question="test",
            generated_sql="SELECT 1",
            llm_judge=self._dummy_judge,
            db_path="/nonexistent/path.db",
            gold_sql="SELECT 1",
        )
        assert "error" in scores.details


class TestPIIWordBoundary:
    def test_no_false_positive_on_substring(self):
        """'address' should not match 'ip_address_log' table name."""
        score, details = safety_score("SELECT * FROM email_logs WHERE ip_address_log = '1.2.3.4'")
        # 'email' has word boundary in 'email_logs' — should NOT match
        # 'address' does NOT have word boundary in 'ip_address_log' — should NOT match
        pii_issues = [i for i in details["issues"] if "PII" in i]
        assert len(pii_issues) == 0

    def test_true_positive_exact_column(self):
        score, details = safety_score("SELECT email, password FROM users")
        pii_issues = [i for i in details["issues"] if "PII" in i]
        assert len(pii_issues) == 2


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])

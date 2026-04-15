"""Tests for SQLAS context quality metrics — no LLM required."""

import sqlite3
import tempfile
import os

from sqlas.context import (
    _extract_sql_elements,
    context_precision,
    context_recall,
    entity_recall,
    noise_robustness,
)
from sqlas.correctness import result_set_similarity


def _create_test_db():
    """Create a temporary SQLite database for testing."""
    path = os.path.join(tempfile.gettempdir(), "sqlas_context_test.db")
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


class TestExtractSqlElements:
    def test_basic_select(self):
        elements = _extract_sql_elements("SELECT name, age FROM users WHERE active = 1")
        assert "users" in elements["tables"]
        assert "name" in elements["columns"]
        assert "age" in elements["columns"]
        assert "active" in elements["columns"]

    def test_with_aggregation(self):
        elements = _extract_sql_elements("SELECT COUNT(*) FROM users GROUP BY active")
        assert "users" in elements["tables"]
        assert "count" in elements["functions"]

    def test_with_literals(self):
        elements = _extract_sql_elements("SELECT * FROM users WHERE name = 'Alice'")
        assert "alice" in elements["literals"] or "Alice" in elements["literals"]

    def test_invalid_sql(self):
        elements = _extract_sql_elements(";;;!!!@@@")
        assert elements["tables"] == set()
        assert elements["columns"] == set()


class TestContextPrecision:
    def test_exact_match(self):
        score, _ = context_precision(
            "SELECT name FROM users WHERE active = 1",
            "SELECT name FROM users WHERE active = 1",
        )
        assert score == 1.0

    def test_extra_elements(self):
        score, details = context_precision(
            "SELECT name, age, id FROM users WHERE active = 1",
            "SELECT name FROM users WHERE active = 1",
        )
        assert score < 1.0
        assert len(details["extra_elements"]) > 0

    def test_subset_is_perfect(self):
        score, _ = context_precision(
            "SELECT name FROM users",
            "SELECT name, age FROM users WHERE active = 1",
        )
        assert score == 1.0  # everything referenced is in gold


class TestContextRecall:
    def test_exact_match(self):
        score, _ = context_recall(
            "SELECT name FROM users WHERE active = 1",
            "SELECT name FROM users WHERE active = 1",
        )
        assert score == 1.0

    def test_missing_elements(self):
        score, details = context_recall(
            "SELECT name FROM users",
            "SELECT name, age FROM users WHERE active = 1",
        )
        assert score < 1.0
        assert len(details["missing_elements"]) > 0

    def test_superset_is_perfect(self):
        score, _ = context_recall(
            "SELECT name, age, id FROM users WHERE active = 1",
            "SELECT name FROM users WHERE active = 1",
        )
        assert score == 1.0  # all gold elements are covered


class TestEntityRecall:
    def test_exact_match(self):
        score, _ = entity_recall(
            "SELECT COUNT(*) FROM users WHERE active = 1",
            "SELECT COUNT(*) FROM users WHERE active = 1",
        )
        assert score == 1.0

    def test_missing_function(self):
        score, details = entity_recall(
            "SELECT * FROM users WHERE active = 1",
            "SELECT COUNT(*) FROM users WHERE active = 1",
        )
        assert score < 1.0
        assert "count" in details["missing_entities"]

    def test_missing_literal(self):
        score, details = entity_recall(
            "SELECT name FROM users WHERE active = 0",
            "SELECT name FROM users WHERE active = 1",
        )
        # literal '1' is in gold but '0' is in generated
        assert score < 1.0


class TestNoiseRobustness:
    def test_no_noise(self):
        score, _ = noise_robustness(
            "SELECT name FROM users WHERE active = 1",
            "SELECT name FROM users WHERE active = 1",
        )
        assert score == 1.0

    def test_noise_detected(self):
        score, details = noise_robustness(
            "SELECT name, age, id FROM users WHERE active = 1",
            "SELECT name FROM users WHERE active = 1",
        )
        assert score < 1.0
        assert details["noise_count"] > 0

    def test_with_schema_filter(self):
        score, details = noise_robustness(
            "SELECT name, age FROM users WHERE active = 1",
            "SELECT name FROM users WHERE active = 1",
            valid_tables={"users", "orders"},
            valid_columns={"users": {"id", "name", "age", "active"}, "orders": {"id", "amount"}},
        )
        # 'age' is in the schema but not in gold — counted as noise
        assert score < 1.0


class TestResultSetSimilarity:
    def test_exact_match(self):
        db = _create_test_db()
        score, details = result_set_similarity(
            "SELECT COUNT(*) FROM users WHERE active = 1",
            "SELECT COUNT(*) FROM users WHERE active = 1",
            db,
        )
        assert score == 1.0
        assert details["jaccard"] == 1.0

    def test_different_results(self):
        db = _create_test_db()
        score, _ = result_set_similarity(
            "SELECT COUNT(*) FROM users WHERE active = 0",
            "SELECT COUNT(*) FROM users WHERE active = 1",
            db,
        )
        assert score < 1.0

    def test_error_on_bad_sql(self):
        db = _create_test_db()
        score, details = result_set_similarity(
            "SELECT * FROM nonexistent_table",
            "SELECT COUNT(*) FROM users",
            db,
        )
        assert score == 0.0
        assert "error" in details


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])

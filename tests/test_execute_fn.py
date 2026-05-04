"""
Tests for execute_fn — database-agnostic execution path.

Verifies that execution_accuracy, result_set_similarity, evaluate(), evaluate_batch(),
and run_suite() all work correctly when an execute_fn is provided instead of db_path.
Includes a large-schema simulation (100+ tables) to confirm O(SQL) not O(schema) behaviour.
"""

import sqlite3
import tempfile
import os
import time

import pytest

from sqlas.correctness import execution_accuracy, result_set_similarity
from sqlas.evaluate import evaluate, evaluate_batch
from sqlas.runner import run_suite
from sqlas.core import TestCase, WEIGHTS, ExecuteFn


# ── Shared fixtures ────────────────────────────────────────────────────────


def _make_sqlite_db() -> str:
    """Create a temporary SQLite database and return its path."""
    path = os.path.join(tempfile.gettempdir(), "sqlas_exec_fn_test.db")
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS users "
        "(id INTEGER, name TEXT, age INTEGER, active INTEGER)"
    )
    conn.execute("DELETE FROM users")
    conn.executemany(
        "INSERT INTO users VALUES (?,?,?,?)",
        [(1, "Alice", 30, 1), (2, "Bob", 25, 1), (3, "Carol", 35, 0),
         (4, "Dave", 28, 1), (5, "Eve", 40, 0)],
    )
    conn.commit()
    conn.close()
    return path


def _sqlite_execute_fn(db_path: str) -> ExecuteFn:
    """Return an execute_fn backed by an existing SQLite file."""
    def execute(sql: str) -> list[tuple]:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            return conn.execute(sql).fetchall()
        finally:
            conn.close()
    return execute


def _dummy_judge(prompt: str) -> str:
    return (
        "Semantic_Score: 1.0\n"
        "Join_Correctness: 1.0\n"
        "Aggregation_Accuracy: 1.0\n"
        "Filter_Accuracy: 1.0\n"
        "Efficiency: 1.0\n"
        "Overall_Quality: 1.0\n"
        "Complexity_Match: 1.0\n"
        "Faithfulness: 1.0\n"
        "Relevance: 1.0\n"
        "Completeness: 1.0\n"
        "Fluency: 5\n"
        "Chart_Relevance: 1.0\n"
        "Data_Alignment: 1.0\n"
        "Commentary_Fit: 1.0\n"
        "Overall_Visualization: 1.0\n"
        "Reasoning: test"
    )


# ── execution_accuracy with execute_fn ────────────────────────────────────


class TestExecutionAccuracyWithExecuteFn:
    def setup_method(self):
        self.db = _make_sqlite_db()
        self.execute_fn = _sqlite_execute_fn(self.db)

    def test_exact_match(self):
        score, details = execution_accuracy(
            "SELECT COUNT(*) FROM users WHERE active = 1",
            "SELECT COUNT(*) FROM users WHERE active = 1",
            execute_fn=self.execute_fn,
        )
        # >= 0.95 not == 1.0: sub-millisecond timing jitter can make efficiency < 1.0
        # when gold_time and pred_time are both at the 0.01ms floor.
        assert score >= 0.95, f"Expected >= 0.95, got {score}"
        assert details["predicted_rows"] == 1
        assert details["gold_rows"] == 1

    def test_wrong_answer(self):
        score, _ = execution_accuracy(
            "SELECT COUNT(*) FROM users WHERE active = 0",
            "SELECT COUNT(*) FROM users WHERE active = 1",
            execute_fn=self.execute_fn,
        )
        assert score < 0.6

    def test_execute_fn_takes_precedence_over_db_path(self):
        """execute_fn should win even if a nonexistent db_path is also passed."""
        score, details = execution_accuracy(
            "SELECT COUNT(*) FROM users WHERE active = 1",
            "SELECT COUNT(*) FROM users WHERE active = 1",
            db_path="/nonexistent/path.db",
            execute_fn=self.execute_fn,
        )
        assert score == 1.0
        assert "error" not in details

    def test_execute_fn_error_returns_zero(self):
        def bad_fn(sql: str) -> list[tuple]:
            raise RuntimeError("connection refused")

        score, details = execution_accuracy(
            "SELECT 1", "SELECT 1", execute_fn=bad_fn
        )
        assert score == 0.0
        assert "error" in details

    def test_no_db_path_no_execute_fn_returns_error(self):
        score, details = execution_accuracy("SELECT 1", "SELECT 1")
        assert score == 0.0
        assert "error" in details

    def test_timing_is_measured(self):
        score, details = execution_accuracy(
            "SELECT COUNT(*) FROM users",
            "SELECT COUNT(*) FROM users",
            execute_fn=self.execute_fn,
        )
        assert "efficiency_score" in details
        assert 0.0 <= details["efficiency_score"] <= 1.0

    def test_aggregate_query(self):
        score, details = execution_accuracy(
            "SELECT AVG(age) FROM users WHERE active = 1",
            "SELECT AVG(age) FROM users WHERE active = 1",
            execute_fn=self.execute_fn,
        )
        assert score >= 0.9

    def test_multirow_result(self):
        score, details = execution_accuracy(
            "SELECT active, COUNT(*) FROM users GROUP BY active ORDER BY active",
            "SELECT active, COUNT(*) FROM users GROUP BY active ORDER BY active",
            execute_fn=self.execute_fn,
        )
        assert score >= 0.9
        assert details["predicted_rows"] == 2
        assert details["gold_rows"] == 2


# ── result_set_similarity with execute_fn ─────────────────────────────────


class TestResultSetSimilarityWithExecuteFn:
    def setup_method(self):
        self.db = _make_sqlite_db()
        self.execute_fn = _sqlite_execute_fn(self.db)

    def test_exact_match(self):
        score, details = result_set_similarity(
            "SELECT COUNT(*) FROM users WHERE active = 1",
            "SELECT COUNT(*) FROM users WHERE active = 1",
            execute_fn=self.execute_fn,
        )
        assert score == 1.0
        assert details["jaccard"] == 1.0

    def test_different_results(self):
        score, _ = result_set_similarity(
            "SELECT COUNT(*) FROM users WHERE active = 0",
            "SELECT COUNT(*) FROM users WHERE active = 1",
            execute_fn=self.execute_fn,
        )
        assert score < 1.0

    def test_execute_fn_takes_precedence_over_db_path(self):
        score, details = result_set_similarity(
            "SELECT COUNT(*) FROM users WHERE active = 1",
            "SELECT COUNT(*) FROM users WHERE active = 1",
            db_path="/nonexistent/path.db",
            execute_fn=self.execute_fn,
        )
        assert score == 1.0
        assert "error" not in details

    def test_execute_fn_error_returns_zero(self):
        def bad_fn(sql: str) -> list[tuple]:
            raise ConnectionError("DB unreachable")

        score, details = result_set_similarity("SELECT 1", "SELECT 1", execute_fn=bad_fn)
        assert score == 0.0
        assert "error" in details

    def test_column_count_inferred_from_rows(self):
        score, details = result_set_similarity(
            "SELECT name, age FROM users WHERE active = 1",
            "SELECT name, age FROM users WHERE active = 1",
            execute_fn=self.execute_fn,
        )
        assert details["column_match"] == 1.0

    def test_column_count_mismatch(self):
        score, details = result_set_similarity(
            "SELECT name FROM users WHERE active = 1",
            "SELECT name, age FROM users WHERE active = 1",
            execute_fn=self.execute_fn,
        )
        assert details["column_match"] < 1.0

    def test_no_db_path_no_execute_fn_returns_error(self):
        score, details = result_set_similarity("SELECT 1", "SELECT 1")
        assert score == 0.0
        assert "error" in details


# ── evaluate() with execute_fn ────────────────────────────────────────────


class TestEvaluateWithExecuteFn:
    def setup_method(self):
        self.db = _make_sqlite_db()
        self.execute_fn = _sqlite_execute_fn(self.db)

    def test_full_pipeline_with_execute_fn(self):
        scores = evaluate(
            question="How many active users are there?",
            generated_sql="SELECT COUNT(*) FROM users WHERE active = 1",
            gold_sql="SELECT COUNT(*) FROM users WHERE active = 1",
            llm_judge=_dummy_judge,
            execute_fn=self.execute_fn,
            response="There are 3 active users.",
            result_data={
                "columns": ["COUNT(*)"],
                "rows": [[3]],
                "row_count": 1,
                "execution_time_ms": 1.0,
            },
        )
        assert scores.execution_accuracy == 1.0
        assert scores.result_set_similarity == 1.0
        assert scores.overall_score > 0.5

    def test_execute_fn_skips_db_path_existence_check(self):
        """No error even though db_path does not exist, because execute_fn is provided."""
        scores = evaluate(
            question="test",
            generated_sql="SELECT COUNT(*) FROM users",
            gold_sql="SELECT COUNT(*) FROM users",
            llm_judge=_dummy_judge,
            db_path="/nonexistent/path.db",
            execute_fn=self.execute_fn,
        )
        assert "error" not in scores.details
        assert scores.execution_accuracy == 1.0

    def test_without_gold_sql_execute_fn_unused(self):
        """Without gold_sql, execution_accuracy is 0.5 (unverified) not 1.0.
        v2.1.1 fix: was 1.0 — gave full marks even for wrong logic queries.
        Now correctly signals that correctness cannot be verified without gold_sql."""
        scores = evaluate(
            question="count users",
            generated_sql="SELECT COUNT(*) FROM users",
            llm_judge=_dummy_judge,
            execute_fn=self.execute_fn,
            result_data={"columns": ["COUNT(*)"], "rows": [[5]], "row_count": 1, "execution_time_ms": 1.0},
        )
        assert scores.execution_accuracy == 0.5
        assert scores.details.get("execution_accuracy", {}).get("unverified") is True


# ── evaluate_batch() with execute_fn ──────────────────────────────────────


class TestEvaluateBatchWithExecuteFn:
    def setup_method(self):
        self.db = _make_sqlite_db()
        self.execute_fn = _sqlite_execute_fn(self.db)

    def test_batch_evaluation(self):
        test_cases = [
            {
                "question": "How many active users?",
                "generated_sql": "SELECT COUNT(*) FROM users WHERE active = 1",
                "gold_sql": "SELECT COUNT(*) FROM users WHERE active = 1",
            },
            {
                "question": "How many inactive users?",
                "generated_sql": "SELECT COUNT(*) FROM users WHERE active = 0",
                "gold_sql": "SELECT COUNT(*) FROM users WHERE active = 0",
            },
        ]
        results = evaluate_batch(
            test_cases=test_cases,
            llm_judge=_dummy_judge,
            execute_fn=self.execute_fn,
        )
        assert len(results) == 2
        assert all(s.execution_accuracy == 1.0 for s in results)


# ── run_suite() with execute_fn ────────────────────────────────────────────


class TestRunSuiteWithExecuteFn:
    def setup_method(self):
        self.db = _make_sqlite_db()
        self.execute_fn = _sqlite_execute_fn(self.db)

    def test_suite_with_execute_fn(self):
        test_cases = [
            TestCase(
                question="How many active users?",
                gold_sql="SELECT COUNT(*) FROM users WHERE active = 1",
                category="easy",
            ),
            TestCase(
                question="Average age of all users",
                gold_sql="SELECT AVG(age) FROM users",
                category="easy",
            ),
        ]

        def mock_agent(question: str) -> dict:
            sql_map = {
                "How many active users?": "SELECT COUNT(*) FROM users WHERE active = 1",
                "Average age of all users": "SELECT AVG(age) FROM users",
            }
            sql = sql_map.get(question, "SELECT 1")
            rows = self.execute_fn(sql)
            return {
                "sql": sql,
                "response": f"Result: {rows}",
                "data": {
                    "columns": ["result"],
                    "rows": [list(r) for r in rows],
                    "row_count": len(rows),
                    "execution_time_ms": 1.0,
                },
            }

        results = run_suite(
            test_cases=test_cases,
            agent_fn=mock_agent,
            llm_judge=_dummy_judge,
            execute_fn=self.execute_fn,
            pass_threshold=0.5,
            verbose=False,
        )
        assert results["summary"]["execution_accuracy"] >= 0.95
        assert results["summary"]["pass_rate"] == 1.0


# ── Large-schema simulation ────────────────────────────────────────────────


class TestLargeSchema:
    """
    Simulates a production warehouse with 200 tables.
    Confirms that SQLAS evaluation cost is O(SQL complexity), not O(schema size).
    """

    def _make_large_schema(self, n_tables: int = 200):
        """Build valid_tables and valid_columns for n tables."""
        valid_tables = set()
        valid_columns = {}
        for i in range(n_tables):
            tname = f"table_{i:03d}"
            valid_tables.add(tname)
            valid_columns[tname] = {f"id", f"name_{i}", f"value_{i}", "created_at", "status"}
        # Include the actual query table
        valid_tables.add("users")
        valid_columns["users"] = {"id", "name", "age", "active"}
        return valid_tables, valid_columns

    def setup_method(self):
        self.db = _make_sqlite_db()
        self.execute_fn = _sqlite_execute_fn(self.db)

    def test_schema_compliance_large_schema_is_fast(self):
        from sqlas.quality import schema_compliance
        valid_tables, valid_columns = self._make_large_schema(200)

        start = time.perf_counter()
        score, details = schema_compliance(
            "SELECT name, age FROM users WHERE active = 1",
            valid_tables=valid_tables,
            valid_columns=valid_columns,
        )
        elapsed = time.perf_counter() - start

        assert score >= 0.9
        assert elapsed < 0.5, f"schema_compliance with 200 tables took {elapsed:.3f}s — too slow"

    def test_noise_robustness_large_schema(self):
        from sqlas.context import noise_robustness
        valid_tables, valid_columns = self._make_large_schema(200)

        score, details = noise_robustness(
            "SELECT name, age FROM users WHERE active = 1",
            "SELECT name FROM users WHERE active = 1",
            valid_tables=valid_tables,
            valid_columns=valid_columns,
        )
        assert score < 1.0  # 'age' is valid schema element but not in gold
        assert details["noise_count"] > 0

    def test_full_evaluate_large_schema(self):
        valid_tables, valid_columns = self._make_large_schema(200)

        start = time.perf_counter()
        scores = evaluate(
            question="How many active users?",
            generated_sql="SELECT COUNT(*) FROM users WHERE active = 1",
            gold_sql="SELECT COUNT(*) FROM users WHERE active = 1",
            llm_judge=_dummy_judge,
            execute_fn=self.execute_fn,
            valid_tables=valid_tables,
            valid_columns=valid_columns,
            result_data={
                "columns": ["COUNT(*)"],
                "rows": [[3]],
                "row_count": 1,
                "execution_time_ms": 1.0,
            },
        )
        elapsed = time.perf_counter() - start

        assert scores.execution_accuracy == 1.0
        assert scores.schema_compliance >= 0.9
        assert elapsed < 2.0, f"evaluate() with 200-table schema took {elapsed:.3f}s — too slow"

    def test_run_suite_200_table_schema(self):
        valid_tables, valid_columns = self._make_large_schema(200)

        test_cases = [
            TestCase(
                question=f"Query {i}",
                gold_sql="SELECT COUNT(*) FROM users WHERE active = 1",
                category="easy",
            )
            for i in range(5)
        ]

        def mock_agent(question: str) -> dict:
            return {
                "sql": "SELECT COUNT(*) FROM users WHERE active = 1",
                "response": "3 active users.",
                "data": {
                    "columns": ["COUNT(*)"],
                    "rows": [[3]],
                    "row_count": 1,
                    "execution_time_ms": 1.0,
                },
            }

        results = run_suite(
            test_cases=test_cases,
            agent_fn=mock_agent,
            llm_judge=_dummy_judge,
            execute_fn=self.execute_fn,
            valid_tables=valid_tables,
            valid_columns=valid_columns,
            pass_threshold=0.5,
            verbose=False,
        )
        assert results["summary"]["execution_accuracy"] >= 0.959
        assert results["summary"]["pass_rate"] == 1.0


# ── Simulated non-SQLite executor ─────────────────────────────────────────


class TestNonSQLiteExecutor:
    """
    Simulates a Postgres/MySQL/Snowflake executor using an in-memory
    dict-based query engine. Verifies that execute_fn is truly DB-agnostic.
    """

    def _make_postgres_mock_executor(self) -> ExecuteFn:
        """Simulate a Postgres executor backed by in-memory data."""
        _tables = {
            "orders": [
                (1, "2026-01-01", 150.0, "US"),
                (2, "2026-01-02", 300.0, "UK"),
                (3, "2026-01-03", 450.0, "US"),
                (4, "2026-01-04", 200.0, "DE"),
                (5, "2026-01-05", 100.0, "UK"),
            ]
        }
        # Use SQLite in-memory as the backend to parse and execute queries
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE orders (id INTEGER, date TEXT, amount REAL, country TEXT)")
        conn.executemany("INSERT INTO orders VALUES (?,?,?,?)", _tables["orders"])
        conn.commit()

        def execute(sql: str) -> list[tuple]:
            return conn.execute(sql).fetchall()

        return execute

    def test_postgres_mock_executor_exact_match(self):
        execute_fn = self._make_postgres_mock_executor()

        score, details = execution_accuracy(
            "SELECT country, SUM(amount) FROM orders GROUP BY country ORDER BY country",
            "SELECT country, SUM(amount) FROM orders GROUP BY country ORDER BY country",
            execute_fn=execute_fn,
        )
        assert score >= 0.9

    def test_postgres_mock_executor_wrong_filter(self):
        execute_fn = self._make_postgres_mock_executor()

        score, _ = execution_accuracy(
            "SELECT SUM(amount) FROM orders WHERE country = 'FR'",
            "SELECT SUM(amount) FROM orders WHERE country = 'US'",
            execute_fn=execute_fn,
        )
        assert score < 0.7

    def test_result_set_similarity_with_mock_postgres(self):
        execute_fn = self._make_postgres_mock_executor()

        score, details = result_set_similarity(
            "SELECT COUNT(*) FROM orders",
            "SELECT COUNT(*) FROM orders",
            execute_fn=execute_fn,
        )
        assert score == 1.0

    def test_full_evaluate_with_mock_postgres(self):
        execute_fn = self._make_postgres_mock_executor()

        scores = evaluate(
            question="Total revenue by country",
            generated_sql="SELECT country, SUM(amount) FROM orders GROUP BY country",
            gold_sql="SELECT country, SUM(amount) FROM orders GROUP BY country",
            llm_judge=_dummy_judge,
            execute_fn=execute_fn,
        )
        assert scores.execution_accuracy >= 0.9
        assert scores.overall_score > 0.4


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])

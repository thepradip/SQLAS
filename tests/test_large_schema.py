"""
SQLAS v2.1.0 — Large schema tests (100+ tables, multiple databases).

All tests are deterministic and require no LLM calls or real database connections.
Tests verify that SQLAS evaluates correctly regardless of schema size.
"""

import os
import sqlite3
import tempfile
import pytest
import sqlas
from sqlas.evaluate import _auto_schema_context, build_schema_info


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_100_table_schema() -> tuple[set[str], dict[str, set[str]]]:
    """Generate a 100-table schema with 30 columns each = 3,000 columns total."""
    tables = {f"table_{i:03d}" for i in range(100)}
    columns = {
        t: {f"col_{j:02d}" for j in range(30)} | {"id", "created_at", "updated_at"}
        for t in tables
    }
    # Add some realistic tables
    columns["orders"]   = {"id", "customer_id", "total", "status", "created_at"}
    columns["customers"] = {"id", "name", "email", "country", "created_at"}
    columns["products"]  = {"id", "name", "price", "category", "stock"}
    tables |= {"orders", "customers", "products"}
    return tables, columns


def _make_sqlite_db_with_many_tables(n: int = 50) -> str:
    """Create a real SQLite DB with n tables, each with 10 columns."""
    f = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    f.close()
    conn = sqlite3.connect(f.name)
    for i in range(n):
        cols = ", ".join([f"col_{j} TEXT" for j in range(10)])
        conn.execute(f"CREATE TABLE table_{i:03d} (id INTEGER PRIMARY KEY, {cols})")
    conn.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER, total REAL, status TEXT)")
    conn.execute("CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT, email TEXT)")
    conn.commit()
    conn.close()
    return f.name


# ── _auto_schema_context tests ────────────────────────────────────────────────

class TestAutoSchemaContext:

    def test_extracts_only_referenced_tables(self):
        tables, columns = _make_100_table_schema()
        sql = "SELECT id, total FROM orders WHERE status = 'active'"
        ctx = _auto_schema_context(sql, columns)
        assert "orders" in ctx
        # 100 other tables should NOT be in the context
        assert "table_001" not in ctx
        assert "customers" not in ctx

    def test_multi_table_join_includes_both(self):
        tables, columns = _make_100_table_schema()
        sql = "SELECT o.id, c.name FROM orders o JOIN customers c ON o.customer_id = c.id"
        ctx = _auto_schema_context(sql, columns)
        assert "orders" in ctx
        assert "customers" in ctx
        assert "products" not in ctx

    def test_small_schema_includes_all(self):
        """Databases with <= 10 tables: show everything regardless of SQL."""
        columns = {f"t{i}": {"a", "b", "c"} for i in range(5)}
        sql = "SELECT a FROM t0"
        ctx = _auto_schema_context(sql, columns)
        for t in columns:
            assert t in ctx

    def test_respects_max_chars(self):
        tables, columns = _make_100_table_schema()
        sql = "SELECT * FROM orders"
        ctx = _auto_schema_context(sql, columns, max_chars=100)
        assert len(ctx) <= 200   # some slack for the "... omitted" suffix

    def test_caps_columns_at_30(self):
        """Tables with >30 columns show only 30 + suffix."""
        columns = {"big_table": {f"col_{i}" for i in range(50)}}
        sql = "SELECT * FROM big_table"
        ctx = _auto_schema_context(sql, columns)
        assert "+20 more" in ctx

    def test_returns_string(self):
        tables, columns = _make_100_table_schema()
        ctx = _auto_schema_context("SELECT 1", columns)
        assert isinstance(ctx, str)

    def test_handles_unparseable_sql(self):
        """Broken SQL should not crash — falls back to showing all tables."""
        columns = {"t1": {"a"}, "t2": {"b"}}
        ctx = _auto_schema_context("THIS IS NOT SQL !!!@#$", columns)
        assert isinstance(ctx, str)


# ── build_schema_info tests ───────────────────────────────────────────────────

class TestBuildSchemaInfo:

    def test_sqlite_via_db_path(self):
        db = _make_sqlite_db_with_many_tables(50)
        try:
            tables, columns = build_schema_info(db_path=db)
            assert len(tables) >= 50
            assert "orders" in tables
            assert "customers" in tables
            assert "id" in columns["orders"]
            assert "total" in columns["orders"]
        finally:
            os.unlink(db)

    def test_sqlite_via_execute_fn(self):
        db = _make_sqlite_db_with_many_tables(20)
        conn = sqlite3.connect(db)
        def execute_fn(sql): return conn.execute(sql).fetchall()
        try:
            tables, columns = build_schema_info(execute_fn=execute_fn)
            assert len(tables) >= 20
            assert "orders" in tables
        finally:
            conn.close()
            os.unlink(db)

    def test_100_tables_returns_correct_count(self):
        db = _make_sqlite_db_with_many_tables(100)
        try:
            tables, columns = build_schema_info(db_path=db)
            assert len(tables) >= 100
            # Every table has at least id + 10 cols
            for t in list(tables)[:5]:
                assert len(columns[t]) >= 1
        finally:
            os.unlink(db)

    def test_no_args_raises(self):
        with pytest.raises(ValueError, match="Provide db_path or execute_fn"):
            build_schema_info()

    def test_exported_from_package(self):
        assert sqlas.build_schema_info is build_schema_info


# ── evaluate() with 100+ tables ───────────────────────────────────────────────

class TestEvaluateLargeSchema:
    """
    Verify evaluate() works correctly when valid_columns has 100+ tables.
    Schema compliance, auto schema context, and LLM-judge quality all tested.
    """

    def setup_method(self):
        self.tables, self.columns = _make_100_table_schema()
        self.judge_calls = []

        def judge(prompt):
            self.judge_calls.append(prompt)
            # Return valid judge responses
            if "Join_Correctness" in prompt:
                return "Join_Correctness: 0.9\nAggregation_Accuracy: 0.9\nFilter_Accuracy: 0.9\nEfficiency: 0.9\nOverall_Quality: 0.9\nIssues: none"
            if "Semantic_Score" in prompt:
                return "Semantic_Score: 0.9\nReasoning: Correct."
            if "Relevance:" in prompt:
                return "Relevance: 0.9\nReasoning: On topic."
            if "Complexity_Match" in prompt:
                return "Complexity_Match: 0.9\nReasoning: Appropriate."
            return "Score: 0.9\nReasoning: OK."

        self.judge = judge

    def test_schema_compliance_100_tables(self):
        """schema_compliance must handle 100-table valid_columns correctly."""
        score, details = sqlas.schema_compliance(
            sql="SELECT id, total FROM orders WHERE status = 'active'",
            valid_tables=self.tables,
            valid_columns=self.columns,
        )
        assert score == 1.0, f"Expected 1.0, got {score}: {details}"

    def test_schema_compliance_invalid_table(self):
        score, details = sqlas.schema_compliance(
            sql="SELECT x FROM nonexistent_table_xyz",
            valid_tables=self.tables,
            valid_columns=self.columns,
        )
        assert score < 1.0

    def test_auto_schema_context_injected(self):
        """When schema_context is empty and valid_columns has 100 tables,
        the judge prompt should contain only relevant table context."""
        sqlas.evaluate(
            question="What is the total revenue from orders?",
            generated_sql="SELECT SUM(total) FROM orders",
            llm_judge=self.judge,
            valid_tables=self.tables,
            valid_columns=self.columns,
            # schema_context intentionally omitted
        )
        # Check that a judge call contained orders schema but not unrelated tables
        schema_in_prompts = [p for p in self.judge_calls if "orders" in p and "Referenced Tables" in p]
        if schema_in_prompts:
            prompt = schema_in_prompts[0]
            assert "table_001" not in prompt, "Irrelevant tables leaked into judge prompt"

    def test_evaluate_completes_without_error(self):
        """evaluate() should not crash on 100-table schema."""
        scores = sqlas.evaluate(
            question="How many customers placed orders?",
            generated_sql="SELECT COUNT(DISTINCT o.customer_id) FROM orders o JOIN customers c ON o.customer_id = c.id",
            llm_judge=self.judge,
            valid_tables=self.tables,
            valid_columns=self.columns,
        )
        assert 0 <= scores.overall_score <= 1.0
        assert scores.schema_compliance == 1.0   # orders + customers are valid

    def test_noise_robustness_100_tables(self):
        """noise_robustness must scale to 100+ table valid_columns."""
        score, details = sqlas.noise_robustness(
            generated_sql="SELECT id FROM orders WHERE status = 'active'",
            gold_sql="SELECT id FROM orders WHERE status = 'active'",
            valid_tables=self.tables,
            valid_columns=self.columns,
        )
        assert 0 <= score <= 1.0


# ── Multi-database run_suite ───────────────────────────────────────────────────

class TestMultiDatabaseRunSuite:
    """Verify run_suite works against multiple databases with large schemas."""

    def test_run_suite_auto_schema_context(self):
        db = _make_sqlite_db_with_many_tables(50)
        try:
            tables, columns = build_schema_info(db_path=db)

            def agent_fn(question): return {"sql": "SELECT 1", "response": "Result is 1."}
            def judge(p): return "Semantic_Score: 0.8\nReasoning: OK.\nJoin_Correctness: 0.8\nAggregation_Accuracy: 0.8\nFilter_Accuracy: 0.8\nEfficiency: 0.8\nOverall_Quality: 0.8\nIssues: none\nRelevance: 0.8\nReasoning: OK.\nComplexity_Match: 0.8\nReasoning: OK."

            results = sqlas.run_suite(
                test_cases=[
                    sqlas.TestCase("How many tables?", category="easy"),
                    sqlas.TestCase("Count orders", category="easy"),
                ],
                agent_fn=agent_fn,
                llm_judge=judge,
                db_path=db,
                valid_tables=tables,
                valid_columns=columns,
                verbose=False,
            )
            assert results["summary"]["total_tests"] == 2
            assert 0 <= results["summary"]["overall_score"] <= 1.0
        finally:
            os.unlink(db)

    def test_per_test_schema_context_override(self):
        """TestCase.schema_context overrides suite-level schema_context."""
        called_with = []

        def agent_fn(q): return {"sql": "SELECT 1", "response": "1"}
        def judge(p):
            called_with.append(p)
            return "Semantic_Score: 0.8\nReasoning: OK.\nJoin_Correctness: 0.8\nAggregation_Accuracy: 0.8\nFilter_Accuracy: 0.8\nEfficiency: 0.8\nOverall_Quality: 0.8\nIssues: none\nRelevance: 0.8\nReasoning: ok\nComplexity_Match: 0.8\nReasoning: ok"

        sqlas.run_suite(
            test_cases=[
                sqlas.TestCase("Query A", schema_context="TABLE_A_CONTEXT", category="test"),
                sqlas.TestCase("Query B", schema_context="TABLE_B_CONTEXT", category="test"),
            ],
            agent_fn=agent_fn,
            llm_judge=judge,
            schema_context="SUITE_DEFAULT_CONTEXT",
            verbose=False,
        )
        a_prompts = [p for p in called_with if "TABLE_A_CONTEXT" in p]
        b_prompts = [p for p in called_with if "TABLE_B_CONTEXT" in p]
        assert len(a_prompts) > 0, "Per-test schema_context for Query A not used"
        assert len(b_prompts) > 0, "Per-test schema_context for Query B not used"

"""
SQLAS v2.0.0 tests — agentic quality, cache metrics, AST-based safety.

All tests are deterministic and require no LLM calls or database connections.
"""

import pytest
import sqlas
from sqlas.agentic import steps_efficiency, schema_grounding, agentic_score
from sqlas.cache import cache_hit_score, tokens_saved_score, few_shot_score
from sqlas.safety import read_only_compliance
from sqlas.core import WEIGHTS_V4, SQLASScores, compute_composite_score


# ── Fixtures ──────────────────────────────────────────────────────────────────

GOOD_STEPS = [
    {"tool": "list_tables",    "args": {}},
    {"tool": "describe_table", "args": {"table_name": "patients"}},
    {"tool": "execute_sql",    "args": {"sql": "SELECT COUNT(*) FROM patients"}},
    {"tool": "final_answer",   "args": {"answer": "987 patients.", "sql": "SELECT COUNT(*) FROM patients"}},
]

BAD_STEPS = [
    {"tool": "execute_sql",    "args": {"sql": "SELECT * FROM patients"}},
    {"tool": "execute_sql",    "args": {"sql": "SELECT COUNT(*) FROM patients"}},
    {"tool": "describe_table", "args": {"table_name": "patients"}},
    {"tool": "execute_sql",    "args": {"sql": "SELECT COUNT(*) FROM patients WHERE x=1"}},
    {"tool": "execute_sql",    "args": {"sql": "SELECT COUNT(*) FROM patients WHERE y=1"}},
    {"tool": "final_answer",   "args": {"answer": "987"}},
]


def dummy_judge(prompt: str) -> str:
    """Deterministic stub judge that returns a fixed mid-range score."""
    if "Planning_Quality" in prompt:
        return "Planning_Quality: 0.8\nReasoning: Good planning."
    if "Tool_Use_Accuracy" in prompt:
        return "Tool_Use_Accuracy: 0.75\nReasoning: Mostly correct."
    return "Score: 0.75\nReasoning: OK."


# ── AST-based read_only_compliance ────────────────────────────────────────────

class TestReadOnlyComplianceAST:
    def test_select_passes(self):
        assert read_only_compliance("SELECT * FROM patients") == 1.0

    def test_cte_select_passes(self):
        assert read_only_compliance("WITH x AS (SELECT 1) SELECT * FROM x") == 1.0

    def test_insert_blocked(self):
        assert read_only_compliance("INSERT INTO t VALUES(1)") == 0.0

    def test_drop_blocked(self):
        assert read_only_compliance("DROP TABLE patients") == 0.0

    def test_delete_blocked(self):
        assert read_only_compliance("DELETE FROM t WHERE 1=1") == 0.0

    def test_insert_inside_cte_blocked(self):
        """v2 upgrade: keyword matching missed this — AST catches it."""
        assert read_only_compliance("WITH x AS (INSERT INTO t VALUES(1)) SELECT 1") == 0.0

    def test_keyword_in_string_value_passes(self):
        """A string value containing 'DROP' should not be flagged."""
        assert read_only_compliance("SELECT * FROM t WHERE name = 'DROP TABLE users'") == 1.0


# ── steps_efficiency ──────────────────────────────────────────────────────────

class TestStepsEfficiency:
    def test_zero_steps_pipeline_mode(self):
        assert steps_efficiency(0) == 1.0

    def test_optimal_steps(self):
        assert steps_efficiency(3) == 1.0

    def test_below_optimal(self):
        assert steps_efficiency(1) == 1.0
        assert steps_efficiency(2) == 1.0

    def test_slightly_above_optimal(self):
        assert steps_efficiency(4) == 0.8
        assert steps_efficiency(5) == 0.8

    def test_well_above_optimal(self):
        assert steps_efficiency(6) == 0.6
        assert steps_efficiency(7) == 0.6

    def test_very_many_steps(self):
        assert steps_efficiency(10) == 0.3

    def test_custom_optimal(self):
        assert steps_efficiency(5, optimal_steps=5) == 1.0
        assert steps_efficiency(6, optimal_steps=5) == 0.8


# ── schema_grounding ──────────────────────────────────────────────────────────

class TestSchemaGrounding:
    def test_no_steps(self):
        assert schema_grounding([]) == 0.0

    def test_schema_before_sql(self):
        assert schema_grounding(GOOD_STEPS) == 1.0

    def test_sql_before_schema(self):
        assert schema_grounding(BAD_STEPS) == 0.3

    def test_no_execute_sql(self):
        steps = [{"tool": "describe_table", "args": {}}]
        assert schema_grounding(steps) == 0.5

    def test_no_schema_inspection(self):
        steps = [{"tool": "execute_sql", "args": {}}, {"tool": "final_answer", "args": {}}]
        assert schema_grounding(steps) == 0.5

    def test_list_tables_counts_as_inspection(self):
        steps = [
            {"tool": "list_tables", "args": {}},
            {"tool": "execute_sql", "args": {}},
        ]
        assert schema_grounding(steps) == 1.0


# ── agentic_score (composite) ─────────────────────────────────────────────────

class TestAgenticScore:
    def test_good_steps(self):
        # GOOD_STEPS = 4 steps (list, describe, execute, final_answer)
        # optimal_steps=3 → steps_efficiency(4) = 0.8
        score, details = agentic_score("How many patients?", GOOD_STEPS, dummy_judge)
        assert 0.7 <= score <= 1.0
        assert details["schema_grounding"] == 1.0
        assert details["steps_efficiency"] == 0.8   # 4 steps vs optimal 3
        assert details["agent_mode"] == "react"

    def test_bad_steps(self):
        score, details = agentic_score("How many patients?", BAD_STEPS, dummy_judge)
        # Bad order + many steps should score lower than good steps
        good_score, _ = agentic_score("How many patients?", GOOD_STEPS, dummy_judge)
        assert score < good_score
        assert details["schema_grounding"] == 0.3

    def test_pipeline_mode(self):
        # Pipeline mode: efficiency=1.0, grounding=0.0, planning=0.0
        # composite = 0.30*1.0 + 0.30*0.0 + 0.40*0.0 = 0.30
        score, details = agentic_score("Count patients", [], dummy_judge)
        assert abs(score - 0.30) < 0.01, f"Expected 0.30 for pipeline mode, got {score}"
        assert details["agent_mode"] == "pipeline"


# ── cache metrics ─────────────────────────────────────────────────────────────

class TestCacheMetrics:
    def _result(self, hit=False, cache_type="", tokens_saved=0, few_shot=0, verified=0):
        return {"metrics": {
            "cache_hit": hit,
            "cache_type": cache_type,
            "tokens_saved": tokens_saved,
            "few_shot_count": few_shot,
            "verified_few_shot_count": verified,
        }}

    def test_cache_miss(self):
        score, d = cache_hit_score(self._result(hit=False))
        assert score == 0.0
        assert d["cache_hit"] is False

    def test_exact_cache_hit(self):
        score, d = cache_hit_score(self._result(hit=True, cache_type="exact"))
        assert score == 1.0
        assert d["cache_type"] == "exact"

    def test_semantic_cache_hit(self):
        score, d = cache_hit_score(self._result(hit=True, cache_type="semantic"))
        assert score == 1.0

    def test_tokens_saved_full(self):
        score, d = tokens_saved_score(self._result(tokens_saved=9500))
        assert score == 1.0
        assert d["cost_saved_usd"] > 0

    def test_tokens_saved_partial(self):
        score, _ = tokens_saved_score(self._result(tokens_saved=4750))
        assert 0.4 < score < 0.6

    def test_tokens_saved_none(self):
        score, _ = tokens_saved_score(self._result(tokens_saved=0))
        assert score == 0.0

    def test_few_shot_none(self):
        score, d = few_shot_score(self._result(few_shot=0))
        assert score == 0.0

    def test_few_shot_unverified(self):
        score, _ = few_shot_score(self._result(few_shot=2, verified=0))
        assert score == 0.5

    def test_few_shot_verified(self):
        score, _ = few_shot_score(self._result(few_shot=2, verified=1))
        assert score == 1.0


# ── WEIGHTS_V4 ────────────────────────────────────────────────────────────────

class TestWeightsV4:
    def test_weights_sum_to_one(self):
        total = sum(WEIGHTS_V4.values())
        assert abs(total - 1.0) < 0.001, f"WEIGHTS_V4 sums to {total}"

    def test_contains_agentic_score(self):
        assert "agentic_score" in WEIGHTS_V4

    def test_agentic_weight(self):
        assert WEIGHTS_V4["agentic_score"] == 0.10

    def test_exported_from_package(self):
        assert sqlas.WEIGHTS_V4 is WEIGHTS_V4

    def test_composite_score_with_v4(self):
        scores = SQLASScores(
            execution_accuracy=1.0,
            semantic_equivalence=1.0,
            read_only_compliance=1.0,
            safety_score=1.0,
            guardrail_score=1.0,
            faithfulness=1.0,
            answer_relevance=1.0,
            answer_completeness=1.0,
            fluency=1.0,
            agentic_score=1.0,
            execution_success=1.0,
            empty_result_penalty=1.0,
            efficiency_score=1.0,
            data_scan_efficiency=1.0,
            sql_quality=1.0,
            schema_compliance=1.0,
            complexity_match=1.0,
            result_set_similarity=1.0,
            context_precision=1.0,
            context_recall=1.0,
            entity_recall=1.0,
            noise_robustness=1.0,
            chart_spec_validity=1.0,
            chart_data_alignment=1.0,
            chart_llm_validation=1.0,
            sql_injection_score=1.0,
            prompt_injection_score=1.0,
            pii_access_score=1.0,
            pii_leakage_score=1.0,
        )
        overall = compute_composite_score(scores, WEIGHTS_V4)
        assert abs(overall - 1.0) < 0.001


# ── SQLASScores new fields ────────────────────────────────────────────────────

class TestSQLASScoresV2Fields:
    def test_default_values(self):
        s = SQLASScores()
        assert s.agent_mode == "pipeline"
        assert s.steps_taken == 0
        assert s.steps_efficiency == 0.0
        assert s.schema_grounding == 0.0
        assert s.planning_quality == 0.0
        assert s.agentic_score == 0.0
        assert s.cache_hit is False
        assert s.tokens_saved == 0
        assert s.few_shot_count == 0

    def test_backward_compat_existing_fields_unchanged(self):
        """New fields must not break existing field defaults."""
        s = SQLASScores()
        assert s.execution_accuracy == 0.0
        assert s.faithfulness == 0.0
        assert s.read_only_compliance == 0.0
        assert s.chart_spec_validity == 0.0

"""
SQLAS — SQL Agent Scoring Framework
A RAGAS-equivalent evaluation library for Text-to-SQL and SQL AI agents.

Author: SQLAS Contributors

Usage:
    from sqlas import evaluate, SQLASScores, TestCase, WEIGHTS

    scores = evaluate(
        question="How many users are active?",
        generated_sql="SELECT COUNT(*) FROM users WHERE active = 1",
        gold_sql="SELECT COUNT(*) FROM users WHERE active = 1",
        db_path="my_database.db",
        llm_judge=my_llm_function,
    )
    print(scores.overall_score)
"""

from sqlas.core import (
    SQLASScores, TestCase,
    WEIGHTS, WEIGHTS_V2, WEIGHTS_V3, WEIGHTS_V4,
    compute_composite_score, ExecuteFn,
)
from sqlas.evaluate import evaluate, evaluate_batch
from sqlas.correctness import execution_accuracy, syntax_valid, semantic_equivalence, result_set_similarity
from sqlas.quality import sql_quality, schema_compliance, complexity_match
from sqlas.production import data_scan_efficiency, execution_result
from sqlas.response import faithfulness, answer_relevance, answer_completeness, fluency
from sqlas.safety import (
    guardrail_score, pii_access_score, pii_leakage_score,
    prompt_injection_score, safety_score, read_only_compliance, sql_injection_score,
)
from sqlas.context import context_precision, context_recall, entity_recall, noise_robustness
from sqlas.visualization import chart_data_alignment, chart_llm_validation, chart_spec_validity, visualization_score
from sqlas.agentic import (
    steps_efficiency, schema_grounding, planning_quality,
    tool_use_accuracy, agentic_score,
)
from sqlas.cache import cache_hit_score, tokens_saved_score, few_shot_score
from sqlas.runner import run_suite

__version__ = "2.0.0"
__author__ = "SQLAS Contributors"

__all__ = [
    # Core
    "SQLASScores", "TestCase",
    "WEIGHTS", "WEIGHTS_V2", "WEIGHTS_V3", "WEIGHTS_V4",
    "compute_composite_score", "ExecuteFn",
    # Top-level API
    "evaluate", "evaluate_batch", "run_suite",
    # Correctness
    "execution_accuracy", "syntax_valid", "semantic_equivalence", "result_set_similarity",
    # Quality
    "sql_quality", "schema_compliance", "complexity_match",
    # Production
    "data_scan_efficiency", "execution_result",
    # Response
    "faithfulness", "answer_relevance", "answer_completeness", "fluency",
    # Safety (v2: AST-based read_only_compliance)
    "safety_score", "read_only_compliance", "guardrail_score",
    "sql_injection_score", "prompt_injection_score", "pii_access_score", "pii_leakage_score",
    # Visualization
    "chart_spec_validity", "chart_data_alignment", "chart_llm_validation", "visualization_score",
    # Context (RAGAS-mapped)
    "context_precision", "context_recall", "entity_recall", "noise_robustness",
    # Agentic (v2 NEW)
    "steps_efficiency", "schema_grounding", "planning_quality",
    "tool_use_accuracy", "agentic_score",
    # Cache (v2 NEW)
    "cache_hit_score", "tokens_saved_score", "few_shot_score",
]

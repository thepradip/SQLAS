"""
SQLAS — SQL Agent Scoring Framework
A RAGAS-equivalent evaluation library for Text-to-SQL and SQL AI agents.

Author: Pradip Tivhale

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

from sqlas.core import SQLASScores, TestCase, WEIGHTS, WEIGHTS_V2, compute_composite_score
from sqlas.evaluate import evaluate, evaluate_batch
from sqlas.correctness import execution_accuracy, syntax_valid, semantic_equivalence, result_set_similarity
from sqlas.quality import sql_quality, schema_compliance, complexity_match
from sqlas.production import data_scan_efficiency, execution_result
from sqlas.response import faithfulness, answer_relevance, answer_completeness, fluency
from sqlas.safety import safety_score, read_only_compliance
from sqlas.context import context_precision, context_recall, entity_recall, noise_robustness
from sqlas.runner import run_suite

__version__ = "1.1.0"
__author__ = "Pradip Tivhale"

__all__ = [
    # Core
    "SQLASScores",
    "TestCase",
    "WEIGHTS",
    "WEIGHTS_V2",
    "compute_composite_score",
    # Top-level API
    "evaluate",
    "evaluate_batch",
    "run_suite",
    # Correctness metrics
    "execution_accuracy",
    "syntax_valid",
    "semantic_equivalence",
    "result_set_similarity",
    # Quality metrics
    "sql_quality",
    "schema_compliance",
    "complexity_match",
    # Production metrics
    "data_scan_efficiency",
    "execution_result",
    # Response metrics
    "faithfulness",
    "answer_relevance",
    "answer_completeness",
    "fluency",
    # Safety metrics
    "safety_score",
    "read_only_compliance",
    # Context metrics (RAGAS-mapped)
    "context_precision",
    "context_recall",
    "entity_recall",
    "noise_robustness",
]

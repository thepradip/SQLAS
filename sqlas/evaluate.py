"""
Main evaluation API — single query and batch evaluation.

Author: SQLAS Contributors
"""

import logging
import os

from sqlas.core import SQLASScores, TestCase, LLMJudge, WEIGHTS, compute_composite_score
from sqlas.correctness import execution_accuracy, syntax_valid, semantic_equivalence, result_set_similarity
from sqlas.quality import sql_quality, schema_compliance, complexity_match
from sqlas.production import data_scan_efficiency, execution_result
from sqlas.response import faithfulness, answer_relevance, answer_completeness, fluency
from sqlas.safety import safety_score, read_only_compliance
from sqlas.context import context_precision, context_recall, entity_recall, noise_robustness

logger = logging.getLogger(__name__)


def evaluate(
    question: str,
    generated_sql: str,
    llm_judge: LLMJudge,
    gold_sql: str | None = None,
    db_path: str | None = None,
    response: str | None = None,
    result_data: dict | None = None,
    valid_tables: set[str] | None = None,
    valid_columns: dict[str, set[str]] | None = None,
    schema_context: str = "",
    expected_nonempty: bool = True,
    pii_columns: list[str] | None = None,
    weights: dict | None = None,
) -> SQLASScores:
    """
    Evaluate a single SQL agent query across all SQLAS metrics.

    Args:
        question:        User's natural language question
        generated_sql:   SQL produced by the agent
        llm_judge:       Function (prompt: str) -> str for LLM-as-judge metrics
        gold_sql:        Ground-truth SQL (optional, enables execution accuracy & context metrics)
        db_path:         Path to SQLite database (required for execution accuracy)
        response:        Agent's natural language response (optional, enables faithfulness/relevance)
        result_data:     Query result dict: {columns, rows, row_count, execution_time_ms}
        valid_tables:    Set of valid table names (enables schema compliance)
        valid_columns:   Dict of {table: {col1, col2}} (enables schema compliance)
        schema_context:  Brief schema text for SQL quality judge
        expected_nonempty: Whether non-empty result is expected
        pii_columns:     Custom PII column names for safety check
        weights:         Custom weight dict (defaults to SQLAS production weights)

    Returns:
        SQLASScores with all metrics and overall_score
    """
    # ── Input validation ────────────────────────────────────────────────
    if not generated_sql or not isinstance(generated_sql, str):
        logger.error("generated_sql must be a non-empty string")
        scores = SQLASScores()
        scores.details["error"] = "generated_sql is empty or invalid"
        return scores

    if db_path and not os.path.exists(db_path):
        logger.error("db_path does not exist: %s", db_path)
        scores = SQLASScores()
        scores.details["error"] = f"db_path not found: {db_path}"
        return scores

    if weights:
        weight_sum = sum(weights.values())
        if abs(weight_sum - 1.0) > 0.01:
            logger.warning("Custom weights sum to %.4f (expected ~1.0)", weight_sum)

    scores = SQLASScores()

    # ── 1. Core Correctness ─────────────────────────────────────────────
    scores.syntax_valid = syntax_valid(generated_sql)

    if gold_sql and db_path:
        ex_acc, ex_details = execution_accuracy(generated_sql, gold_sql, db_path)
        scores.execution_accuracy = ex_acc
        scores.details["execution_accuracy"] = ex_details

        # Efficiency (VES) — reuse timing from execution_accuracy
        scores.efficiency_score = ex_details.get("efficiency_score", 0.0)
    else:
        scores.execution_accuracy = 1.0 if result_data else 0.0
        scores.efficiency_score = 1.0 if result_data else 0.0

    sem, sem_details = semantic_equivalence(question, generated_sql, llm_judge, gold_sql)
    scores.semantic_equivalence = sem
    scores.details["semantic_equivalence"] = sem_details

    # ── 2. Context Quality (RAGAS-mapped) ───────────────────────────────
    if gold_sql:
        cp, cp_details = context_precision(generated_sql, gold_sql)
        scores.context_precision = cp
        scores.details["context_precision"] = cp_details

        cr, cr_details = context_recall(generated_sql, gold_sql)
        scores.context_recall = cr
        scores.details["context_recall"] = cr_details

        er, er_details = entity_recall(generated_sql, gold_sql)
        scores.entity_recall = er
        scores.details["entity_recall"] = er_details

        nr, nr_details = noise_robustness(generated_sql, gold_sql, valid_tables, valid_columns)
        scores.noise_robustness = nr
        scores.details["noise_robustness"] = nr_details

        if db_path:
            rs, rs_details = result_set_similarity(generated_sql, gold_sql, db_path)
            scores.result_set_similarity = rs
            scores.details["result_set_similarity"] = rs_details

    # ── 3. SQL Quality ──────────────────────────────────────────────────
    if valid_tables and valid_columns:
        sc, sc_details = schema_compliance(generated_sql, valid_tables, valid_columns)
        scores.schema_compliance = sc
        scores.details["schema_compliance"] = sc_details
    else:
        scores.schema_compliance = 1.0  # can't check without schema

    sq, sq_details = sql_quality(question, generated_sql, llm_judge, schema_context)
    scores.sql_quality = sq
    scores.details["sql_quality"] = sq_details

    cm, cm_details = complexity_match(question, generated_sql, llm_judge)
    scores.complexity_match = cm
    scores.details["complexity_match"] = cm_details

    # ── 4. Production Execution ─────────────────────────────────────────
    exec_eval = execution_result(result_data, expected_nonempty)
    scores.execution_success = exec_eval["execution_success"]
    scores.execution_time_ms = exec_eval["execution_time_ms"]
    scores.result_row_count = exec_eval["result_row_count"]
    scores.empty_result_penalty = exec_eval["empty_result_penalty"]
    scores.row_explosion_detected = exec_eval["row_explosion_detected"]

    scan, scan_details = data_scan_efficiency(generated_sql, scores.result_row_count)
    scores.data_scan_efficiency = scan
    scores.details["data_scan"] = scan_details

    # ── 5. Response Quality ─────────────────────────────────────────────
    if response and result_data:
        result_preview = f"Columns: {result_data.get('columns', [])}\n"
        for row in result_data.get("rows", [])[:15]:
            result_preview += f"{row}\n"

        f_score, f_details = faithfulness(question, response, result_preview, llm_judge)
        scores.faithfulness = f_score
        scores.details["faithfulness"] = f_details

        r_score, r_details = answer_relevance(question, response, llm_judge)
        scores.answer_relevance = r_score
        scores.details["answer_relevance"] = r_details

        c_score, c_details = answer_completeness(question, response, result_preview, llm_judge)
        scores.answer_completeness = c_score
        scores.details["answer_completeness"] = c_details

        fl_score, fl_details = fluency(response, llm_judge)
        scores.fluency = fl_score
        scores.details["fluency"] = fl_details

    # ── 6. Safety ───────────────────────────────────────────────────────
    scores.read_only_compliance = read_only_compliance(generated_sql)

    safety, safety_details = safety_score(generated_sql, response or "", pii_columns)
    scores.safety_score = safety
    scores.details["safety"] = safety_details

    # ── Composite ───────────────────────────────────────────────────────
    scores.overall_score = compute_composite_score(scores, weights)

    return scores


def evaluate_batch(
    test_cases: list[dict],
    llm_judge: LLMJudge,
    db_path: str | None = None,
    valid_tables: set[str] | None = None,
    valid_columns: dict[str, set[str]] | None = None,
    schema_context: str = "",
    pii_columns: list[str] | None = None,
    weights: dict | None = None,
) -> list[SQLASScores]:
    """
    Evaluate a batch of test cases.

    Each dict in test_cases should have:
        question, generated_sql, and optionally:
        gold_sql, response, result_data, expected_nonempty

    Returns list of SQLASScores.
    """
    results = []
    for tc in test_cases:
        scores = evaluate(
            question=tc["question"],
            generated_sql=tc["generated_sql"],
            llm_judge=llm_judge,
            gold_sql=tc.get("gold_sql"),
            db_path=db_path,
            response=tc.get("response"),
            result_data=tc.get("result_data"),
            valid_tables=valid_tables,
            valid_columns=valid_columns,
            schema_context=tc.get("schema_context", schema_context),
            expected_nonempty=tc.get("expected_nonempty", True),
            pii_columns=pii_columns,
            weights=weights,
        )
        results.append(scores)
    return results

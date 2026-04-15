"""
Test suite runner with optional MLflow integration.

Author: SQLAS Contributors
"""

import logging
import time
from sqlas.core import SQLASScores, TestCase, LLMJudge
from sqlas.evaluate import evaluate

logger = logging.getLogger(__name__)


def run_suite(
    test_cases: list[TestCase],
    agent_fn,
    llm_judge: LLMJudge,
    db_path: str | None = None,
    valid_tables: set[str] | None = None,
    valid_columns: dict[str, set[str]] | None = None,
    weights: dict | None = None,
    pass_threshold: float = 0.6,
    verbose: bool = True,
) -> dict:
    """
    Run SQLAS evaluation suite.

    Args:
        test_cases:      List of TestCase objects
        agent_fn:        Function(question: str) -> dict with keys:
                         sql, response, data (optional: {columns, rows, row_count, execution_time_ms})
        llm_judge:       Function (prompt: str) -> str
        db_path:         SQLite database path (for execution accuracy)
        valid_tables:    Set of valid table names
        valid_columns:   Dict {table: {cols}}
        weights:         Custom weights (optional)
        pass_threshold:  Minimum overall_score to count as PASS (default 0.6)
        verbose:         Print progress

    Returns:
        {"summary": {...}, "details": [SQLASScores, ...]}
    """
    if verbose:
        print(f"SQLAS — Running {len(test_cases)} test cases...\n")
    logger.info("SQLAS suite started: %d test cases", len(test_cases))

    all_scores: list[SQLASScores] = []
    category_scores: dict[str, list[float]] = {}
    start = time.perf_counter()

    for i, tc in enumerate(test_cases):
        if verbose:
            print(f"  [{i+1}/{len(test_cases)}] {tc.category:12s} | {tc.question[:55]}...")
        logger.info("Running test %d/%d: %s", i + 1, len(test_cases), tc.question[:80])

        # Run agent
        result = agent_fn(tc.question)

        # Evaluate
        scores = evaluate(
            question=tc.question,
            generated_sql=result.get("sql", ""),
            llm_judge=llm_judge,
            gold_sql=tc.gold_sql,
            db_path=db_path,
            response=result.get("response"),
            result_data=result.get("data"),
            valid_tables=valid_tables,
            valid_columns=valid_columns,
            expected_nonempty=tc.expected_nonempty,
            weights=weights,
        )

        all_scores.append(scores)
        category_scores.setdefault(tc.category, []).append(scores.overall_score)

        if verbose:
            status = "PASS" if scores.overall_score >= pass_threshold else "WARN" if scores.overall_score >= pass_threshold * 0.67 else "FAIL"
            print(f"           {status} | {scores.overall_score:.2f} | "
                  f"ExAcc:{scores.execution_accuracy:.2f} Sem:{scores.semantic_equivalence:.2f} "
                  f"Faith:{scores.faithfulness:.2f} Safety:{scores.safety_score:.2f}")

    elapsed = time.perf_counter() - start
    n = len(all_scores)
    avg = lambda attr: round(sum(getattr(s, attr) for s in all_scores) / n, 4) if n else 0

    summary = {
        "total_tests": n,
        "overall_score": avg("overall_score"),
        "pass_rate": round(sum(1 for s in all_scores if s.overall_score >= pass_threshold) / n, 4) if n else 0,
        "time_seconds": round(elapsed, 1),
        # Correctness
        "execution_accuracy": avg("execution_accuracy"),
        "semantic_equivalence": avg("semantic_equivalence"),
        # Context Quality
        "context_precision": avg("context_precision"),
        "context_recall": avg("context_recall"),
        "entity_recall": avg("entity_recall"),
        "noise_robustness": avg("noise_robustness"),
        "result_set_similarity": avg("result_set_similarity"),
        # Quality
        "sql_quality": avg("sql_quality"),
        "schema_compliance": avg("schema_compliance"),
        # Efficiency
        "efficiency_score": avg("efficiency_score"),
        "data_scan_efficiency": avg("data_scan_efficiency"),
        # Response
        "faithfulness": avg("faithfulness"),
        "answer_relevance": avg("answer_relevance"),
        "answer_completeness": avg("answer_completeness"),
        "fluency": avg("fluency"),
        # Safety
        "read_only_compliance": avg("read_only_compliance"),
        "safety_score": avg("safety_score"),
        "by_category": {
            cat: round(sum(s) / len(s), 4) for cat, s in category_scores.items()
        },
    }

    logger.info("SQLAS suite complete: score=%.4f pass_rate=%.0f%% time=%.1fs",
                summary["overall_score"], summary["pass_rate"] * 100, summary["time_seconds"])

    if verbose:
        print(f"\n{'='*60}")
        print(f"  SQLAS Score: {summary['overall_score']:.4f} / 1.0  |  Pass Rate: {summary['pass_rate']*100:.0f}%")
        print(f"  Time: {summary['time_seconds']}s  |  Metrics: 20")
        for cat, avg_val in summary["by_category"].items():
            bar = "#" * int(avg_val * 20) + "." * (20 - int(avg_val * 20))
            print(f"  {cat:15s}  [{bar}] {avg_val:.4f}")
        print(f"{'='*60}")

    return {"summary": summary, "details": all_scores}

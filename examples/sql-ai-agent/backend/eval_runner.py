from __future__ import annotations
"""
SQLAS Evaluation Runner — executes the full test suite and logs results to MLflow.

Usage:
    python eval_runner.py              # Run all test cases (25)
    python eval_runner.py --quick      # Run first 5 test cases only

Author: Pradip Tivhale
"""

import asyncio
import sys
import time
from pathlib import Path

import mlflow
from mlflow.entities import SpanType

from config import get_settings
from database import get_table_list, get_full_schema, build_full_context, execute_readonly_query
from agent import init_agent, run_query
from tracing import init_tracing
from eval_framework import (
    TestCase, SQLASScores, evaluate_single, SCORE_WEIGHTS,
    compute_overall_score, _match_result_sets,
)

settings = get_settings()
DB_PATH = str(Path(__file__).resolve().parent / "health.db")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST SUITE — 25 ground truth queries across 4 difficulty tiers
# Gold SQL uses raw numeric codes (not CASE labels) for robust comparison
# ═══════════════════════════════════════════════════════════════════════════════

TEST_SUITE: list[TestCase] = [
    # ── EASY (7): Single table, simple filter/count/avg ──────────────────
    TestCase(
        question="How many patients have abnormal blood pressure?",
        gold_sql="SELECT COUNT(*) FROM health_demographics WHERE Blood_Pressure_Abnormality = 1",
        expected_tables=["health_demographics"],
        expects_join=False,
        category="easy",
    ),
    TestCase(
        question="What is the average BMI of male patients?",
        gold_sql="SELECT ROUND(AVG(BMI), 2) FROM health_demographics WHERE Sex = 0",
        expected_tables=["health_demographics"],
        expects_join=False,
        category="easy",
    ),
    TestCase(
        question="How many female patients are pregnant?",
        gold_sql="SELECT COUNT(*) FROM health_demographics WHERE Sex = 1 AND Pregnancy = 1",
        expected_tables=["health_demographics"],
        expects_join=False,
        category="easy",
    ),
    TestCase(
        question="What is the total number of smokers in the dataset?",
        gold_sql="SELECT COUNT(*) FROM health_demographics WHERE Smoking = 1",
        expected_tables=["health_demographics"],
        expects_join=False,
        category="easy",
    ),
    TestCase(
        question="What is the average hemoglobin level across all patients?",
        gold_sql="SELECT ROUND(AVG(Level_of_Hemoglobin), 2) FROM health_demographics",
        expected_tables=["health_demographics"],
        expects_join=False,
        category="easy",
    ),
    TestCase(
        question="How many patients have chronic kidney disease?",
        gold_sql="SELECT COUNT(*) FROM health_demographics WHERE Chronic_kidney_disease = 1",
        expected_tables=["health_demographics"],
        expects_join=False,
        category="easy",
    ),
    TestCase(
        question="What is the maximum salt content in the diet across all patients?",
        gold_sql="SELECT MAX(salt_content_in_the_diet) FROM health_demographics",
        expected_tables=["health_demographics"],
        expects_join=False,
        category="easy",
    ),

    # ── MEDIUM (8): Aggregation, GROUP BY, multi-condition ───────────────
    TestCase(
        question="What is the average BMI for smokers vs non-smokers?",
        gold_sql="""
            SELECT Smoking, ROUND(AVG(BMI), 2) AS avg_bmi
            FROM health_demographics
            GROUP BY Smoking
            ORDER BY Smoking
        """,
        expected_tables=["health_demographics"],
        expects_join=False,
        category="medium",
    ),
    TestCase(
        question="What is the distribution of stress levels among patients with chronic kidney disease?",
        gold_sql="""
            SELECT Level_of_Stress, COUNT(*) AS patient_count
            FROM health_demographics
            WHERE Chronic_kidney_disease = 1
            GROUP BY Level_of_Stress
            ORDER BY Level_of_Stress
        """,
        expected_tables=["health_demographics"],
        expects_join=False,
        category="medium",
    ),
    TestCase(
        question="What percentage of patients with high stress also have abnormal blood pressure?",
        gold_sql="""
            SELECT ROUND(CAST(SUM(CASE WHEN Blood_Pressure_Abnormality = 1 THEN 1 ELSE 0 END) AS REAL) / COUNT(*) * 100, 2)
            FROM health_demographics
            WHERE Level_of_Stress = 3
        """,
        expected_tables=["health_demographics"],
        expects_join=False,
        category="medium",
    ),
    TestCase(
        question="What is the average alcohol consumption by gender?",
        gold_sql="""
            SELECT Sex, ROUND(AVG(alcohol_consumption_per_day), 2) AS avg_alcohol
            FROM health_demographics
            GROUP BY Sex
            ORDER BY Sex
        """,
        expected_tables=["health_demographics"],
        expects_join=False,
        category="medium",
    ),
    TestCase(
        question="How many patients have both chronic kidney disease and adrenal/thyroid disorders?",
        gold_sql="SELECT COUNT(*) FROM health_demographics WHERE Chronic_kidney_disease = 1 AND Adrenal_and_thyroid_disorders = 1",
        expected_tables=["health_demographics"],
        expects_join=False,
        category="medium",
    ),
    TestCase(
        question="What is the average age of patients with abnormal blood pressure vs normal?",
        gold_sql="""
            SELECT Blood_Pressure_Abnormality, ROUND(AVG(Age), 2) AS avg_age
            FROM health_demographics
            GROUP BY Blood_Pressure_Abnormality
            ORDER BY Blood_Pressure_Abnormality
        """,
        expected_tables=["health_demographics"],
        expects_join=False,
        category="medium",
    ),
    TestCase(
        question="What is the average genetic pedigree coefficient for patients with abnormal blood pressure?",
        gold_sql="""
            SELECT ROUND(AVG(Genetic_Pedigree_Coefficient), 2)
            FROM health_demographics
            WHERE Blood_Pressure_Abnormality = 1 AND Genetic_Pedigree_Coefficient IS NOT NULL
        """,
        expected_tables=["health_demographics"],
        expects_join=False,
        category="medium",
    ),
    TestCase(
        question="How many patients have zero alcohol consumption?",
        gold_sql="SELECT COUNT(*) FROM health_demographics WHERE alcohol_consumption_per_day = 0",
        expected_tables=["health_demographics"],
        expects_join=False,
        category="medium",
    ),

    # ── HARD (7): Cross-table JOIN queries ───────────────────────────────
    TestCase(
        question="What is the average daily steps for patients with chronic kidney disease vs without?",
        gold_sql="""
            SELECT h.Chronic_kidney_disease, ROUND(AVG(p.Physical_activity), 2) AS avg_steps
            FROM physical_activity p
            JOIN health_demographics h ON p.Patient_Number = h.Patient_Number
            WHERE p.Physical_activity IS NOT NULL
            GROUP BY h.Chronic_kidney_disease
            ORDER BY h.Chronic_kidney_disease
        """,
        expected_tables=["health_demographics", "physical_activity"],
        expects_join=True,
        category="hard",
    ),
    TestCase(
        question="Compare average daily steps of smokers vs non-smokers",
        gold_sql="""
            SELECT h.Smoking, ROUND(AVG(p.Physical_activity), 2) AS avg_steps
            FROM physical_activity p
            JOIN health_demographics h ON p.Patient_Number = h.Patient_Number
            WHERE p.Physical_activity IS NOT NULL
            GROUP BY h.Smoking
            ORDER BY h.Smoking
        """,
        expected_tables=["health_demographics", "physical_activity"],
        expects_join=True,
        category="hard",
    ),
    TestCase(
        question="What is the average daily steps by stress level?",
        gold_sql="""
            SELECT h.Level_of_Stress, ROUND(AVG(p.Physical_activity), 2) AS avg_steps
            FROM physical_activity p
            JOIN health_demographics h ON p.Patient_Number = h.Patient_Number
            WHERE p.Physical_activity IS NOT NULL
            GROUP BY h.Level_of_Stress
            ORDER BY h.Level_of_Stress
        """,
        expected_tables=["health_demographics", "physical_activity"],
        expects_join=True,
        category="hard",
    ),
    TestCase(
        question="What is the average daily steps for patients with abnormal blood pressure vs normal?",
        gold_sql="""
            SELECT h.Blood_Pressure_Abnormality, ROUND(AVG(p.Physical_activity), 2) AS avg_steps
            FROM physical_activity p
            JOIN health_demographics h ON p.Patient_Number = h.Patient_Number
            WHERE p.Physical_activity IS NOT NULL
            GROUP BY h.Blood_Pressure_Abnormality
            ORDER BY h.Blood_Pressure_Abnormality
        """,
        expected_tables=["health_demographics", "physical_activity"],
        expects_join=True,
        category="hard",
    ),
    TestCase(
        question="Which gender has higher average physical activity?",
        gold_sql="""
            SELECT h.Sex, ROUND(AVG(p.Physical_activity), 2) AS avg_steps
            FROM physical_activity p
            JOIN health_demographics h ON p.Patient_Number = h.Patient_Number
            WHERE p.Physical_activity IS NOT NULL
            GROUP BY h.Sex
            ORDER BY avg_steps DESC
        """,
        expected_tables=["health_demographics", "physical_activity"],
        expects_join=True,
        category="hard",
    ),
    TestCase(
        question="Show the top 5 most active patients with their age and BMI",
        gold_sql="""
            SELECT h.Patient_Number, h.Age, h.BMI, ROUND(AVG(p.Physical_activity), 0) AS avg_steps
            FROM physical_activity p
            JOIN health_demographics h ON p.Patient_Number = h.Patient_Number
            WHERE p.Physical_activity IS NOT NULL
            GROUP BY h.Patient_Number
            ORDER BY avg_steps DESC
            LIMIT 5
        """,
        expected_tables=["health_demographics", "physical_activity"],
        expects_join=True,
        category="hard",
    ),
    TestCase(
        question="How many days of physical activity data are missing per patient on average?",
        gold_sql="""
            SELECT ROUND(AVG(null_days), 2) AS avg_missing_days
            FROM (
                SELECT Patient_Number, SUM(CASE WHEN Physical_activity IS NULL THEN 1 ELSE 0 END) AS null_days
                FROM physical_activity
                GROUP BY Patient_Number
            )
        """,
        expected_tables=["physical_activity"],
        expects_join=False,
        category="hard",
    ),

    # ── EXTRA HARD (3): Complex analytics ────────────────────────────────
    TestCase(
        question="Is there a correlation between hemoglobin level and average physical activity?",
        gold_sql="""
            WITH patient_data AS (
                SELECT h.Level_of_Hemoglobin AS hgb, AVG(p.Physical_activity) AS avg_steps
                FROM physical_activity p
                JOIN health_demographics h ON p.Patient_Number = h.Patient_Number
                WHERE p.Physical_activity IS NOT NULL
                GROUP BY h.Patient_Number
            )
            SELECT ROUND(
                (COUNT(*) * SUM(hgb * avg_steps) - SUM(hgb) * SUM(avg_steps)) /
                (SQRT(COUNT(*) * SUM(hgb * hgb) - SUM(hgb) * SUM(hgb)) *
                 SQRT(COUNT(*) * SUM(avg_steps * avg_steps) - SUM(avg_steps) * SUM(avg_steps)))
            , 2) AS pearson_r
            FROM patient_data
        """,
        expected_tables=["health_demographics", "physical_activity"],
        expects_join=True,
        category="extra_hard",
    ),
    TestCase(
        question="What is the average BMI by age group (under 30, 30-50, 50-70, over 70)?",
        gold_sql="""
            SELECT
                CASE
                    WHEN Age < 30 THEN 'Under 30'
                    WHEN Age BETWEEN 30 AND 50 THEN '30-50'
                    WHEN Age BETWEEN 51 AND 70 THEN '51-70'
                    ELSE 'Over 70'
                END AS age_group,
                ROUND(AVG(BMI), 2) AS avg_bmi,
                COUNT(*) AS patient_count
            FROM health_demographics
            GROUP BY age_group
            ORDER BY MIN(Age)
        """,
        expected_tables=["health_demographics"],
        expects_join=False,
        category="extra_hard",
    ),
    TestCase(
        question="Which combination of smoking and stress level has the highest rate of abnormal blood pressure?",
        gold_sql="""
            SELECT Smoking, Level_of_Stress,
                   COUNT(*) AS total,
                   SUM(Blood_Pressure_Abnormality) AS abnormal,
                   ROUND(CAST(SUM(Blood_Pressure_Abnormality) AS REAL) / COUNT(*) * 100, 2) AS abnormal_pct
            FROM health_demographics
            GROUP BY Smoking, Level_of_Stress
            ORDER BY abnormal_pct DESC
        """,
        expected_tables=["health_demographics"],
        expects_join=False,
        category="extra_hard",
    ),
]


# ═══════════════════════════════════════════════════════════════════════════════
# RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

async def run_evaluation(quick: bool = False) -> dict:
    """Run SQLAS evaluation suite and log to MLflow."""
    print("=" * 70)
    print("  SQLAS — SQL Agent Scoring Framework")
    print("  Author: Pradip Tivhale")
    print("=" * 70)

    init_tracing()
    await init_agent()

    schema = await get_full_schema()
    valid_tables = set(schema.keys())
    valid_columns = {
        table: {col["name"] for col in info["columns"]}
        for table, info in schema.items()
    }
    schema_context = str(schema)[:500]  # brief schema for SQL quality judge

    test_cases = TEST_SUITE[:5] if quick else TEST_SUITE

    print(f"\nRunning {len(test_cases)} test cases ({len(SCORE_WEIGHTS)} metrics each)...\n")

    all_scores: list[dict] = []
    category_scores: dict[str, list[float]] = {}

    with mlflow.start_span("sqlas_evaluation_suite", span_type=SpanType.EVALUATOR) as eval_span:
        eval_span.set_inputs({"test_count": len(test_cases), "mode": "quick" if quick else "full"})
        suite_start = time.perf_counter()

        for i, tc in enumerate(test_cases):
            print(f"  [{i+1}/{len(test_cases)}] {tc.category.upper():12s} | {tc.question[:55]}...")

            with mlflow.start_span(f"eval_{i+1}_{tc.category}", span_type=SpanType.TASK) as case_span:
                case_span.set_inputs({"question": tc.question, "category": tc.category})

                agent_result = await run_query(tc.question, [])

                scores = await evaluate_single(
                    test_case=tc,
                    agent_result=agent_result,
                    db_path=DB_PATH,
                    valid_tables=valid_tables,
                    valid_columns=valid_columns,
                    schema_context=schema_context,
                )

                case_span.set_outputs({
                    "overall_score": scores.overall_score,
                    "execution_accuracy": scores.execution_accuracy,
                    "semantic_equivalence": scores.semantic_equivalence,
                    "faithfulness": scores.faithfulness,
                })
                case_span.set_attributes({
                    "eval.category": tc.category,
                    "eval.overall": scores.overall_score,
                    # Correctness
                    "eval.execution_accuracy": scores.execution_accuracy,
                    "eval.semantic_equivalence": scores.semantic_equivalence,
                    # SQL Quality
                    "eval.sql_quality": scores.sql_quality,
                    "eval.schema_compliance": scores.schema_compliance,
                    "eval.complexity_match": scores.query_complexity_appropriate,
                    # Production
                    "eval.efficiency": scores.efficiency_score,
                    "eval.data_scan_efficiency": scores.data_scan_efficiency,
                    # Response
                    "eval.faithfulness": scores.faithfulness,
                    "eval.relevance": scores.answer_relevance,
                    "eval.completeness": scores.answer_completeness,
                    # Safety
                    "eval.safety": scores.safety_score,
                    "eval.read_only": scores.read_only_compliance,
                })

                result_row = {
                    "question": tc.question,
                    "category": tc.category,
                    "overall": scores.overall_score,
                    # 1. Correctness
                    "exec_accuracy": scores.execution_accuracy,
                    "semantic_eq": scores.semantic_equivalence,
                    # 2. SQL Quality
                    "sql_quality": scores.sql_quality,
                    "schema_compliance": scores.schema_compliance,
                    "complexity_match": scores.query_complexity_appropriate,
                    # 3. Production
                    "efficiency": scores.efficiency_score,
                    "data_scan": scores.data_scan_efficiency,
                    "exec_success": scores.execution_success,
                    # 4. Response
                    "faithfulness": scores.faithfulness,
                    "relevance": scores.answer_relevance,
                    "completeness": scores.answer_completeness,
                    "fluency": scores.fluency,
                    # 5. Safety
                    "read_only": scores.read_only_compliance,
                    "safety": scores.safety_score,
                }
                all_scores.append(result_row)

                cat = tc.category
                category_scores.setdefault(cat, []).append(scores.overall_score)

                status = "PASS" if scores.overall_score >= 0.6 else "WARN" if scores.overall_score >= 0.4 else "FAIL"
                print(f"           {status} | Overall: {scores.overall_score:.2f} | "
                      f"ExAcc: {scores.execution_accuracy:.2f} | Semantic: {scores.semantic_equivalence:.2f} | "
                      f"Faith: {scores.faithfulness:.2f} | Safety: {scores.safety_score:.2f}")

            time.sleep(0.5)

        suite_time = (time.perf_counter() - suite_start) * 1000

        overall_scores = [s["overall"] for s in all_scores]
        n = len(all_scores)
        avg = lambda key: round(sum(s[key] for s in all_scores) / n, 4)

        summary = {
            "total_tests": n,
            "avg_overall_score": round(sum(overall_scores) / n, 4),
            "pass_rate": round(sum(1 for s in overall_scores if s >= 0.6) / n, 4),
            "suite_time_seconds": round(suite_time / 1000, 1),
            # 1. Correctness (40%)
            "avg_exec_accuracy": avg("exec_accuracy"),
            "avg_semantic_equivalence": avg("semantic_eq"),
            # 2. SQL Quality (15%)
            "avg_sql_quality": avg("sql_quality"),
            "avg_schema_compliance": avg("schema_compliance"),
            "avg_complexity_match": avg("complexity_match"),
            # 3. Production (15%)
            "avg_efficiency": avg("efficiency"),
            "avg_data_scan": avg("data_scan"),
            # 4. Response (20%)
            "avg_faithfulness": avg("faithfulness"),
            "avg_relevance": avg("relevance"),
            "avg_completeness": avg("completeness"),
            "avg_fluency": avg("fluency"),
            # 5. Safety (10%)
            "avg_read_only": avg("read_only"),
            "avg_safety": avg("safety"),
            # By category
            "category_breakdown": {
                cat: round(sum(scores) / len(scores), 4)
                for cat, scores in category_scores.items()
            },
        }

        eval_span.set_outputs(summary)
        eval_span.set_attributes({
            "eval.suite.avg_overall": summary["avg_overall_score"],
            "eval.suite.pass_rate": summary["pass_rate"],
            "eval.suite.total_tests": summary["total_tests"],
        })

    print("\n" + "=" * 70)
    print("  SQLAS v2.0 — PRODUCTION EVALUATION REPORT")
    print("  Author: Pradip Tivhale")
    print("=" * 70)
    print(f"\n  Tests: {summary['total_tests']}  |  Time: {summary['suite_time_seconds']}s  |  Pass Rate: {summary['pass_rate']*100:.0f}%")
    print(f"\n  OVERALL SQLAS SCORE:  {summary['avg_overall_score']:.4f} / 1.0")

    print(f"\n  ── 1. Core Correctness (40%) ──")
    print(f"  Execution Accuracy:      {summary['avg_exec_accuracy']:.4f}")
    print(f"  Semantic Equivalence:    {summary['avg_semantic_equivalence']:.4f}")

    print(f"\n  ── 2. SQL Quality (15%) ──")
    print(f"  SQL Quality:             {summary['avg_sql_quality']:.4f}")
    print(f"  Schema Compliance:       {summary['avg_schema_compliance']:.4f}")
    print(f"  Complexity Match:        {summary['avg_complexity_match']:.4f}")

    print(f"\n  ── 3. Production Execution (15%) ──")
    print(f"  Efficiency (VES):        {summary['avg_efficiency']:.4f}")
    print(f"  Data Scan Efficiency:    {summary['avg_data_scan']:.4f}")

    print(f"\n  ── 4. Response Quality (20%) ──")
    print(f"  Faithfulness:            {summary['avg_faithfulness']:.4f}")
    print(f"  Answer Relevance:        {summary['avg_relevance']:.4f}")
    print(f"  Answer Completeness:     {summary['avg_completeness']:.4f}")
    print(f"  Fluency:                 {summary['avg_fluency']:.4f}")

    print(f"\n  ── 5. Safety & Governance (10%) ──")
    print(f"  Read-Only Compliance:    {summary['avg_read_only']:.4f}")
    print(f"  Safety Score:            {summary['avg_safety']:.4f}")

    print(f"\n  ── By Difficulty ──")
    for cat, avg_val in summary["category_breakdown"].items():
        bar = "█" * int(avg_val * 20) + "░" * (20 - int(avg_val * 20))
        print(f"  {cat:15s}  {bar} {avg_val:.4f}")

    print("\n" + "=" * 70)

    return {"summary": summary, "details": all_scores}


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    asyncio.run(run_evaluation(quick=quick))

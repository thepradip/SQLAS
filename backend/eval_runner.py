from __future__ import annotations
"""
SQLAS Evaluation Runner — uses the sqlas package to evaluate the LangGraph agent.

Usage:
    python eval_runner.py              # Run all test cases (25)
    python eval_runner.py --quick      # Run first 5 test cases only

Author: Pradip Tivhale
"""

import asyncio
import sys
import time
import logging
from pathlib import Path

from sqlas import TestCase, run_suite, evaluate as sqlas_evaluate
from openai import AzureOpenAI

from config import get_settings
from agent.graph import run_query
from agent.nodes import init_schema

logger = logging.getLogger(__name__)
settings = get_settings()

DB_PATH = str(Path(__file__).resolve().parent / "health.db")

client = AzureOpenAI(
    azure_endpoint=settings.azure_openai_endpoint,
    api_key=settings.azure_openai_api_key,
    api_version=settings.azure_openai_api_version,
)


def llm_judge(prompt: str) -> str:
    """LLM judge function for SQLAS evaluation."""
    response = client.chat.completions.create(
        model=settings.azure_openai_deployment_name,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=500,
    )
    return response.choices[0].message.content


# ═══════════════════════════════════════════════════════════════════════════════
# TEST SUITE — 25 ground truth queries across 4 difficulty tiers
# ═══════════════════════════════════════════════════════════════════════════════

TEST_SUITE: list[TestCase] = [
    # ── EASY (7) ───────────────────────────────────────────────────────────
    TestCase(
        question="How many patients have abnormal blood pressure?",
        gold_sql="SELECT COUNT(*) FROM health_demographics WHERE Blood_Pressure_Abnormality = 1",
        category="easy",
    ),
    TestCase(
        question="What is the average BMI of male patients?",
        gold_sql="SELECT ROUND(AVG(BMI), 2) FROM health_demographics WHERE Sex = 0",
        category="easy",
    ),
    TestCase(
        question="How many female patients are pregnant?",
        gold_sql="SELECT COUNT(*) FROM health_demographics WHERE Sex = 1 AND Pregnancy = 1",
        category="easy",
    ),
    TestCase(
        question="What is the total number of smokers in the dataset?",
        gold_sql="SELECT COUNT(*) FROM health_demographics WHERE Smoking = 1",
        category="easy",
    ),
    TestCase(
        question="What is the average hemoglobin level across all patients?",
        gold_sql="SELECT ROUND(AVG(Level_of_Hemoglobin), 2) FROM health_demographics",
        category="easy",
    ),
    TestCase(
        question="How many patients have chronic kidney disease?",
        gold_sql="SELECT COUNT(*) FROM health_demographics WHERE Chronic_kidney_disease = 1",
        category="easy",
    ),
    TestCase(
        question="What is the maximum salt content in the diet across all patients?",
        gold_sql="SELECT MAX(salt_content_in_the_diet) FROM health_demographics",
        category="easy",
    ),

    # ── MEDIUM (8) ─────────────────────────────────────────────────────────
    TestCase(
        question="What is the average BMI for smokers vs non-smokers?",
        gold_sql="SELECT Smoking, ROUND(AVG(BMI), 2) AS avg_bmi FROM health_demographics GROUP BY Smoking ORDER BY Smoking",
        category="medium",
    ),
    TestCase(
        question="What is the distribution of stress levels among patients with chronic kidney disease?",
        gold_sql="SELECT Level_of_Stress, COUNT(*) AS patient_count FROM health_demographics WHERE Chronic_kidney_disease = 1 GROUP BY Level_of_Stress ORDER BY Level_of_Stress",
        category="medium",
    ),
    TestCase(
        question="What percentage of patients with high stress also have abnormal blood pressure?",
        gold_sql="SELECT ROUND(CAST(SUM(CASE WHEN Blood_Pressure_Abnormality = 1 THEN 1 ELSE 0 END) AS REAL) / COUNT(*) * 100, 2) FROM health_demographics WHERE Level_of_Stress = 3",
        category="medium",
    ),
    TestCase(
        question="What is the average alcohol consumption by gender?",
        gold_sql="SELECT Sex, ROUND(AVG(alcohol_consumption_per_day), 2) AS avg_alcohol FROM health_demographics GROUP BY Sex ORDER BY Sex",
        category="medium",
    ),
    TestCase(
        question="How many patients have both chronic kidney disease and adrenal/thyroid disorders?",
        gold_sql="SELECT COUNT(*) FROM health_demographics WHERE Chronic_kidney_disease = 1 AND Adrenal_and_thyroid_disorders = 1",
        category="medium",
    ),
    TestCase(
        question="What is the average age of patients with abnormal blood pressure vs normal?",
        gold_sql="SELECT Blood_Pressure_Abnormality, ROUND(AVG(Age), 2) AS avg_age FROM health_demographics GROUP BY Blood_Pressure_Abnormality ORDER BY Blood_Pressure_Abnormality",
        category="medium",
    ),
    TestCase(
        question="What is the average genetic pedigree coefficient for patients with abnormal blood pressure?",
        gold_sql="SELECT ROUND(AVG(Genetic_Pedigree_Coefficient), 2) FROM health_demographics WHERE Blood_Pressure_Abnormality = 1 AND Genetic_Pedigree_Coefficient IS NOT NULL",
        category="medium",
    ),
    TestCase(
        question="How many patients have zero alcohol consumption?",
        gold_sql="SELECT COUNT(*) FROM health_demographics WHERE alcohol_consumption_per_day = 0",
        category="medium",
    ),

    # ── HARD (7): Cross-table JOIN queries ─────────────────────────────────
    TestCase(
        question="What is the average daily steps for patients with chronic kidney disease vs without?",
        gold_sql="SELECT h.Chronic_kidney_disease, ROUND(AVG(p.Physical_activity), 2) AS avg_steps FROM physical_activity p JOIN health_demographics h ON p.Patient_Number = h.Patient_Number WHERE p.Physical_activity IS NOT NULL GROUP BY h.Chronic_kidney_disease ORDER BY h.Chronic_kidney_disease",
        category="hard",
    ),
    TestCase(
        question="Compare average daily steps of smokers vs non-smokers",
        gold_sql="SELECT h.Smoking, ROUND(AVG(p.Physical_activity), 2) AS avg_steps FROM physical_activity p JOIN health_demographics h ON p.Patient_Number = h.Patient_Number WHERE p.Physical_activity IS NOT NULL GROUP BY h.Smoking ORDER BY h.Smoking",
        category="hard",
    ),
    TestCase(
        question="What is the average daily steps by stress level?",
        gold_sql="SELECT h.Level_of_Stress, ROUND(AVG(p.Physical_activity), 2) AS avg_steps FROM physical_activity p JOIN health_demographics h ON p.Patient_Number = h.Patient_Number WHERE p.Physical_activity IS NOT NULL GROUP BY h.Level_of_Stress ORDER BY h.Level_of_Stress",
        category="hard",
    ),
    TestCase(
        question="What is the average daily steps for patients with abnormal blood pressure vs normal?",
        gold_sql="SELECT h.Blood_Pressure_Abnormality, ROUND(AVG(p.Physical_activity), 2) AS avg_steps FROM physical_activity p JOIN health_demographics h ON p.Patient_Number = h.Patient_Number WHERE p.Physical_activity IS NOT NULL GROUP BY h.Blood_Pressure_Abnormality ORDER BY h.Blood_Pressure_Abnormality",
        category="hard",
    ),
    TestCase(
        question="Which gender has higher average physical activity?",
        gold_sql="SELECT h.Sex, ROUND(AVG(p.Physical_activity), 2) AS avg_steps FROM physical_activity p JOIN health_demographics h ON p.Patient_Number = h.Patient_Number WHERE p.Physical_activity IS NOT NULL GROUP BY h.Sex ORDER BY avg_steps DESC",
        category="hard",
    ),
    TestCase(
        question="Show the top 5 most active patients with their age and BMI",
        gold_sql="SELECT h.Patient_Number, h.Age, h.BMI, ROUND(AVG(p.Physical_activity), 0) AS avg_steps FROM physical_activity p JOIN health_demographics h ON p.Patient_Number = h.Patient_Number WHERE p.Physical_activity IS NOT NULL GROUP BY h.Patient_Number ORDER BY avg_steps DESC LIMIT 5",
        category="hard",
    ),
    TestCase(
        question="How many days of physical activity data are missing per patient on average?",
        gold_sql="SELECT ROUND(AVG(null_days), 2) AS avg_missing_days FROM (SELECT Patient_Number, SUM(CASE WHEN Physical_activity IS NULL THEN 1 ELSE 0 END) AS null_days FROM physical_activity GROUP BY Patient_Number)",
        category="hard",
    ),

    # ── EXTRA HARD (3): Complex analytics ──────────────────────────────────
    TestCase(
        question="Is there a correlation between hemoglobin level and average physical activity?",
        gold_sql="WITH patient_data AS (SELECT h.Level_of_Hemoglobin AS hgb, AVG(p.Physical_activity) AS avg_steps FROM physical_activity p JOIN health_demographics h ON p.Patient_Number = h.Patient_Number WHERE p.Physical_activity IS NOT NULL GROUP BY h.Patient_Number) SELECT ROUND((COUNT(*) * SUM(hgb * avg_steps) - SUM(hgb) * SUM(avg_steps)) / (SQRT(COUNT(*) * SUM(hgb * hgb) - SUM(hgb) * SUM(hgb)) * SQRT(COUNT(*) * SUM(avg_steps * avg_steps) - SUM(avg_steps) * SUM(avg_steps))), 2) AS pearson_r FROM patient_data",
        category="extra_hard",
    ),
    TestCase(
        question="What is the average BMI by age group (under 30, 30-50, 50-70, over 70)?",
        gold_sql="SELECT CASE WHEN Age < 30 THEN 'Under 30' WHEN Age BETWEEN 30 AND 50 THEN '30-50' WHEN Age BETWEEN 51 AND 70 THEN '51-70' ELSE 'Over 70' END AS age_group, ROUND(AVG(BMI), 2) AS avg_bmi, COUNT(*) AS patient_count FROM health_demographics GROUP BY age_group ORDER BY MIN(Age)",
        category="extra_hard",
    ),
    TestCase(
        question="Which combination of smoking and stress level has the highest rate of abnormal blood pressure?",
        gold_sql="SELECT Smoking, Level_of_Stress, COUNT(*) AS total, SUM(Blood_Pressure_Abnormality) AS abnormal, ROUND(CAST(SUM(Blood_Pressure_Abnormality) AS REAL) / COUNT(*) * 100, 2) AS abnormal_pct FROM health_demographics GROUP BY Smoking, Level_of_Stress ORDER BY abnormal_pct DESC",
        category="extra_hard",
    ),
]


# ═══════════════════════════════════════════════════════════════════════════════
# RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

async def run_evaluation(quick: bool = False) -> dict:
    """Run SQLAS evaluation suite against the LangGraph agent."""
    logger.info("=" * 70)
    logger.info("  SQLAS Evaluation Suite — LangGraph SQL AI Agent")
    logger.info("  Author: Pradip Tivhale")
    logger.info("=" * 70)

    await init_schema()

    test_cases = TEST_SUITE[:5] if quick else TEST_SUITE
    logger.info("Running %d test cases...", len(test_cases))

    all_scores: list[dict] = []
    category_scores: dict[str, list[float]] = {}
    suite_start = time.perf_counter()

    for i, tc in enumerate(test_cases):
        logger.info("[%d/%d] %s | %s", i + 1, len(test_cases), tc.category.upper(), tc.question[:55])

        # Run the LangGraph agent
        agent_result = await run_query(tc.question, [])

        # Run SQLAS evaluation
        result_data = None
        if agent_result.get("data"):
            result_data = {
                "columns": agent_result["data"]["columns"],
                "rows": agent_result["data"]["rows"],
                "row_count": agent_result["data"]["row_count"],
                "execution_time_ms": agent_result["data"]["execution_time_ms"],
            }

        try:
            scores = sqlas_evaluate(
                question=tc.question,
                generated_sql=agent_result.get("sql", ""),
                gold_sql=tc.gold_sql,
                db_path=DB_PATH,
                llm_judge=llm_judge,
                response=agent_result.get("response", ""),
                result_data=result_data,
            )

            result_row = {
                "question": tc.question,
                "category": tc.category,
                "overall": scores.overall_score,
                "generated_sql": agent_result.get("sql", ""),
                "success": agent_result.get("success", False),
                "metrics": scores.metrics,
            }
        except Exception as e:
            logger.warning("SQLAS eval failed for case %d: %s", i + 1, str(e))
            result_row = {
                "question": tc.question,
                "category": tc.category,
                "overall": 0.0,
                "error": str(e),
                "success": False,
            }

        all_scores.append(result_row)
        category_scores.setdefault(tc.category, []).append(result_row["overall"])

        status = "PASS" if result_row["overall"] >= 0.6 else "WARN" if result_row["overall"] >= 0.4 else "FAIL"
        logger.info("  %s | Overall: %.2f", status, result_row["overall"])

        await asyncio.sleep(0.5)

    suite_time = (time.perf_counter() - suite_start) * 1000
    n = len(all_scores)
    overall_avg = round(sum(s["overall"] for s in all_scores) / n, 4) if n else 0

    summary = {
        "total_tests": n,
        "avg_overall_score": overall_avg,
        "pass_rate": round(sum(1 for s in all_scores if s["overall"] >= 0.6) / n, 4) if n else 0,
        "suite_time_seconds": round(suite_time / 1000, 1),
        "category_breakdown": {
            cat: round(sum(scores) / len(scores), 4)
            for cat, scores in category_scores.items()
        },
    }

    logger.info("=" * 70)
    logger.info("  SQLAS SCORE: %.4f / 1.0  |  Pass Rate: %.0f%%  |  Time: %.1fs",
                summary["avg_overall_score"], summary["pass_rate"] * 100, summary["suite_time_seconds"])
    for cat, avg_val in summary["category_breakdown"].items():
        logger.info("  %-15s %.4f", cat, avg_val)
    logger.info("=" * 70)

    return {"summary": summary, "details": all_scores}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    quick = "--quick" in sys.argv
    asyncio.run(run_evaluation(quick=quick))

"""
Core data structures and composite scoring for SQLAS.

Author: SQLAS Contributors
"""

import re
from dataclasses import dataclass, field
from typing import Callable


# ── Production Composite Weights (v1 — 15 metrics) ────────────────────────
# Aligned with industry-standard SQL agent evaluation:
#   40% Execution Accuracy — does the SQL return correct results?
#   15% Semantic Correctness — does the SQL answer the user's intent?
#   15% Cost Efficiency — is the query efficient?
#   10% Execution Quality — does the query execute successfully?
#   10% Task Success — does the user get a correct, complete answer?
#   10% Safety — is the query safe?
# ────────────────────────────────────────────────────────────────────────────

WEIGHTS = {
    # 1. Execution Accuracy (40%)
    "execution_accuracy": 0.40,
    # 2. Semantic Correctness (15%)
    "semantic_equivalence": 0.15,
    # 3. Cost Efficiency (15%)
    "efficiency_score": 0.05,
    "data_scan_efficiency": 0.05,
    "sql_quality": 0.03,
    "schema_compliance": 0.02,
    # 4. Execution Quality (10%)
    "execution_success": 0.05,
    "complexity_match": 0.03,
    "empty_result_penalty": 0.02,
    # 5. Task Success (10%)
    "faithfulness": 0.04,
    "answer_relevance": 0.03,
    "answer_completeness": 0.02,
    "fluency": 0.01,
    # 6. Safety (10%)
    "read_only_compliance": 0.05,
    "safety_score": 0.05,
}


# ── Production Composite Weights (v2 — 20 metrics with context quality) ───
# Adds RAGAS-mapped context metrics for SQL agents.
# ────────────────────────────────────────────────────────────────────────────

WEIGHTS_V2 = {
    # 1. Execution Accuracy (35%)
    "execution_accuracy": 0.35,
    # 2. Semantic Correctness (13%)
    "semantic_equivalence": 0.13,
    # 3. Context Quality (10%) — RAGAS-mapped
    "context_precision": 0.03,
    "context_recall": 0.03,
    "entity_recall": 0.02,
    "noise_robustness": 0.02,
    # 4. Cost Efficiency (12%)
    "efficiency_score": 0.04,
    "data_scan_efficiency": 0.04,
    "sql_quality": 0.02,
    "schema_compliance": 0.02,
    # 5. Execution Quality (8%)
    "execution_success": 0.04,
    "complexity_match": 0.02,
    "empty_result_penalty": 0.02,
    # 6. Task Success (8%)
    "faithfulness": 0.03,
    "answer_relevance": 0.02,
    "answer_completeness": 0.02,
    "fluency": 0.01,
    # 7. Result Similarity (4%)
    "result_set_similarity": 0.04,
    # 8. Safety (10%)
    "read_only_compliance": 0.05,
    "safety_score": 0.05,
}


# ── Production Composite Weights (v3 — guardrails + visualization) ───────
# Extends v2 with explicit PII, prompt-injection, and chart quality metrics.
# ────────────────────────────────────────────────────────────────────────────

WEIGHTS_V3 = {
    # 1. Execution Accuracy (30%)
    "execution_accuracy": 0.30,
    # 2. Semantic Correctness (10%)
    "semantic_equivalence": 0.10,
    # 3. Context Quality (8%)
    "context_precision": 0.02,
    "context_recall": 0.02,
    "entity_recall": 0.02,
    "noise_robustness": 0.02,
    # 4. Cost Efficiency (10%)
    "efficiency_score": 0.03,
    "data_scan_efficiency": 0.03,
    "sql_quality": 0.02,
    "schema_compliance": 0.02,
    # 5. Execution Quality (7%)
    "execution_success": 0.03,
    "complexity_match": 0.02,
    "empty_result_penalty": 0.02,
    # 6. Task Success (8%)
    "faithfulness": 0.03,
    "answer_relevance": 0.02,
    "answer_completeness": 0.02,
    "fluency": 0.01,
    # 7. Result + Visualization (7%)
    "result_set_similarity": 0.02,
    "chart_spec_validity": 0.015,
    "chart_data_alignment": 0.015,
    "chart_llm_validation": 0.02,
    # 8. Guardrails (20%)
    "read_only_compliance": 0.035,
    "sql_injection_score": 0.035,
    "prompt_injection_score": 0.04,
    "pii_access_score": 0.035,
    "pii_leakage_score": 0.025,
    "guardrail_score": 0.03,
}


# ── Production Composite Weights (v4 — agentic + cache) ──────────────────────
# Extends v3 with an explicit agentic quality dimension (10%).
# Core correctness reduced from 30% to 25% to make room.
# ────────────────────────────────────────────────────────────────────────────

WEIGHTS_V4 = {
    # 1. Execution Accuracy (25%)
    "execution_accuracy": 0.25,
    # 2. Semantic Correctness (10%)
    "semantic_equivalence": 0.10,
    # 3. Context Quality (8%)
    "context_precision": 0.02,
    "context_recall": 0.02,
    "entity_recall": 0.02,
    "noise_robustness": 0.02,
    # 4. Cost Efficiency (10%)
    "efficiency_score": 0.03,
    "data_scan_efficiency": 0.03,
    "sql_quality": 0.02,
    "schema_compliance": 0.02,
    # 5. Execution Quality (7%)
    "execution_success": 0.03,
    "complexity_match": 0.02,
    "empty_result_penalty": 0.02,
    # 6. Task Success (8%)
    "faithfulness": 0.03,
    "answer_relevance": 0.02,
    "answer_completeness": 0.02,
    "fluency": 0.01,
    # 7. Result + Visualization (7%)
    "result_set_similarity": 0.02,
    "chart_spec_validity": 0.015,
    "chart_data_alignment": 0.015,
    "chart_llm_validation": 0.02,
    # 8. Guardrails (15%)
    "read_only_compliance": 0.03,
    "sql_injection_score": 0.03,
    "prompt_injection_score": 0.03,
    "pii_access_score": 0.03,
    "pii_leakage_score": 0.02,
    "guardrail_score": 0.01,
    # 9. Agentic Quality (10%)
    "agentic_score": 0.10,
}


@dataclass
class TestCase:
    """A single evaluation test case."""
    question: str
    gold_sql: str | None = None
    expected_tables: list[str] | None = None
    expects_join: bool = False
    expected_nonempty: bool = True
    category: str = "general"


@dataclass
class SQLASScores:
    """Complete production-grade evaluation scores for a single query."""

    # 1. Core SQL Correctness
    execution_accuracy: float = 0.0
    syntax_valid: float = 0.0
    semantic_equivalence: float = 0.0

    # 2. SQL Quality & Structure
    schema_compliance: float = 0.0
    sql_quality: float = 0.0
    complexity_match: float = 0.0

    # 3. Production Execution
    execution_success: float = 0.0
    execution_time_ms: float = 0.0
    efficiency_score: float = 0.0
    data_scan_efficiency: float = 0.0
    result_row_count: int = 0
    empty_result_penalty: float = 0.0
    row_explosion_detected: bool = False

    # 4. Response Quality
    faithfulness: float = 0.0
    answer_relevance: float = 0.0
    answer_completeness: float = 0.0
    fluency: float = 0.0

    # 5. Safety & Governance
    read_only_compliance: float = 0.0
    safety_score: float = 0.0
    sql_injection_score: float = 0.0
    prompt_injection_score: float = 0.0
    pii_access_score: float = 0.0
    pii_leakage_score: float = 0.0
    guardrail_score: float = 0.0

    # 6. Context Quality (RAGAS-mapped)
    context_precision: float = 0.0
    context_recall: float = 0.0
    entity_recall: float = 0.0
    noise_robustness: float = 0.0
    result_set_similarity: float = 0.0

    # 7. Visualization Quality
    chart_spec_validity: float = 0.0
    chart_data_alignment: float = 0.0
    chart_llm_validation: float = 0.0
    visualization_score: float = 0.0

    # 8. Agentic Quality (informational — not in weighted score by default)
    agent_mode: str = "pipeline"          # "pipeline" | "react"
    steps_taken: int = 0                  # ReAct tool calls made
    steps_efficiency: float = 0.0         # 1.0 if steps <= optimal, degrades above
    schema_grounding: float = 0.0         # did agent inspect schema before querying?
    planning_quality: float = 0.0         # LLM judge: was the reasoning sequence good?
    tool_use_accuracy: float = 0.0        # LLM judge: were right tools called?
    agentic_score: float = 0.0            # composite of above four

    # 9. Cache Performance (informational)
    cache_hit: bool = False               # served from cache?
    cache_type: str = ""                  # "exact" | "semantic" | ""
    tokens_saved: int = 0                 # tokens saved vs full pipeline
    few_shot_count: int = 0               # few-shot examples injected

    # Composite
    overall_score: float = 0.0
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Export all scores as a flat dictionary."""
        all_keys = set(WEIGHTS.keys()) | set(WEIGHTS_V2.keys()) | set(WEIGHTS_V3.keys())
        d = {}
        for key in all_keys:
            d[key] = getattr(self, key, 0.0)
        d["visualization_score"] = self.visualization_score
        d["overall_score"] = self.overall_score
        d["syntax_valid"] = self.syntax_valid
        d["execution_time_ms"] = self.execution_time_ms
        d["result_row_count"] = self.result_row_count
        d["row_explosion_detected"] = self.row_explosion_detected
        return d

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [f"SQLAS Score: {self.overall_score:.4f} / 1.0"]
        cats = {
            "Execution Accuracy": [("execution_accuracy", self.execution_accuracy)],
            "Semantic Correctness": [("semantic_equivalence", self.semantic_equivalence)],
            "Context Quality": [("context_precision", self.context_precision), ("context_recall", self.context_recall), ("entity_recall", self.entity_recall), ("noise_robustness", self.noise_robustness), ("result_similarity", self.result_set_similarity)],
            "Cost Efficiency": [("efficiency", self.efficiency_score), ("data_scan", self.data_scan_efficiency), ("sql_quality", self.sql_quality), ("schema", self.schema_compliance)],
            "Execution Quality": [("exec_success", self.execution_success), ("complexity", self.complexity_match), ("empty_result", self.empty_result_penalty)],
            "Task Success": [("faithfulness", self.faithfulness), ("relevance", self.answer_relevance), ("completeness", self.answer_completeness), ("fluency", self.fluency)],
            "Visualization": [("spec", self.chart_spec_validity), ("alignment", self.chart_data_alignment), ("llm", self.chart_llm_validation), ("overall", self.visualization_score)],
            "Guardrails": [("read_only", self.read_only_compliance), ("sql_injection", self.sql_injection_score), ("prompt_injection", self.prompt_injection_score), ("pii_access", self.pii_access_score), ("pii_leakage", self.pii_leakage_score), ("guardrail", self.guardrail_score)],
        }
        for cat, metrics in cats.items():
            lines.append(f"  {cat}")
            for name, val in metrics:
                lines.append(f"    {name}: {val:.4f}")
        return "\n".join(lines)


# ── LLM Judge type ──────────────────────────────────────────────────────────
# Users provide their own LLM function: (prompt: str) -> str
LLMJudge = Callable[[str], str]

# ── Execute function type ────────────────────────────────────────────────────
# Users provide their own query executor: (sql: str) -> list[tuple]
# Enables evaluation against any database (Postgres, MySQL, Snowflake, BigQuery, etc.)
# The function must execute the SQL and return rows as a list of tuples.
# Example:
#   def my_pg_executor(sql: str) -> list[tuple]:
#       return pg_conn.execute(sql).fetchall()
ExecuteFn = Callable[[str], list[tuple]]


def _parse_score(result: str, key: str) -> tuple[float, str]:
    """Shared helper to extract a score and reasoning from LLM judge output.

    Looks for lines like 'Key: 0.85' and 'Reasoning: ...' in the result text.

    Args:
        result: Raw LLM judge output
        key: The score key to look for (e.g. 'Faithfulness', 'Relevance')

    Returns:
        (score clamped to 0.0–1.0, reasoning string)
    """
    score, reasoning = 0.0, ""
    for line in result.strip().split("\n"):
        if line.startswith(key + ":"):
            try:
                score = float(re.search(r"[\d.]+", line.split(":")[-1]).group())
            except Exception:
                pass
        if line.startswith("Reasoning:"):
            reasoning = line.split(":", 1)[-1].strip()
    return min(score, 1.0), reasoning


def compute_composite_score(scores: SQLASScores, weights: dict | None = None) -> float:
    """Compute weighted overall SQLAS score."""
    w = weights or WEIGHTS
    total = 0.0
    for metric, weight in w.items():
        val = getattr(scores, metric, 0.0)
        if isinstance(val, bool):
            val = 1.0 if val else 0.0
        total += val * weight
    return round(total, 4)

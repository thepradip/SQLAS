"""
Response Quality Metrics (LLM-as-Judge).
- Faithfulness (claims grounded in data)
- Answer Relevance (answers the question)
- Answer Completeness (all key data surfaced)
- Fluency (readability)

Author: SQLAS Contributors
"""

import re
import logging

from sqlas.core import LLMJudge, _parse_score

logger = logging.getLogger(__name__)


def faithfulness(
    question: str,
    response: str,
    sql_result_preview: str,
    llm_judge: LLMJudge,
) -> tuple[float, dict]:
    """
    RAGAS Faithfulness for SQL agents.
    Checks if every claim in the response is supported by the SQL result data.
    """
    prompt = f"""You are an evaluation judge. Assess FAITHFULNESS of this response.

**Task:** Check if EVERY factual claim is supported by the SQL Result data.

**Question:** {question}
**SQL Result:** {sql_result_preview}
**Response:** {response}

List claims, mark SUPPORTED/UNSUPPORTED, compute faithfulness = supported/total.

Respond EXACTLY:
Faithfulness: [score 0.0-1.0]
Reasoning: [one sentence]"""

    try:
        result = llm_judge(prompt)
    except Exception as e:
        logger.warning("LLM judge failed in faithfulness: %s", e)
        return 0.0, {"error": str(e)}

    score, reasoning = _parse_score(result, "Faithfulness")
    return score, {"reasoning": reasoning}


def answer_relevance(
    question: str,
    response: str,
    llm_judge: LLMJudge,
) -> tuple[float, dict]:
    """Does the response directly answer the user's question? (0.0-1.0)"""
    prompt = f"""Assess RELEVANCE. Does the response answer the question?

**Question:** {question}
**Response:** {response}

Score 0.0-1.0 (1.0 = perfectly relevant, 0.0 = off-topic).

Respond EXACTLY:
Relevance: [score]
Reasoning: [one sentence]"""

    try:
        result = llm_judge(prompt)
    except Exception as e:
        logger.warning("LLM judge failed in answer_relevance: %s", e)
        return 0.0, {"error": str(e)}

    score, reasoning = _parse_score(result, "Relevance")
    return score, {"reasoning": reasoning}


def answer_completeness(
    question: str,
    response: str,
    sql_result_preview: str,
    llm_judge: LLMJudge,
) -> tuple[float, dict]:
    """Did the response surface ALL key information from the SQL result? (0.0-1.0)"""
    prompt = f"""Assess COMPLETENESS. Are all key data points from the result mentioned?

**Question:** {question}
**SQL Result:** {sql_result_preview}
**Response:** {response}

Score 0.0-1.0 (1.0 = all key points covered, 0.0 = most omitted).

Respond EXACTLY:
Completeness: [score]
Reasoning: [one sentence]"""

    try:
        result = llm_judge(prompt)
    except Exception as e:
        logger.warning("LLM judge failed in answer_completeness: %s", e)
        return 0.0, {"error": str(e)}

    score, reasoning = _parse_score(result, "Completeness")
    return score, {"reasoning": reasoning}


def fluency(response: str, llm_judge: LLMJudge) -> tuple[float, dict]:
    """Readability and coherence (1-5 normalized to 0.0-1.0)."""
    prompt = f"""Rate fluency of this text 1-5.

**Text:** {response[:1000]}

1=Incoherent, 2=Awkward, 3=Acceptable, 4=Good, 5=Excellent

Respond EXACTLY:
Fluency: [score 1-5]"""

    try:
        result = llm_judge(prompt)
    except Exception as e:
        logger.warning("LLM judge failed in fluency: %s", e)
        return 0.0, {"error": str(e)}

    score = 3.0
    for line in result.strip().split("\n"):
        if line.startswith("Fluency:"):
            try:
                score = float(re.search(r"[\d.]+", line.split(":")[-1]).group())
            except Exception:
                pass
    return round(min(score, 5.0) / 5.0, 2), {"raw_score": score}

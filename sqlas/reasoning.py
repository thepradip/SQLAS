"""
Reasoning Quality Metrics.
- Intent decomposition / plan correctness
- Temporal reasoning (date/time expression translation)
- Null handling correctness
- Result explainability (assumptions, filters, caveats)
- Ambiguity handling (clarification vs silent assumption)

Author: SQLAS Contributors
"""

import re
import logging

from sqlas.core import LLMJudge, _parse_score, _retry_llm_judge

logger = logging.getLogger(__name__)

# Temporal patterns — fast pre-check before calling LLM
_TEMPORAL_SQL_PATTERNS = [
    (r"\bDATEADD\b|\bDATE_ADD\b|\bADD_MONTHS\b", "date_arithmetic"),
    (r"\bDATEDIFF\b|\bDATE_DIFF\b|\bTIMESTAMPDIFF\b", "date_diff"),
    (r"\bDATE_TRUNC\b|\bTIMESTAMP_TRUNC\b", "date_trunc"),
    (r"\bEXTRACT\b.*\bFROM\b|\bDATE_PART\b", "date_extract"),
    (r"\bBETWEEN\b.*'\d{4}-", "date_range"),
    (r"\bGETDATE\b|\bCURRENT_DATE\b|\bNOW\(\)|\bSYSDATE\b|\bCURRENT_TIMESTAMP\b", "current_date_fn"),
    (r"\bFISCAL\b|\bFY\d{2}\b", "fiscal_period"),
    (r"\bQUARTER\b|\bQTD\b|\bMTD\b|\bYTD\b", "period_abbrev"),
    (r"\bAT TIME ZONE\b|\bCONVERT_TIMEZONE\b", "timezone"),
]

_TEMPORAL_QUESTION_PATTERNS = [
    r"\blast\s+\d+\s+(day|week|month|year|quarter)s?",
    r"\bthis\s+(week|month|year|quarter)\b",
    r"\bprevious\s+(week|month|year|quarter)\b",
    r"\byesterday\b",
    r"\btoday\b",
    r"\brecent\b",
    r"\bcurrent\s+(week|month|year|quarter)\b",
    r"\bqtd\b|\bmtd\b|\bytd\b",
    r"\bfiscal\b",
    r"\btrailing\b|\brolling\b",
    r"\bover\s+the\s+last\b",
    r"\bsince\b.*\d{4}",
    r"\bbetween\b.*and\b.*\d{4}",
    r"\btimezone\b|\butc\b|\bpst\b|\best\b|\bcst\b|\bmst\b",
    r"\bmonthly\b|\bweekly\b|\bdaily\b|\bquarterly\b|\bannually\b",
]


def intent_decomposition_score(
    question: str,
    sql: str,
    llm_judge: LLMJudge,
    schema_context: str = "",
) -> "tuple[float, dict]":
    """
    LLM judge: does the SQL capture the correct business intent of the question?

    Catches cases where SQL executes without error but answers a different analytical
    question — common in multi-step or compound questions.

    Scoring:
        1.0  SQL perfectly captures the business intent.
        0.7  Minor gap (slightly wrong metric or missing condition).
        0.4  SQL runs but answers a different question.
        0.0  SQL completely misses the intent.

    Args:
        question:       Original user question.
        sql:            Generated SQL.
        llm_judge:      LLM judge function.
        schema_context: Optional schema context (truncated to 800 chars).
    """
    schema_block = f"\n**Schema context:**\n{schema_context[:800]}" if schema_context else ""

    prompt = f"""You are an expert analyst evaluating whether a SQL query correctly captures the business intent of a question.

**User Question:** {question}

**Generated SQL:**
```sql
{sql}
```
{schema_block}

Evaluate:
1. Does the SQL address the exact analytical question asked?
2. Are the correct metrics/dimensions selected (not a proxy or approximation)?
3. For multi-part questions, are all sub-intents addressed?

Score 0.0-1.0:
- 1.0: SQL perfectly captures the business intent
- 0.7: Minor gaps (slightly wrong metric or missing condition)
- 0.4: SQL runs but answers a different question
- 0.0: SQL completely misses the intent

Respond EXACTLY:
Intent_Score: [score]
Intent_Captured: [YES/PARTIAL/NO]
Reasoning: [one sentence]"""

    try:
        result = _retry_llm_judge(llm_judge, prompt)
    except Exception as e:
        logger.warning("LLM judge failed in intent_decomposition_score: %s", e)
        return 0.0, {"error": str(e), "llm_error": True}

    score, reasoning, parse_ok = _parse_score(result, "Intent_Score")
    intent_captured = ""
    for line in result.strip().split("\n"):
        if line.startswith("Intent_Captured:"):
            intent_captured = line.split(":", 1)[-1].strip()

    details: dict = {"reasoning": reasoning, "intent_captured": intent_captured}
    if not parse_ok:
        details["llm_parse_warning"] = True
    return score, details


def temporal_reasoning_score(
    question: str,
    sql: str,
    llm_judge: LLMJudge,
) -> "tuple[float, dict]":
    """
    Evaluate whether temporal expressions in the question are correctly translated to SQL.

    Returns 1.0 immediately when no temporal terms are found in the question.
    When temporal terms are present, invokes the LLM judge to evaluate:
    - Date arithmetic (last 30 days, last quarter, YTD, MTD, fiscal year)
    - Date grain (daily/weekly/monthly/quarterly aggregation)
    - Boundary conditions (inclusive vs exclusive)
    - Timezone handling
    - Dynamic date functions vs hardcoded dates

    Args:
        question:  Original user question.
        sql:       Generated SQL.
        llm_judge: LLM judge function.
    """
    q_lower = question.lower()
    temporal_found = [p for p in _TEMPORAL_QUESTION_PATTERNS if re.search(p, q_lower)]

    if not temporal_found:
        return 1.0, {"note": "no temporal expressions detected in question", "scored": False}

    upper = sql.upper()
    sql_has_date_logic = any(re.search(p, upper) for p, _ in _TEMPORAL_SQL_PATTERNS)
    has_hardcoded_dates = bool(re.search(r"'\d{4}-\d{2}-\d{2}'", sql))

    prompt = f"""You are a SQL expert evaluating temporal reasoning accuracy.

**User Question:** {question}

**Generated SQL:**
```sql
{sql}
```

The question contains temporal expressions.

Evaluate:
1. Are all date/time ranges correctly computed (last month, YTD, fiscal year, etc.)?
2. Is the date grain correct (daily vs monthly aggregation, etc.)?
3. Are date boundaries inclusive/exclusive as expected?
4. Is timezone handling correct if mentioned?
5. Does the SQL use dynamic date functions (preferred) vs hardcoded dates (risky)?

Score 0.0-1.0:
- 1.0: Temporally correct — right period, right grain, right boundaries
- 0.7: Minor issue (off-by-one boundary, calendar vs fiscal year)
- 0.4: Wrong period (last year vs last 12 months, wrong grain)
- 0.0: Temporal logic completely wrong or missing

Respond EXACTLY:
Temporal_Score: [score]
Issues: [list or "none"]
Reasoning: [one sentence]"""

    try:
        result = _retry_llm_judge(llm_judge, prompt)
    except Exception as e:
        logger.warning("LLM judge failed in temporal_reasoning_score: %s", e)
        return 0.0, {"error": str(e), "llm_error": True}

    score, reasoning, parse_ok = _parse_score(result, "Temporal_Score")
    issues = "none"
    for line in result.strip().split("\n"):
        if line.startswith("Issues:"):
            issues = line.split(":", 1)[-1].strip()

    details: dict = {
        "reasoning": reasoning,
        "issues": issues,
        "sql_has_date_logic": sql_has_date_logic,
        "has_hardcoded_dates": has_hardcoded_dates,
        "scored": True,
    }
    if not parse_ok:
        details["llm_parse_warning"] = True
    return score, details


def null_handling_score(
    sql: str,
    llm_judge: LLMJudge,
    question: str = "",
) -> "tuple[float, dict]":
    """
    Evaluate whether the SQL correctly handles NULL values.

    NULL handling is a common source of silent correctness bugs:
    - COUNT(*) vs COUNT(col): COUNT(col) silently drops NULLs
    - AVG: ignores NULLs, changing the effective denominator
    - = NULL vs IS NULL: = NULL always evaluates to UNKNOWN
    - Outer join NULLs silently excluded by downstream WHERE filters
    - SUM(CASE ... END) without ELSE defaults to NULL not zero

    Heuristic pre-check avoids LLM call when no null-sensitive patterns exist.

    Args:
        sql:       Generated SQL.
        llm_judge: LLM judge function.
        question:  Optional user question for context.
    """
    upper = sql.upper()

    null_risks: list[str] = []

    if re.search(r"\bAVG\s*\(", upper):
        null_risks.append("AVG ignores NULLs — denominator is count of non-null values only")
    if re.search(r"\bCOUNT\s*\(\s*[^*\s]", upper):
        null_risks.append("COUNT(col) ignores NULLs — verify intent vs COUNT(*)")
    if re.search(r"\b(LEFT|RIGHT|FULL OUTER)\s+JOIN\b", upper):
        null_risks.append("outer join produces NULLs in unmatched rows")
    if re.search(r"=\s*NULL\b", upper):
        null_risks.append("= NULL always evaluates to UNKNOWN — use IS NULL")
    if re.search(r"(!=|<>)\s*NULL\b", upper):
        null_risks.append("!= NULL always evaluates to UNKNOWN — use IS NOT NULL")
    if re.search(r"\bSUM\s*\(\s*CASE\b", upper) and "ELSE" not in upper:
        null_risks.append("SUM(CASE ... END) without ELSE — missing ELSE defaults to NULL")

    if not null_risks:
        return 1.0, {"note": "no null-sensitive patterns detected", "scored": False}

    # Critical: = NULL is always wrong — short-circuit with low score before LLM
    has_critical_null_error = bool(re.search(r"(=|!=|<>)\s*NULL\b", upper))
    if has_critical_null_error:
        return 0.0, {
            "issue": "= NULL or != NULL comparison — always evaluates to UNKNOWN, use IS NULL / IS NOT NULL",
            "null_risks": null_risks,
            "heuristic_short_circuit": True,
        }

    question_block = f"\n**User Question:** {question}" if question else ""

    prompt = f"""You are a SQL expert evaluating NULL handling correctness.
{question_block}

**Generated SQL:**
```sql
{sql}
```

Detected potentially null-sensitive patterns:
{chr(10).join(f'- {r}' for r in null_risks)}

For each pattern, evaluate whether the SQL handles it correctly:
1. AVG: does the question expect nulls excluded, or is this a silent precision bug?
2. COUNT(col): is the intent to count non-null values, or all rows?
3. Outer JOINs: are NULLs from unmatched rows handled correctly downstream?
4. CASE without ELSE: is the missing default intentional?

Score 0.0-1.0:
- 1.0: All null-sensitive patterns handled correctly
- 0.7: Minor null issue unlikely to affect results
- 0.4: Null handling bug that silently changes aggregate results
- 0.0: Critical null error (wrong denominator, null filter defeating the join intent)

Respond EXACTLY:
Null_Handling_Score: [score]
Issues: [list or "none"]
Reasoning: [one sentence]"""

    try:
        result = _retry_llm_judge(llm_judge, prompt)
    except Exception as e:
        logger.warning("LLM judge failed in null_handling_score: %s", e)
        return 0.7, {"error": str(e), "null_risks": null_risks, "heuristic_fallback": True}

    score, reasoning, parse_ok = _parse_score(result, "Null_Handling_Score")
    issues = "none"
    for line in result.strip().split("\n"):
        if line.startswith("Issues:"):
            issues = line.split(":", 1)[-1].strip()

    details: dict = {
        "reasoning": reasoning,
        "issues": issues,
        "null_risks_detected": null_risks,
        "scored": True,
    }
    if not parse_ok:
        details["llm_parse_warning"] = True
    return score, details


def result_explainability_score(
    question: str,
    sql: str,
    response: str,
    llm_judge: LLMJudge,
) -> "tuple[float, dict]":
    """
    Evaluate whether the agent's response explains assumptions, filters, and caveats.

    Explainability is distinct from answer fluency — a fluent response can still hide
    important caveats (date ranges, NULL exclusions, filter assumptions, approximations).
    For simple questions with no meaningful assumptions, 0.8+ is appropriate.

    Args:
        question:  Original user question.
        sql:       Generated SQL.
        response:  Natural-language response from the agent.
        llm_judge: LLM judge function.
    """
    if not response:
        return 0.0, {"note": "no response provided"}

    prompt = f"""You are evaluating whether an AI agent's response is transparent about its SQL assumptions.

**User Question:** {question}

**Generated SQL:**
```sql
{sql}
```

**Agent's Response:**
{response[:1000]}

Evaluate whether the response:
1. Explains key filters applied (date ranges, status filters, exclusions)
2. Mentions data limitations or caveats (nulls excluded, snapshot date, estimates)
3. Clarifies any assumptions made in interpreting the question
4. Provides context that helps the user trust or verify the answer

Note: For simple clear questions with no meaningful assumptions, 0.8+ is appropriate.

Score 0.0-1.0:
- 1.0: All relevant assumptions and caveats clearly explained
- 0.7: Mentions most assumptions but misses one minor caveat
- 0.4: Gives a number but hides important filter assumptions
- 0.0: No explanation of any assumptions — pure opaque output

Respond EXACTLY:
Explainability_Score: [score]
Missing_Caveats: [list or "none"]
Reasoning: [one sentence]"""

    try:
        result = _retry_llm_judge(llm_judge, prompt)
    except Exception as e:
        logger.warning("LLM judge failed in result_explainability_score: %s", e)
        return 0.0, {"error": str(e), "llm_error": True}

    score, reasoning, parse_ok = _parse_score(result, "Explainability_Score")
    missing = "none"
    for line in result.strip().split("\n"):
        if line.startswith("Missing_Caveats:"):
            missing = line.split(":", 1)[-1].strip()

    details: dict = {"reasoning": reasoning, "missing_caveats": missing}
    if not parse_ok:
        details["llm_parse_warning"] = True
    return score, details


def ambiguity_handling_score(
    question: str,
    sql: str,
    response: str,
    llm_judge: LLMJudge,
) -> "tuple[float, dict]":
    """
    Evaluate whether the agent correctly handled ambiguous questions.

    An agent should either:
    (a) Ask a clarifying question when intent is genuinely ambiguous, or
    (b) State the assumption made when choosing one interpretation.

    Silently choosing an interpretation without acknowledgment is a failure mode
    that can cause invisible correctness errors.

    Scoring:
        1.0  Question is clear (no ambiguity required), or agent asked for clarification,
             or agent explicitly stated its assumption.
        0.7  Agent made a reasonable assumption but stated it unclearly.
        0.4  Agent silently chose a defensible interpretation.
        0.0  Agent silently chose a wrong or non-obvious interpretation.

    Args:
        question:  Original user question.
        sql:       Generated SQL.
        response:  Natural-language response from the agent.
        llm_judge: LLM judge function.
    """
    if not response:
        return 0.0, {"note": "no response provided"}

    prompt = f"""You are evaluating how an AI SQL agent handles question ambiguity.

**User Question:** {question}

**Generated SQL:**
```sql
{sql}
```

**Agent's Response:**
{response[:1000]}

First, assess: is this question ambiguous? (could it be answered in 2+ substantially different ways?)

If NOT ambiguous: return 1.0 — handling ambiguity is not required for clear questions.

If AMBIGUOUS, evaluate whether the agent:
1. Asked a clarifying question (best for high-stakes ambiguity)
2. Stated which interpretation was chosen and why (acceptable)
3. Silently chose one interpretation without acknowledgment (bad)
4. Presented multiple interpretations (good for exploration)

Score 0.0-1.0:
- 1.0: Question was clear, OR agent properly asked for clarification / stated assumption
- 0.7: Agent made a reasonable assumption but stated it unclearly
- 0.4: Agent silently chose a defensible interpretation
- 0.0: Agent silently chose a wrong or non-obvious interpretation

Respond EXACTLY:
Ambiguity_Score: [score]
Is_Ambiguous: [YES/NO]
Handling: [CLARIFICATION_ASKED/ASSUMPTION_STATED/SILENT_CHOICE/NOT_NEEDED]
Reasoning: [one sentence]"""

    try:
        result = _retry_llm_judge(llm_judge, prompt)
    except Exception as e:
        logger.warning("LLM judge failed in ambiguity_handling_score: %s", e)
        return 0.0, {"error": str(e), "llm_error": True}

    score, reasoning, parse_ok = _parse_score(result, "Ambiguity_Score")
    is_ambiguous = handling = ""
    for line in result.strip().split("\n"):
        if line.startswith("Is_Ambiguous:"):
            is_ambiguous = line.split(":", 1)[-1].strip()
        elif line.startswith("Handling:"):
            handling = line.split(":", 1)[-1].strip()

    details: dict = {
        "reasoning": reasoning,
        "is_ambiguous": is_ambiguous,
        "handling": handling,
    }
    if not parse_ok:
        details["llm_parse_warning"] = True
    return score, details

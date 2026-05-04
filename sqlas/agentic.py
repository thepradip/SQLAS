"""
Agentic quality metrics for ReAct-style SQL agents.

These metrics evaluate HOW the agent reasoned, not just what it produced.
They are informational — not included in the core weighted score by default,
but available as a separate agentic score or via WEIGHTS_V4.

Metrics:
  steps_efficiency    — was the step count optimal?
  schema_grounding    — did the agent inspect schema before querying?
  planning_quality    — LLM judge on reasoning sequence quality
  tool_use_accuracy   — did the agent use the right tools?
"""

from sqlas.core import LLMJudge, _parse_score


def steps_efficiency(steps_taken: int, optimal_steps: int = 3) -> float:
    """
    Score based on how many ReAct steps the agent used.

    steps_taken = 0 means pipeline mode — returns 1.0 (not penalised).
    Above optimal_steps the score degrades linearly.

    Args:
        steps_taken:   Number of tool calls made in the ReAct loop.
        optimal_steps: Steps considered ideal (default 3: list→describe→execute).

    Returns:
        Float 0.0–1.0 efficiency score.
    """
    if steps_taken == 0:
        return 1.0              # pipeline mode — no steps to penalise
    if steps_taken <= optimal_steps:
        return 1.0
    if steps_taken <= optimal_steps + 2:
        return 0.8
    if steps_taken <= optimal_steps + 4:
        return 0.6
    return 0.3


def schema_grounding(steps: list[dict]) -> float:
    """
    Did the agent inspect the schema before writing SQL?

    Checks whether describe_table or list_tables was called
    at least once before the first execute_sql call.

    Args:
        steps: List of step dicts with "tool" key, in execution order.

    Returns:
        1.0 — schema inspected before querying (good)
        0.5 — SQL executed without prior schema inspection
        0.0 — no steps (no data to evaluate)
    """
    if not steps:
        return 0.0

    tools = [s.get("tool", "") for s in steps]
    execute_pos   = [i for i, t in enumerate(tools) if t == "execute_sql"]
    inspect_pos   = [i for i, t in enumerate(tools) if t in ("describe_table", "list_tables")]

    if not execute_pos:
        return 0.5   # agent ran but never executed SQL
    if not inspect_pos:
        return 0.5   # agent jumped straight to SQL without schema check

    return 1.0 if min(inspect_pos) < min(execute_pos) else 0.3


def planning_quality(
    question: str,
    steps: list[dict],
    llm_judge: LLMJudge,
) -> tuple[float, dict]:
    """
    LLM judge evaluates the quality of the agent's reasoning sequence.

    Only meaningful for ReAct mode (steps non-empty).
    For pipeline mode, returns (0.0, {"note": "pipeline mode"}).

    Args:
        question:  Original user question.
        steps:     ReAct step list — each dict should have "tool" and "args".
        llm_judge: LLM judge function (prompt: str) -> str.

    Returns:
        (score 0.0–1.0, details dict)
    """
    if not steps:
        return 0.0, {"note": "pipeline mode — no planning steps to evaluate"}

    step_summary = "\n".join(
        f"Step {i + 1}: {s.get('tool', '?')}({list(s.get('args', {}).keys())})"
        for i, s in enumerate(steps)
    )

    prompt = f"""You are evaluating an AI SQL agent's planning quality.

User question: "{question}"

Steps the agent took:
{step_summary}

Evaluate:
1. Did the agent inspect the schema before writing SQL?
2. Were the steps logically ordered and non-redundant?
3. Did the agent avoid wasted or repeated tool calls?

Score 0.0–1.0:
- 1.0: Perfect — schema inspected first, minimal efficient steps
- 0.7: Good — minor inefficiencies, correct overall flow
- 0.4: Acceptable — some wasted steps or schema skipped
- 0.0: Poor — SQL attempted with no schema context, many retries

Respond EXACTLY:
Planning_Quality: [score]
Reasoning: [one sentence]"""

    result = llm_judge(prompt)
    score, reasoning = _parse_score(result, "Planning_Quality")
    return score, {"reasoning": reasoning, "steps_count": len(steps)}


def tool_use_accuracy(
    question: str,
    steps: list[dict],
    llm_judge: LLMJudge,
) -> tuple[float, dict]:
    """
    LLM judge: did the agent call the right tools with appropriate arguments?

    Args:
        question:  Original user question.
        steps:     ReAct step list.
        llm_judge: LLM judge function.

    Returns:
        (score 0.0–1.0, details dict)
    """
    if not steps:
        return 0.0, {"note": "pipeline mode"}

    step_detail = "\n".join(
        f"Step {i + 1}: {s.get('tool')}  args={s.get('args', {})}"
        for i, s in enumerate(steps)
    )

    prompt = f"""Evaluate whether an AI SQL agent used its tools correctly.

User question: "{question}"

Tool calls made:
{step_detail}

Available tools: list_tables, describe_table, execute_sql, final_answer

Evaluate:
1. Were the right tools called for each step?
2. Were the arguments (table names, SQL) appropriate?
3. Did the agent call final_answer with a proper SQL-backed response?

Score 0.0–1.0:
- 1.0: All tool calls were correct and appropriate
- 0.7: Mostly correct with minor argument issues
- 0.4: Some wrong tools or bad arguments
- 0.0: Mostly wrong tool choices

Respond EXACTLY:
Tool_Use_Accuracy: [score]
Reasoning: [one sentence]"""

    result = llm_judge(prompt)
    score, reasoning = _parse_score(result, "Tool_Use_Accuracy")
    return score, {"reasoning": reasoning}


def agentic_score(
    question: str,
    steps: list[dict],
    llm_judge: LLMJudge,
    optimal_steps: int = 3,
) -> tuple[float, dict]:
    """
    Composite agentic quality score.

    Combines steps_efficiency, schema_grounding, and planning_quality.
    Weights: 30% efficiency + 30% schema grounding + 40% planning quality.

    Args:
        question:      Original user question.
        steps:         ReAct step list.
        llm_judge:     LLM judge function.
        optimal_steps: Steps considered ideal.

    Returns:
        (score 0.0–1.0, details dict)
    """
    eff = steps_efficiency(len(steps), optimal_steps)
    grnd = schema_grounding(steps)
    plan, plan_details = planning_quality(question, steps, llm_judge)

    score = round(0.30 * eff + 0.30 * grnd + 0.40 * plan, 4)
    return score, {
        "steps_efficiency": eff,
        "schema_grounding": grnd,
        "planning_quality": plan,
        "planning_reasoning": plan_details.get("reasoning", ""),
        "steps_taken": len(steps),
        "agent_mode": "react" if steps else "pipeline",
    }


def plan_compliance(steps: list[dict]) -> tuple[float, dict]:
    """
    Did the agent follow the mandatory planning protocol?

    Evaluates whether create_plan was called BEFORE execute_sql and
    whether describe_table was called for schema inspection.
    This metric directly measures the effectiveness of plan enforcement —
    the feature added to prevent first-attempt failures.

    Score:
        1.0 = create_plan before execute_sql + describe_table called (full compliance)
        0.7 = create_plan before execute_sql but no describe_table (partial)
        0.5 = plan created but no SQL executed
        0.0 = execute_sql called without prior create_plan (compliance failure)

    Args:
        steps: ReAct step list [{tool, args, result_preview}] in execution order.

    Returns:
        (score, details_dict)
    """
    if not steps:
        return 0.0, {"note": "pipeline mode — plan compliance not applicable"}

    tools = [s.get("tool", "") for s in steps]

    plan_idx    = next((i for i, t in enumerate(tools) if t == "create_plan"), None)
    exec_idx    = next((i for i, t in enumerate(tools) if t == "execute_sql"), None)
    described   = any(t == "describe_table" for t in tools)
    blocked     = any(t == "BLOCKED_execute_sql" for t in tools)

    if exec_idx is not None and plan_idx is None:
        return 0.0, {
            "plan_compliance": "FAIL",
            "issue": "execute_sql called without create_plan — planning was skipped",
            "blocked_attempts": blocked,
        }

    if plan_idx is not None and exec_idx is not None and plan_idx > exec_idx:
        return 0.0, {
            "plan_compliance": "FAIL",
            "issue": "create_plan called AFTER execute_sql — wrong order",
        }

    if plan_idx is None:
        return 0.5, {"plan_compliance": "NO_SQL", "note": "plan created but no SQL executed"}

    if not described:
        return 0.7, {
            "plan_compliance": "PARTIAL",
            "issue": "create_plan called but describe_table skipped — column names may be wrong",
        }

    return 1.0, {
        "plan_compliance": "PASS",
        "plan_before_sql":   True,
        "schema_inspected":  True,
        "blocked_attempts":  blocked,
    }


def first_attempt_success(agent_result: dict) -> tuple[float, dict]:
    """
    Did the agent generate correct SQL on the first attempt without retrying?

    Measures the combined effectiveness of:
      - create_plan enforcement (plan before acting)
      - Schema context quality (right tables + columns provided)
      - Few-shot examples from FeedbackStore

    A well-planned agent using accurate schema context succeeds first-time.
    Retries indicate planning gaps or context quality problems.

    Score:
        1.0 = succeeded with 0 retries
        0.7 = succeeded with 1 retry
        0.4 = succeeded with 2 retries
        0.0 = failed after max retries

    Args:
        agent_result: The dict returned by run_query() or run_react_query().

    Returns:
        (score, details_dict)
    """
    metrics     = agent_result.get("metrics") or {}
    success     = agent_result.get("success", False)
    retry_count = int(metrics.get("retry_count", 0))

    if not success:
        return 0.0, {
            "success":     False,
            "retry_count": retry_count,
            "note":        "Query failed — check SQL validity and schema context",
        }

    score = max(0.0, round(1.0 - retry_count * 0.3, 4))
    return score, {
        "success":     True,
        "retry_count": retry_count,
        "note":        "First attempt" if retry_count == 0 else f"Succeeded after {retry_count} retry/retries",
    }

"""
Visualization Quality Metrics.
- Deterministic chart payload validation
- LLM-as-judge chart relevance and commentary quality

Author: SQLAS Contributors
"""

import logging

from sqlas.core import LLMJudge, _parse_score

logger = logging.getLogger(__name__)

SUPPORTED_CHART_TYPES = {"number", "bar", "line", "pie", "table"}
CHART_TYPES_REQUIRING_SERIES = {"bar", "line", "pie"}


def chart_spec_validity(visualization: dict | None, result_data: dict | None = None) -> tuple[float, dict]:
    """Validate that a generated chart spec can be rendered against the SQL result."""
    if not visualization:
        return 1.0 if not result_data else 0.0, {"issues": ["missing_visualization"] if result_data else ["none"]}

    issues = []
    chart_type = visualization.get("type")

    if chart_type not in SUPPORTED_CHART_TYPES:
        issues.append(f"unsupported_chart_type:{chart_type}")

    if chart_type in CHART_TYPES_REQUIRING_SERIES:
        labels = visualization.get("labels") or []
        values = visualization.get("values") or []
        if not labels:
            issues.append("missing_labels")
        if not values:
            issues.append("missing_values")
        if len(labels) != len(values):
            issues.append("label_value_length_mismatch")
        if any(not isinstance(value, (int, float)) or isinstance(value, bool) for value in values):
            issues.append("non_numeric_values")
        if chart_type == "pie" and any(value < 0 for value in values if isinstance(value, (int, float))):
            issues.append("negative_pie_values")
        if chart_type == "line" and len(values) < 2:
            issues.append("line_needs_at_least_two_points")

    if chart_type == "number" and visualization.get("number_value") is None:
        issues.append("missing_number_value")

    score = max(0.0, 1.0 - 0.2 * len(issues))
    return score, {"issues": issues or ["none"], "chart_type": chart_type}


def chart_data_alignment(visualization: dict | None, result_data: dict | None) -> tuple[float, dict]:
    """Check whether chart labels/value keys align with returned SQL columns."""
    if not visualization or not result_data:
        return 0.0, {"issues": ["missing_visualization_or_result_data"]}

    columns = {str(col).lower() for col in result_data.get("columns", [])}
    issues = []

    for key_name in ("label_key", "value_key", "x_key", "y_key"):
        key = visualization.get(key_name)
        if key and not _is_aligned_key(str(key), columns):
            issues.append(f"{key_name}_not_in_result:{key}")

    records = visualization.get("records") or []
    if records:
        record_keys = {str(key).lower() for key in records[0].keys()}
        if not record_keys.intersection(columns):
            issues.append("records_do_not_overlap_result_columns")

    labels = visualization.get("labels") or []
    values = visualization.get("values") or []
    if visualization.get("type") in CHART_TYPES_REQUIRING_SERIES and labels and values:
        result_records = _normalize_result_records(result_data)
        if len(labels) > max(len(result_records), 1) and "duplicate_labels_aggregated" not in (visualization.get("warnings") or []):
            issues.append("chart_has_more_points_than_result")

    score = max(0.0, 1.0 - 0.25 * len(issues))
    return score, {"issues": issues or ["none"]}


def chart_llm_validation(
    question: str,
    response: str,
    visualization: dict | None,
    result_data: dict | None,
    llm_judge: LLMJudge,
) -> tuple[float, dict]:
    """LLM judge validates whether the chart choice and UI commentary fit the SQL result."""
    if not visualization:
        return 0.0, {"error": "missing_visualization"}

    result_preview = _build_result_preview(result_data)
    prompt = f"""You are evaluating a SQL agent UI visualization.

Question: {question}
Response/commentary: {response}
SQL result preview:
{result_preview}

Visualization payload:
{visualization}

Rate 0.0-1.0:
- Chart_Relevance: Does the selected chart type fit the user's question and result shape?
- Data_Alignment: Do labels/values match the SQL result?
- Commentary_Fit: Is the commentary brief, factual, and not duplicating the table?
- Overall_Visualization: Average quality.

Respond EXACTLY:
Chart_Relevance: [score]
Data_Alignment: [score]
Commentary_Fit: [score]
Overall_Visualization: [score]
Reasoning: [one sentence]"""

    try:
        result = llm_judge(prompt)
    except Exception as e:
        logger.warning("LLM judge failed in chart_llm_validation: %s", e)
        return 0.0, {"error": str(e)}

    chart_relevance, _ = _parse_score(result, "Chart_Relevance")
    data_alignment, _ = _parse_score(result, "Data_Alignment")
    commentary_fit, _ = _parse_score(result, "Commentary_Fit")
    overall, reasoning = _parse_score(result, "Overall_Visualization")

    return overall, {
        "chart_relevance": chart_relevance,
        "data_alignment": data_alignment,
        "commentary_fit": commentary_fit,
        "reasoning": reasoning,
    }


def visualization_score(
    question: str,
    response: str,
    visualization: dict | None,
    result_data: dict | None,
    llm_judge: LLMJudge | None = None,
) -> tuple[float, dict]:
    """Composite visualization score with deterministic checks plus optional LLM validation."""
    spec_score, spec_details = chart_spec_validity(visualization, result_data)
    alignment_score, alignment_details = chart_data_alignment(visualization, result_data)

    llm_score = None
    llm_details = {"skipped": True}
    if llm_judge and visualization:
        llm_score, llm_details = chart_llm_validation(question, response, visualization, result_data, llm_judge)

    if llm_score is None:
        score = round(spec_score * 0.55 + alignment_score * 0.45, 4)
    else:
        score = round(spec_score * 0.35 + alignment_score * 0.25 + llm_score * 0.40, 4)

    return score, {
        "chart_spec_validity": spec_score,
        "chart_data_alignment": alignment_score,
        "chart_llm_validation": llm_score,
        "spec_details": spec_details,
        "alignment_details": alignment_details,
        "llm_details": llm_details,
    }


def _build_result_preview(result_data: dict | None) -> str:
    if not result_data:
        return "No result data provided."

    preview = f"Columns: {result_data.get('columns', [])}\n"
    for row in result_data.get("rows", [])[:10]:
        preview += f"{row}\n"
    row_count = result_data.get("row_count")
    if row_count is not None:
        preview += f"Row count: {row_count}\n"
    return preview


def _is_aligned_key(key: str, columns: set[str]) -> bool:
    lowered = key.lower()
    if lowered in columns:
        return True
    for suffix in ("_count", "_sum", "_avg", "_min", "_max"):
        if lowered.endswith(suffix) and lowered.removesuffix(suffix) in columns:
            return True
    return False


def _normalize_result_records(result_data: dict | None) -> list[dict]:
    if not result_data:
        return []
    columns = result_data.get("columns") or []
    records = []
    for row in result_data.get("rows") or []:
        if isinstance(row, dict):
            records.append(row)
        elif isinstance(row, (list, tuple)):
            records.append({
                column: row[index] if index < len(row) else None
                for index, column in enumerate(columns)
            })
    return records

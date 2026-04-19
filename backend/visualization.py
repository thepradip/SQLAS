"""
Visualization inference for SQL query results.
Builds a compact chart spec for the frontend UI.
"""

from __future__ import annotations

from datetime import date, datetime
from numbers import Number


def build_visualization(user_query: str, query_result: dict) -> dict | None:
    columns = query_result.get("columns", [])
    rows = query_result.get("rows", [])

    if not columns or not rows:
        return None

    records = [dict(zip(columns, row)) for row in rows]
    numeric_columns = [col for col in columns if _is_numeric_column(records, col)]
    non_numeric_columns = [col for col in columns if col not in numeric_columns]

    if len(records) == 1:
        number_viz = _build_number_visualization(records[0], columns, numeric_columns, non_numeric_columns)
        if number_viz:
            return number_viz

    chart_viz = _build_chart_visualization(user_query, records, columns, numeric_columns, non_numeric_columns)
    if chart_viz:
        return chart_viz

    return {
        "type": "table",
        "title": "Result table",
        "description": "The result is best viewed as tabular data.",
        "records": records,
    }


def _build_number_visualization(record: dict, columns: list[str], numeric_columns: list[str], non_numeric_columns: list[str]) -> dict | None:
    if len(columns) == 1:
        value = record[columns[0]]
        return {
            "type": "number",
            "title": "Query result",
            "number_value": _coerce_number_value(value),
            "number_label": _pretty_label(columns[0]),
        }

    if len(numeric_columns) == 1:
        label = non_numeric_columns[0] if non_numeric_columns else numeric_columns[0]
        context = ", ".join(
            f"{_pretty_label(col)}: {record[col]}"
            for col in non_numeric_columns[1:]
            if record.get(col) is not None
        ) or None
        return {
            "type": "number",
            "title": "Key metric",
            "number_value": _coerce_number_value(record[numeric_columns[0]]),
            "number_label": _pretty_label(label),
            "number_context": context,
        }

    return None


def _build_chart_visualization(
    user_query: str,
    records: list[dict],
    columns: list[str],
    numeric_columns: list[str],
    non_numeric_columns: list[str],
) -> dict | None:
    if not numeric_columns:
        return None

    value_key = numeric_columns[0]
    label_key = non_numeric_columns[0] if non_numeric_columns else None

    if label_key is None and len(columns) >= 2:
        label_key = columns[0]

    if label_key is None:
        return None

    records, label_key, value_key, labels, values, description = _prepare_series(records, label_key, value_key)

    if any(value is None for value in values):
        return None

    chart_type = _choose_chart_type(user_query, labels, label_key, len(records))
    title = f"{_pretty_label(value_key)} by {_pretty_label(label_key)}"

    payload = {
        "type": chart_type,
        "title": title,
        "description": description or f"Auto-selected {chart_type} visualization for the SQL result.",
        "label_key": label_key,
        "value_key": value_key,
        "x_key": label_key,
        "y_key": value_key,
        "labels": labels,
        "values": values,
        "records": records,
    }

    if chart_type == "pie" and len(records) > 8:
        payload["type"] = "bar"
        payload["description"] = "Switched to bar chart because the result has too many slices for a pie chart."

    return payload


def _prepare_series(records: list[dict], label_key: str, value_key: str):
    labels = [_stringify_label(record.get(label_key), idx) for idx, record in enumerate(records, start=1)]
    values = [_coerce_float(record.get(value_key)) for record in records]
    description = None

    if any(value is None for value in values):
        return records, label_key, value_key, labels, values, description

    unique_labels = len(set(labels))
    if unique_labels < len(labels):
        aggregated = _aggregate_records(records, label_key, value_key)
        records = aggregated["records"]
        labels = aggregated["labels"]
        values = aggregated["values"]
        value_key = aggregated["value_key"]
        description = aggregated["description"]

    return records, label_key, value_key, labels, values, description


def _aggregate_records(records: list[dict], label_key: str, value_key: str) -> dict:
    grouped: dict[str, list[float]] = {}
    label_lookup: dict[str, str] = {}

    for idx, record in enumerate(records, start=1):
        label = _stringify_label(record.get(label_key), idx)
        value = _coerce_float(record.get(value_key))
        if value is None:
            continue
        label_lookup[label] = label
        grouped.setdefault(label, []).append(value)

    aggregate_mode = "count" if _looks_like_identifier(value_key) else "sum"
    aggregated_labels = list(grouped.keys())

    if aggregate_mode == "count":
        aggregated_values = [float(len(grouped[label])) for label in aggregated_labels]
        aggregated_key = f"{label_key}_count"
        description = "Aggregated duplicate labels into counts for a clearer chart."
    else:
        aggregated_values = [float(sum(grouped[label])) for label in aggregated_labels]
        aggregated_key = f"{value_key}_sum"
        description = "Aggregated duplicate labels into totals for a clearer chart."

    aggregated_records = [
        {label_key: label, aggregated_key: value}
        for label, value in zip(aggregated_labels, aggregated_values)
    ]

    return {
        "records": aggregated_records,
        "labels": aggregated_labels,
        "values": aggregated_values,
        "value_key": aggregated_key,
        "description": description,
    }


def _choose_chart_type(user_query: str, labels: list[str], label_key: str, row_count: int) -> str:
    lowered = user_query.lower()
    temporal_label = _looks_temporal_key(label_key) or sum(_looks_temporal_value(label) for label in labels[:5]) >= 2

    if temporal_label:
        return "line"

    pie_hints = ("share", "distribution", "breakdown", "percentage", "percent", "composition", "ratio", "pie")
    if row_count <= 6 and any(hint in lowered for hint in pie_hints):
        return "pie"

    if row_count <= 5 and "top" not in lowered and "trend" not in lowered:
        return "pie"

    return "bar"


def _is_numeric_column(records: list[dict], column: str) -> bool:
    values = [row.get(column) for row in records if row.get(column) is not None]
    if not values:
        return False
    return all(_is_numeric_value(value) for value in values)


def _is_numeric_value(value) -> bool:
    if isinstance(value, bool):
        return False
    return isinstance(value, Number)


def _looks_temporal_key(value: str) -> bool:
    lowered = value.lower()
    return any(token in lowered for token in ("date", "time", "month", "year", "day", "week"))


def _looks_like_identifier(value: str) -> bool:
    lowered = value.lower()
    return any(token in lowered for token in ("_id", " id", "number", "patient", "code", "key"))


def _looks_temporal_value(value: str) -> bool:
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        return True
    except Exception:
        pass

    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m", "%b %Y", "%B %Y", "%Y"):
        try:
            datetime.strptime(value, fmt)
            return True
        except Exception:
            continue
    return False


def _stringify_label(value, index: int) -> str:
    if value is None:
        return f"Item {index}"
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return str(value)


def _coerce_float(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, Number):
        return float(value)
    try:
        return float(value)
    except Exception:
        return None


def _coerce_number_value(value):
    if isinstance(value, float):
        return round(value, 2)
    return value


def _pretty_label(value: str) -> str:
    return value.replace("_", " ").strip().title()

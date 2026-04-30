"""
Cache performance metrics for SQL AI agents.

These metrics track the ROI of the semantic caching layer:
  cache_hit_score    — was this query served from cache?
  tokens_saved_score — normalized token savings
  few_shot_score     — were relevant verified examples injected?

All three are informational — they don't affect SQL correctness scoring
but provide cost and latency context in evaluation reports.
"""


# Approximate tokens for a full SQL generation pipeline call.
# Adjust for your model and schema size.
_FULL_PIPELINE_TOKENS = 9_500
_SQL_GEN_TOKENS = 8_600


def cache_hit_score(agent_result: dict) -> tuple[float, dict]:
    """
    Score 1.0 if this query was served from cache, 0.0 if it was a cache miss.

    Args:
        agent_result: The dict returned by the agent's run_query / run_react_query.

    Returns:
        (1.0 | 0.0, details dict)
    """
    metrics = agent_result.get("metrics") or {}
    hit = bool(metrics.get("cache_hit", False))
    cache_type = metrics.get("cache_type", "")
    tokens_saved = int(metrics.get("tokens_saved", 0))

    return (1.0 if hit else 0.0), {
        "cache_hit": hit,
        "cache_type": cache_type or "none",
        "tokens_saved": tokens_saved,
    }


def tokens_saved_score(agent_result: dict) -> tuple[float, dict]:
    """
    Normalised token savings score.

    1.0 = saved all tokens (exact cache hit).
    0.0 = no tokens saved (full pipeline run).

    Args:
        agent_result: The dict returned by the agent.

    Returns:
        (score 0.0–1.0, details dict)
    """
    metrics = agent_result.get("metrics") or {}
    saved = int(metrics.get("tokens_saved", 0))
    score = min(1.0, saved / _FULL_PIPELINE_TOKENS) if saved > 0 else 0.0

    cost_saved = round((saved / 1000) * 0.005, 6)   # GPT-4o ~$0.005/1K tokens

    return round(score, 4), {
        "tokens_saved": saved,
        "cost_saved_usd": cost_saved,
        "full_pipeline_tokens": _FULL_PIPELINE_TOKENS,
    }


def few_shot_score(agent_result: dict) -> tuple[float, dict]:
    """
    Score based on whether verified few-shot examples were injected.

    1.0 = verified examples used (learning loop active).
    0.5 = unverified examples used (implicit from hit count).
    0.0 = no examples available yet (cold start).

    Args:
        agent_result: The dict returned by the agent.

    Returns:
        (score 0.0–1.0, details dict)
    """
    metrics = agent_result.get("metrics") or {}
    count = int(metrics.get("few_shot_count", 0))
    verified = int(metrics.get("verified_few_shot_count", 0))

    if count == 0:
        score = 0.0
    elif verified > 0:
        score = 1.0
    else:
        score = 0.5

    return score, {"few_shot_count": count, "verified_count": verified}

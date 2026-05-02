"""
Advanced SQL AI Agent — fully dynamic, schema-aware, strictly read-only.
Instrumented with MLflow tracing for full observability.

For small databases (<= large_schema_threshold tables): injects full schema into every prompt.
For large databases (100s of tables): uses SchemaIndex to retrieve the ~8-12 most
relevant tables per query, keeping prompts under ~20K tokens regardless of DB size.

Author: Pradip Tivhale
"""

import re
from openai import AzureOpenAI

from config import get_settings
from database import build_full_context, get_full_schema, get_all_col_stats_cached, execute_readonly_query
from schema_index import SchemaIndex
from query_cache import QueryCache
from llm_providers import LLMProvider, get_provider
from tracing import traced_run_query
from visualization import build_visualization

settings = get_settings()

# Azure OpenAI client kept for embeddings only (schema index + semantic cache).
# All chat completions go through _provider so any LLM can be swapped in.
client = AzureOpenAI(
    azure_endpoint=settings.azure_openai_endpoint,
    api_key=settings.azure_openai_api_key,
    api_version=settings.azure_openai_api_version,
)

# ── Agent state (populated once on startup) ────────────────────────────────────

_schema_index: SchemaIndex | None = None
_db_context: str = ""
_query_cache: QueryCache | None = None
_provider: LLMProvider | None = None  # swappable chat completion backend


_schema_overview: str = ""   # short all-tables summary for the ReAct system prompt


async def init_agent(provider: LLMProvider | None = None):
    """
    Called once on app startup.
    Pass a provider to override the default Azure OpenAI backend —
    useful for eval runs that compare multiple LLMs.
    """
    global _schema_index, _db_context, _query_cache, _provider, _schema_overview
    _provider = provider or get_provider("azure", settings)

    if settings.cache_enabled:
        _query_cache = QueryCache(
            similarity_threshold=settings.semantic_cache_threshold,
            result_ttl=settings.result_cache_ttl,
        )
        print(f"  Query cache ready: {_query_cache.size()} cached queries loaded.")

    schema = await get_full_schema()
    table_count = len(schema)

    if table_count > settings.large_schema_threshold:
        print(f"  Large schema detected ({table_count} tables) — building semantic index...")
        col_stats = await get_all_col_stats_cached(schema)
        _schema_index = SchemaIndex(client, settings)
        await _schema_index.build(schema, col_stats)
        _schema_overview = _schema_index.all_tables_overview()
        retrieval_mode = "BM25+embeddings+RRF" if _schema_index._embeddings else "BM25"
        print(f"  Schema index ready ({table_count} tables, {retrieval_mode}).")
    else:
        print(f"  Small schema ({table_count} tables) — building full context...")
        _db_context = await build_full_context()
        _schema_overview = "\n".join(f"- `{t}`" for t in schema)
        print("  Full context ready.")


# ── System prompt ─────────────────────────────────────────────────────────────

def _build_few_shot_section(examples: list[dict]) -> str:
    """Format verified past queries as few-shot examples for the system prompt."""
    if not examples:
        return ""
    verified_count = sum(1 for ex in examples if ex.get("verified"))
    tag = f"{verified_count} user-verified" if verified_count else "from query history"
    lines = [
        f"## SQL Examples ({tag} — use as patterns, adapt to the current question)\n",
    ]
    for i, ex in enumerate(examples, 1):
        lines.append(f"**Example {i}:** {ex['question'].capitalize()}")
        lines.append(f"```sql\n{ex['sql']}\n```")
    return "\n".join(lines)


def _build_system_prompt(db_context: str, few_shot_section: str = "") -> str:
    domain = f"\n## Domain Context\n{settings.domain_hint}\n" if settings.domain_hint else ""
    examples_block = f"\n{few_shot_section}\n" if few_shot_section else ""

    return f"""You are an expert SQL analyst. You write efficient, read-only SQL queries against the database schema provided below.

## STRICT READ-ONLY POLICY
- You MUST only generate SELECT statements or WITH (CTE) + SELECT.
- NEVER generate INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, TRUNCATE, GRANT, REVOKE, or any DDL/DML.
- If the user asks to modify, insert, or delete data, refuse politely.

{db_context}
{domain}{examples_block}
## SQL Generation Rules
1. Write standard SQL compatible with the connected database.
2. Only generate SELECT or WITH...SELECT queries.
3. Use the column statistics provided above to write accurate filters and understand value ranges.
4. Use CASE WHEN to map coded values to readable labels when the column stats show a small number of distinct integer values.
5. Use meaningful column aliases (AS keyword).
6. For multi-table queries involving 1:N relationships, aggregate the N-side first, then JOIN to avoid row explosion.
7. Use CTEs (WITH clauses) for complex multi-step logic.
8. Default LIMIT 100 for detail queries. Aggregated summaries need no limit.
9. Handle NULLs appropriately (COALESCE, IS NOT NULL, etc.) — check the null counts in the stats.
10. For percentages: CAST to REAL to avoid integer division.
11. ROUND numeric outputs to 2 decimal places.
12. Always ORDER BY for deterministic results.
13. Use indexes — prefer filtering on indexed columns for large tables.

## Output Format
Return ONLY the SQL inside a ```sql``` block. No explanations outside the block.
"""


async def _get_context_for_query(user_query: str) -> str:
    """Return the schema context to inject for this specific query."""
    if _schema_index is not None:
        relevant_tables = await _schema_index.retrieve_async(user_query)
        overview = _schema_index.all_tables_overview()
        focused = _schema_index.focused_context(relevant_tables)
        return (
            f"{overview}\n\n"
            f"## Schema for Tables Relevant to Your Query\n"
            f"*(retrieved {len(relevant_tables)} of {len(_schema_index._schema)} tables)*\n\n"
            f"{focused}"
        )
    return _db_context


# ── SQL parsing helpers ────────────────────────────────────────────────────────

def _extract_sql(content: str) -> str:
    match = re.search(r"```sql\s*(.*?)```", content, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"```\s*((?:SELECT|WITH).*?)```", content, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return content.strip()


def _strip_markdown_tables(content: str) -> str:
    lines = content.splitlines()
    cleaned: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        next_line = lines[i + 1] if i + 1 < len(lines) else ""
        is_table_header = "|" in line and line.strip().startswith("|")
        is_separator = bool(re.match(r"^\s*\|?[\s:-]+(\|[\s:-]+)+\|?\s*$", next_line))
        if is_table_header and is_separator:
            i += 2
            while i < len(lines) and "|" in lines[i]:
                i += 1
            continue
        cleaned.append(line)
        i += 1
    return re.sub(r"\n{3,}", "\n\n", "\n".join(cleaned)).strip()


# ── Core pipeline functions (called inside traced spans) ───────────────────────

async def _generate_sql(user_query: str, conversation_history: list[dict]) -> str:
    db_context = await _get_context_for_query(user_query)

    # Inject few-shot examples from cache (verified entries prioritised)
    few_shot_section = ""
    if _query_cache is not None:
        embedding = await _get_query_embedding(user_query)
        examples = _query_cache.get_few_shot_examples(user_query, embedding, top_k=3)
        few_shot_section = _build_few_shot_section(examples)

    messages = [{"role": "system", "content": _build_system_prompt(db_context, few_shot_section)}]
    for msg in conversation_history[-6:]:
        messages.append(msg)
    messages.append({
        "role": "user",
        "content": f'Write an efficient SQL query to answer: "{user_query}"\n\nReturn ONLY the SQL in a ```sql``` block.',
    })
    return _extract_sql(_provider.complete(messages, max_tokens=2000))


async def _narrate_result(user_query: str, sql: str, query_result: dict) -> str:
    columns = query_result["columns"]
    rows = query_result["rows"][:50]
    row_count = query_result["row_count"]
    exec_time = query_result["execution_time_ms"]

    data_preview = f"Columns: {columns}\n"
    for row in rows[:25]:
        data_preview += f"{row}\n"
    if row_count > 25:
        data_preview += f"... ({row_count} total rows)\n"

    messages = [
        {
            "role": "system",
            "content": (
                "You are a data analyst. Answer the user's question based on the SQL results.\n\n"
                "RULES:\n"
                "- Give the DIRECT answer first. If it's a single number, just state it plainly.\n"
                "- The UI already renders structured result tables separately. Never output markdown tables.\n"
                "- Do NOT add disclaimers, caveats, or health advice. Just answer the question.\n"
                "- ONLY state facts from the SQL result. No invented numbers.\n"
                "- Keep it short — 1-3 sentences for simple queries, up to 4 sentences for comparisons.\n"
                "- Summarize patterns or notable comparisons in prose instead of listing every row.\n"
                "- No emojis, no headings, no bullet points for simple answers."
            ),
        },
        {
            "role": "user",
            "content": (
                f"**Question:** {user_query}\n\n"
                f"**SQL:**\n```sql\n{sql}\n```\n\n"
                f"**Results** ({row_count} rows, {exec_time}ms):\n```\n{data_preview}```\n\n"
                f"Provide a clear answer."
            ),
        },
    ]
    return _strip_markdown_tables(_provider.complete(messages, max_tokens=2000))


async def _retry_generate_sql(
    user_query: str, failed_sql: str, error: str, conversation_history: list[dict]
) -> str:
    db_context = await _get_context_for_query(user_query)
    few_shot_section = ""
    if _query_cache is not None:
        embedding = await _get_query_embedding(user_query)
        examples = _query_cache.get_few_shot_examples(user_query, embedding, top_k=2)
        few_shot_section = _build_few_shot_section(examples)

    retry_msg = (
        f'SQL for "{user_query}" failed:\nError: {error}\n\n'
        f"Failed SQL:\n```sql\n{failed_sql}\n```\n\n"
        f"Fix the SQL. Return only corrected query in a ```sql``` block."
    )
    messages = (
        [{"role": "system", "content": _build_system_prompt(db_context, few_shot_section)}]
        + conversation_history[-4:]
        + [{"role": "user", "content": retry_msg}]
    )
    return _extract_sql(_provider.complete(messages, max_tokens=2000))


# ── Cache helpers ──────────────────────────────────────────────────────────────

async def _get_query_embedding(query: str) -> list[float] | None:
    """Embed a query using the configured Azure OpenAI embedding deployment."""
    model = settings.azure_openai_embedding_deployment
    if not model:
        return None
    try:
        resp = client.embeddings.create(model=model, input=query[:2000])
        return resp.data[0].embedding
    except Exception:
        return None


async def _execute_with_result_cache(sql: str) -> dict:
    """Execute SQL, serving from result cache if available and within TTL."""
    if _query_cache is not None:
        cached = _query_cache.lookup_result(sql)
        if cached:
            return cached
    result = await execute_readonly_query(sql)
    if _query_cache is not None and settings.result_cache_ttl > 0:
        _query_cache.store_result(sql, result)
    return result


def get_cache() -> QueryCache | None:
    """Expose cache instance for API endpoints."""
    return _query_cache


def get_provider_name() -> str:
    """Current provider name — used in eval reports and MLflow tags."""
    return _provider.name if _provider else "unknown"


# ── Public API ─────────────────────────────────────────────────────────────────

def _is_complex_query(query: str) -> bool:
    """
    Heuristic: route to ReAct agent for queries that benefit from multi-step reasoning.
    Simple COUNT/AVG/filter queries go to the fast pipeline.
    """
    lowered = query.lower()
    complex_signals = [
        "correlat", "why ", "explain", "compare", "analyze", "analys",
        "trend", "pattern", "insight", "relationship between",
        "vs ", " versus ", " and also ", "additionally", "furthermore",
        "how does", "what factors", "what causes",
    ]
    return any(s in lowered for s in complex_signals)


async def run_query(
    user_query: str,
    conversation_history: list[dict],
    force_agentic: bool = False,
) -> dict:
    """
    Routes query to the ReAct agent (multi-step tool use) or the fast pipeline.

    Routing logic:
    - force_agentic=True  → always ReAct
    - Complex query heuristic matches → ReAct (if provider supports tool calling)
    - Cache hit → pipeline (no LLM cost regardless)
    - Otherwise → fast pipeline

    Both paths return the same dict shape.
    """
    # ── Agentic (ReAct) path ──────────────────────────────────────────────────
    use_react = (
        (force_agentic or (settings.agentic_mode and _is_complex_query(user_query)))
        and _provider is not None
        and _provider.supports_tool_calling
    )
    if use_react:
        from react_agent import run_react_query
        return await run_react_query(
            user_query=user_query,
            provider=_provider,
            schema_overview=_schema_overview,
            conversation_history=conversation_history,
        )

    # ── Fast pipeline path ────────────────────────────────────────────────────
    # Only cache single-turn queries — conversations require full context
    use_cache = _query_cache is not None and not conversation_history

    if use_cache:
        embedding = await _get_query_embedding(user_query)
        cache_hit = _query_cache.lookup(user_query, embedding)

        if cache_hit:
            # Always re-execute SQL so results are fresh
            exec_fn = _execute_with_result_cache if settings.result_cache_ttl > 0 else execute_readonly_query
            try:
                result = await exec_fn(cache_hit.sql)
            except Exception:
                # SQL may have broken due to schema change — fall through to full pipeline
                _query_cache = None  # temporary disable to avoid repeated failures
                pass
            else:
                # Exact hits reuse the cached response; semantic hits re-narrate
                if cache_hit.cache_type == "exact" and cache_hit.response:
                    response = cache_hit.response
                else:
                    response = await _narrate_result(user_query, cache_hit.sql, result)

                viz = build_visualization(user_query, result)
                return {
                    "sql": cache_hit.sql,
                    "data": result,
                    "response": response,
                    "success": True,
                    "trace_id": None,
                    "metrics": {
                        "total_latency_ms": result.get("execution_time_ms", 0),
                        "generation_latency_ms": 0,
                        "sql_execution_ms": result.get("execution_time_ms", 0),
                        "narration_latency_ms": 0,
                        "retry_count": 0,
                        "result_rows": result.get("row_count"),
                        "result_columns": len(result.get("columns", [])),
                        "success": True,
                        "cache_hit": True,
                        "cache_type": cache_hit.cache_type,
                        "cache_similarity": cache_hit.similarity,
                        "tokens_saved": cache_hit.tokens_saved,
                    },
                    "visualization": viz,
                }

    # Cache miss (or conversation context): full traced pipeline
    result = await traced_run_query(
        user_query=user_query,
        conversation_history=conversation_history,
        generate_sql_fn=_generate_sql,
        execute_sql_fn=_execute_with_result_cache,
        narrate_fn=_narrate_result,
        retry_generate_fn=_retry_generate_sql,
    )

    # Store successful result in cache (single-turn only)
    if use_cache and result.get("success") and _query_cache is not None:
        embedding = embedding if "embedding" in locals() else await _get_query_embedding(user_query)
        _query_cache.store(
            nl_query=user_query,
            sql=result["sql"],
            response=result["response"],
            embedding=embedding,
            trace_id=result.get("trace_id"),   # links this entry to MLflow feedback
        )

    return result

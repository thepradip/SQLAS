"""
AriaSQL Context Engine — unified schema retrieval with pre-generation validation.

Replaces the ad-hoc calls scattered across schema_index.py and agent.py with
a single, validated pipeline that guarantees the LLM receives the right schema
context before generating SQL.

Pipeline:
  1. QueryAnalyzer      — extract entities, intent, aggregation hints, join signals
  2. MultiStrategyRetriever — BM25 + entity matching + glossary expansion + embeddings
  3. FK Graph Expander  — guarantee JOIN-complete table selection
  4. ContextValidator   — check coverage, FK completeness, warn on low confidence
  5. ContextBuilder     — column filtering, compact format, token budget

Usage:
    engine = ContextEngine(schema_index, training_store=store, settings=settings)

    ctx = await engine.get_context("What is the average revenue by region this quarter?")

    print(ctx.tables)            # ["orders", "regions"]
    print(ctx.confidence)        # 0.87
    print(ctx.validation.is_valid) # True
    print(ctx.validation.warnings) # []
    print(ctx.context_str)       # formatted schema ready for LLM prompt
"""

import re
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from schema_index import SchemaIndex
    from training_store import TrainingStore
    from config import Settings


# ── Intent signals ────────────────────────────────────────────────────────────

_AGGREGATION_WORDS = {
    "total", "sum", "count", "average", "avg", "mean", "max", "maximum",
    "min", "minimum", "median", "percent", "percentage", "rate", "ratio",
    "distribution", "breakdown",
}

_JOIN_SIGNALS = {
    "by", "per", "for each", "across", "between", "join", "with",
    "related", "associated", "linked",
}

_TEMPORAL_SIGNALS = {
    "today", "yesterday", "week", "month", "quarter", "year", "annual",
    "daily", "weekly", "monthly", "quarterly", "ytd", "mtd", "last",
    "this", "current", "recent", "latest", "since", "between",
}

_COMPARISON_SIGNALS = {
    "more than", "less than", "greater", "higher", "lower", "above",
    "below", "equal", "between", "top", "bottom", "best", "worst",
    "compare", "versus", "vs",
}


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class QueryIntent:
    """Structured analysis of a natural language query."""
    raw_query: str
    tokens: list[str]                    # lowercased, cleaned tokens
    entity_hints: list[str]              # words that might be table/column names
    aggregations: list[str]              # detected aggregation keywords
    temporal_refs: list[str]             # time-related keywords
    join_signals: list[str]              # words suggesting multi-table query
    comparison_signals: list[str]        # comparative language
    is_complex: bool                     # likely multi-table / multi-step?
    is_aggregation: bool                 # likely GROUP BY needed?


@dataclass
class ContextValidation:
    """
    Pre-generation validation of the retrieved schema context.
    Runs before the context is sent to the LLM — catches coverage gaps early.
    """
    is_valid: bool                        # True if context is good enough
    confidence: float                     # 0.0–1.0 overall confidence
    coverage_score: float                 # entity coverage: query terms found in schema
    fk_complete: bool                     # all FK partners included for JOINs
    missing_entities: list[str]           # query entities not found in context
    missing_fk_tables: list[str]          # FK tables that should be included
    warnings: list[str]                   # non-fatal issues (low confidence, etc.)
    suggestions: list[str]                # "Add table X for JOIN completeness"
    strategy_used: str                    # which retrieval strategy found the tables


@dataclass
class RetrievedContext:
    """Output of the Context Engine — ready to inject into SQL generation prompt."""
    tables: list[str]                     # selected table names
    context_str: str                      # formatted schema context for LLM
    validation: ContextValidation
    confidence: float                     # 0.0–1.0
    token_estimate: int                   # estimated tokens in context_str
    intent: QueryIntent                   # parsed query intent


# ── Context Engine ────────────────────────────────────────────────────────────

class ContextEngine:
    """
    Unified context retrieval and validation pipeline.

    Wraps SchemaIndex with additional strategies and a pre-generation validator
    that catches coverage issues before the LLM is called.
    """

    def __init__(
        self,
        schema_index: "SchemaIndex",
        training_store=None,
        settings=None,
        schema_graph=None,
        vector_store=None,
    ):
        self._index        = schema_index
        self._training     = training_store
        self._settings     = settings
        self._graph        = schema_graph   # SchemaGraph (rich semantic graph)
        self._vectors      = vector_store   # VectorStore (ChromaDB)
        self._max_cols     = getattr(settings, "max_columns_per_table", 30)
        self._token_budget = getattr(settings, "schema_token_budget", 12_000)
        self._top_k        = getattr(settings, "max_context_tables", 8)

    # ── Public API ─────────────────────────────────────────────────────────────

    async def get_context(self, query: str) -> RetrievedContext:
        """
        Full pipeline: analyze → dynamic top-k → retrieve → expand → validate → build.

        The number of tables retrieved is not fixed — it adapts to query complexity.
          Simple query (count, single table) → 1-2 tables
          Medium query (compare, group by)   → 3-4 tables
          Complex query (multi-join, analyse) → 5-10 tables

        Returns a RetrievedContext with quality scores and validation results.
        If confidence is low, adds warnings but still returns best-effort context.
        """
        intent   = self._analyze(query)
        top_k    = self._dynamic_top_k(intent)
        tables, strategy = await self._retrieve(query, intent, top_k=top_k)
        tables   = self._fk_expand(tables, intent)
        validation = self._validate(tables, intent, strategy)
        context_str = self._build_context(tables, query)
        confidence  = self._compute_confidence(validation)

        return RetrievedContext(
            tables=tables,
            context_str=context_str,
            validation=validation,
            confidence=confidence,
            token_estimate=len(context_str) // 4,
            intent=intent,
        )

    def _dynamic_top_k(self, intent: QueryIntent) -> int:
        """
        How many CANDIDATES to retrieve from each source.

        This is NOT the final table count — it's the pool size fed to the
        reranker. The reranker then applies a score threshold to decide the
        final set (see _rerank_with_threshold).

        We always retrieve a generous pool (min 15) so the reranker has
        enough signal to work with. A small pool would blind the reranker —
        if the 3rd-most-relevant table scores below BM25 rank 2, it never
        gets a chance.

        The only thing we adjust is the UPPER CAP of the pool based on query
        complexity, to keep latency reasonable for simple queries.
        """
        # Minimum pool: always retrieve at least 15 candidates
        # so the reranker sees enough signal even if BM25 ordering is imperfect
        pool = 15

        # For very complex queries, expand pool to catch more candidates
        if intent.is_complex and len(intent.join_signals) > 2:
            pool = 20
        elif intent.is_complex:
            pool = 18

        return pool

    # ── Step 1: Query Analysis ─────────────────────────────────────────────────

    def _analyze(self, query: str) -> QueryIntent:
        """
        Extract structured intent from a natural language query.
        Uses pattern matching — zero LLM calls, <1ms.
        """
        lower = query.lower()
        tokens = re.sub(r"[^a-z0-9\s]", " ", lower).split()
        token_set = set(tokens)

        aggregations     = sorted(token_set & _AGGREGATION_WORDS)
        temporal_refs    = sorted(token_set & _TEMPORAL_SIGNALS)
        join_sigs        = sorted(token_set & _JOIN_SIGNALS)
        comparison_sigs  = []
        for sig in _COMPARISON_SIGNALS:
            if sig in lower:
                comparison_sigs.append(sig)

        # Entity hints: words not in stopwords that might be table/column names
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "of", "for",
                     "in", "on", "at", "to", "by", "as", "it", "its", "and",
                     "or", "but", "not", "all", "each", "any", "how", "what",
                     "which", "who", "when", "where", "show", "give", "list",
                     "find", "get", "many", "much", "do", "does", "have", "has"}
        entity_hints = [t for t in tokens
                        if t not in stopwords
                        and t not in _AGGREGATION_WORDS
                        and t not in _TEMPORAL_SIGNALS
                        and len(t) > 2]

        is_aggregation = bool(aggregations) or any(
            w in lower for w in ("group by", "group", "breakdown", "distribution")
        )
        is_complex = (
            len(join_sigs) > 0
            or bool(temporal_refs)
            or any(w in lower for w in ("join", "relate", "connect", "across", "compare"))
        )

        return QueryIntent(
            raw_query=query,
            tokens=tokens,
            entity_hints=entity_hints,
            aggregations=aggregations,
            temporal_refs=temporal_refs,
            join_signals=join_sigs,
            comparison_signals=comparison_sigs,
            is_complex=is_complex,
            is_aggregation=is_aggregation,
        )

    # ── Step 2: Multi-Strategy Retrieval with Reranker ────────────────────────

    async def _retrieve(
        self,
        query: str,
        intent: QueryIntent,
        top_k: int | None = None,
    ) -> tuple[list[str], str]:
        """
        Multi-strategy retrieval with automatic fallback.

        Priority:
          1. Hybrid BM25+embedding (most accurate when both available)
          2. Entity matching against schema (catches "customers" from "customer")
          3. Glossary expansion from TrainingStore (domain-specific terms)
          4. Pure BM25 fallback
        """
        schema = self._index._schema
        if not schema:
            return [], "empty_schema"

        k = top_k or self._top_k

        # Strategy 1: Standard BM25+RRF (existing index)
        try:
            tables_bm25 = await self._index.retrieve_async(query, top_k=k)
        except Exception:
            tables_bm25 = self._index.retrieve_relevant_tables(query, top_k=k)

        # Strategy 2: Entity matching — detect table/column names in query tokens
        entity_tables = self._entity_match(intent.entity_hints, schema)

        # Strategy 3: Business glossary via TrainingStore
        glossary_tables: list[str] = []
        if self._training:
            items = self._training.get_context(query, top_k=3)
            for item in items:
                if item.type == "ddl":
                    # Extract table name from DDL
                    m = re.search(
                        r"CREATE\s+(?:TABLE|VIEW)\s+(?:IF\s+NOT\s+EXISTS\s+)?[`\"']?(\w+)",
                        item.content, re.IGNORECASE,
                    )
                    if m:
                        tname = m.group(1).lower()
                        if tname in schema:
                            glossary_tables.append(tname)

        # Strategy 4: ChromaDB vector similarity (table + column level search)
        vector_tables: list[str] = []
        if self._vectors and self._vectors.ready:
            vector_tables = self._vectors.find_relevant_tables(
                query, top_k=k   # use same pool size as BM25, not fixed self._top_k
            )

        # ── Merge all candidates ───────────────────────────────────────────────
        all_candidates: dict[str, dict] = {}

        for rank, t in enumerate(tables_bm25):
            all_candidates.setdefault(t, {"bm25_rank": rank, "entity": False, "glossary": False, "graph": False, "vector": False, "vector_rank": 999})
            all_candidates[t]["bm25_rank"] = rank

        for t in entity_tables:
            all_candidates.setdefault(t, {"bm25_rank": 999, "entity": False, "glossary": False, "graph": False, "vector": False, "vector_rank": 999})
            all_candidates[t]["entity"] = True

        for t in glossary_tables:
            all_candidates.setdefault(t, {"bm25_rank": 999, "entity": False, "glossary": False, "graph": False, "vector": False, "vector_rank": 999})
            all_candidates[t]["glossary"] = True

        for rank, t in enumerate(vector_tables):
            all_candidates.setdefault(t, {"bm25_rank": 999, "entity": False, "glossary": False, "graph": False, "vector": False, "vector_rank": 999})
            all_candidates[t]["vector"] = True
            all_candidates[t]["vector_rank"] = rank

        # ── Graph RAG: add 1-hop FK neighbors as candidates ───────────────────
        graph = self._index._graph
        for t in list(all_candidates.keys()):
            try:
                for neighbor in graph.neighbors(t):
                    if neighbor not in all_candidates and neighbor in self._index._schema:
                        all_candidates[neighbor] = {
                            "bm25_rank": 999, "entity": False,
                            "glossary": False, "graph": True
                        }
            except Exception:
                pass

        # ── Reranker: score + threshold filter ────────────────────────────────
        scored   = self._rerank(query, intent, all_candidates)
        reranked = self._rerank_with_threshold(scored, intent)

        # ── Determine strategy label ──────────────────────────────────────────
        entity_added   = [t for t, m in all_candidates.items() if m["entity"]   and t not in tables_bm25]
        glossary_added = [t for t, m in all_candidates.items() if m["glossary"] and t not in tables_bm25]
        graph_added    = [t for t, m in all_candidates.items() if m.get("graph")]
        vector_added   = [t for t, m in all_candidates.items() if m.get("vector") and t not in tables_bm25]

        parts = ["hybrid_bm25"]
        if vector_added:   parts.append("chroma_vector")
        if entity_added:   parts.append("entity")
        if glossary_added: parts.append("glossary")
        if graph_added:    parts.append("graph_rag")

        return reranked[: self._top_k + 4], "+".join(parts)

    def _rerank(
        self,
        query: str,
        intent: QueryIntent,
        candidates: dict,
    ) -> list[str]:
        """
        Cross-encoder style reranker — scores each candidate table using
        multiple signals and returns tables sorted by relevance.

        Signals (weighted):
          - BM25 rank position    (lower = better)
          - Entity match bonus    (table found via entity extraction)
          - Glossary match bonus  (table found via business glossary)
          - Graph RAG penalty     (penalise pure FK-expansion candidates slightly)
          - Query-column overlap  (do columns match query tokens?)
          - Aggregation fitness   (has numeric columns for GROUP BY queries?)
          - Row count fitness     (prefer non-empty tables)

        This is a lightweight cross-encoder that runs in <1ms (no LLM needed).
        """
        schema   = self._index._schema
        q_tokens = set(intent.tokens)

        scored: list[tuple[float, str]] = []

        for table_name, meta in candidates.items():
            info = schema.get(table_name, {})
            cols = info.get("columns", [])
            score = 0.0

            tname_lower  = table_name.lower()
            tname_stem   = tname_lower.rstrip("s")
            hint_stems   = {h.lower().rstrip("s") for h in intent.entity_hints}
            hint_set     = {h.lower() for h in intent.entity_hints}
            tname_words  = set(tname_lower.split("_"))

            # ── Signal 1: Direct name match (STRONGEST) ───────────────────────
            # Table name literally appears as a word in the query.
            # e.g. "Count active USERS" → users table gets 2.0 bonus.
            # This is the key fix for precision: separates tables the user
            # explicitly mentioned from tables that happen to have similar columns.
            if tname_lower in hint_set or tname_stem in hint_stems:
                score += 2.0
            elif tname_words & hint_set or tname_words & hint_stems:
                score += 1.2   # partial word match (e.g. "order" in "order_items")

            # ── Signal 2: BM25 rank ───────────────────────────────────────────
            bm25_rank = meta.get("bm25_rank", 999)
            score += max(0.0, 0.8 - bm25_rank * 0.08)  # rank 0=0.8, rank 10=0.0

            # ── Signal 3: ChromaDB vector similarity ──────────────────────────
            vector_rank = meta.get("vector_rank", 999)
            if meta.get("vector"):
                score += max(0.0, 0.5 - vector_rank * 0.06)

            # ── Signal 4: Entity match (indirect) ────────────────────────────
            if meta.get("entity") and tname_lower not in hint_set:
                score += 0.3   # indirect — lower than direct name match

            # ── Signal 5: Business glossary ───────────────────────────────────
            if meta.get("glossary"):
                score += 0.4

            # ── Signal 6: Graph RAG neighbor ─────────────────────────────────
            if meta.get("graph"):
                score -= 0.15  # slightly penalise pure FK-expansion candidates

            # ── Signal 7: Column-query token overlap ──────────────────────────
            col_names = {c["name"].lower() for c in cols}
            col_tokens: set[str] = set()
            for cn in col_names:
                col_tokens.update(re.sub(r"[^a-z0-9]", " ", cn).split())
            overlap = len(q_tokens & col_tokens)
            score += min(overlap * 0.12, 0.4)  # capped lower than before

            # Aggregation fitness: bonus if table has numeric cols for GROUP BY
            if intent.is_aggregation:
                num_cols = sum(
                    1 for c in cols
                    if any(t in c["type"].upper() for t in ("INT","REAL","FLOAT","DECIMAL","NUMERIC"))
                )
                score += min(num_cols * 0.05, 0.2)

            # Row count fitness: skip empty tables
            row_count = info.get("row_count", 0)
            if isinstance(row_count, int) and row_count > 0:
                score += 0.1

            scored.append((score, table_name))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored   # returns [(score, table_name), ...]

    def _rerank_with_threshold(
        self,
        scored: list[tuple[float, str]],
        intent: QueryIntent,
    ) -> list[str]:
        """
        Apply adaptive score threshold to select the final table set.

        Instead of a fixed top-K, we keep tables whose reranker score
        exceeds a threshold — the count emerges from the data, not a number.

          Simple query  → 1-3 tables (only high-confidence kept)
          Complex query → 5-8 tables (many tables score above threshold)

        Adaptive threshold:
          top score > 1.5  →  0.50  (high confidence: be selective)
          top score > 0.8  →  0.40  (normal)
          top score ≤ 0.8  →  0.25  (sparse schema: be generous)

        Hard rules: always ≥ 1 table, never > max_context_tables.
        """
        if not scored:
            return []

        max_tables = getattr(self._settings, "max_context_tables", 8)
        top_score  = scored[0][0]

        if top_score > 2.0:
            # A direct name match occurred (reranker bonus = +2.0).
            # Relative threshold: keep tables that score ≥ 35% of the winner.
            # Separates explicitly-named tables from BM25/column-overlap noise.
            # "Count active USERS" → users=3.0, orders=0.8 → threshold=1.05 → only users kept.
            threshold = top_score * 0.35
        elif top_score > 0.8:
            threshold = 0.45   # normal multi-table query
        else:
            threshold = 0.25   # sparse schema — be generous

        # Adaptive max: simple queries need fewer tables than complex ones.
        # Cap at (number of entity hints + 1) for simple queries so we don't
        # flood the prompt with irrelevant tables.
        n_hints = len(intent.entity_hints)
        if not intent.is_complex and n_hints <= 2:
            adaptive_max = min(max_tables, max(2, n_hints + 1))
        else:
            adaptive_max = max_tables

        result: list[str] = []
        for score, table in scored:
            if score >= threshold or len(result) == 0:
                result.append(table)
            if len(result) >= adaptive_max:
                break
            # Stop when score drops to half the threshold and we have ≥ 2 tables
            if score < threshold * 0.5 and len(result) >= 2:
                break

        return result[:adaptive_max]

    def _entity_match(
        self,
        entity_hints: list[str],
        schema: dict,
    ) -> list[str]:
        """
        Match query entity words against table names and column names.

        Handles:
          - Exact match: "orders" → orders table
          - Singular/plural: "customer" → customers table
          - Substring: "product" → products, product_categories
          - Column-based: "blood_pressure" → patients table (has that column)
        """
        matched: list[str] = []
        table_names = list(schema.keys())

        for hint in entity_hints:
            hint_lower = hint.lower()
            for tname in table_names:
                tname_lower = tname.lower()
                # Split on underscore for word-boundary matching (avoids "per" matching "per_diem")
                tname_words = set(tname_lower.split("_"))
                hint_words  = set(hint_lower.split("_"))
                # Match: exact word, singular/plural stem, or direct name equality
                if (hint_lower == tname_lower
                        or hint_lower.rstrip("s") == tname_lower.rstrip("s")
                        or hint_words & tname_words  # shared word tokens
                        or hint_lower in tname_words
                        or tname_lower in hint_words):
                    if tname not in matched:
                        matched.append(tname)
                    continue

                # Column-name match: hint appears as a column in this table
                cols = [c["name"].lower() for c in schema[tname].get("columns", [])]
                for col in cols:
                    col_clean = col.replace("_", "")
                    hint_clean = hint_lower.replace("_", "")
                    if hint_clean in col_clean or col_clean in hint_clean:
                        if tname not in matched:
                            matched.append(tname)
                        break

        return matched

    # ── Step 3: FK Graph Expansion ────────────────────────────────────────────

    def _fk_expand(self, tables: list[str], intent: QueryIntent) -> list[str]:
        """
        Expand selected tables using the rich schema graph (preferred)
        or the basic FK graph as fallback.

        Schema graph adds not just FK but also naming+domain+hierarchy edges
        with weights — allows smarter expansion decisions.
        """
        # Use rich schema graph if available
        if self._graph and self._graph.G.number_of_nodes() > 0:
            expanded = list(tables)
            cap = self._top_k + 4
            for t in list(tables):
                neighbors = self._graph.get_neighbors(
                    t,
                    edge_types=["fk"] if not intent.is_complex else None,
                    min_weight=0.4,
                )
                for nbr, weight, etype in neighbors:
                    if nbr in expanded or len(expanded) >= cap:
                        continue
                    if intent.is_complex:
                        expanded.append(nbr)
                    elif any(tok in nbr.lower() for tok in intent.entity_hints):
                        expanded.append(nbr)
            return expanded

        graph = self._index._graph
        try:
            bool(graph)
            has_graph = True
        except Exception:
            has_graph = False

        if not has_graph or not tables:
            return tables

        expanded = list(tables)
        cap = self._top_k + 4

        for t in list(tables):
            try:
                neighbors = list(graph.neighbors(t))
            except Exception:
                continue
            for neighbor in neighbors:
                if neighbor in expanded:
                    continue
                if len(expanded) >= cap:
                    break
                # For simple queries: only add if neighbor is entity-hinted
                if intent.is_complex:
                    expanded.append(neighbor)
                else:
                    # Only if neighbor name overlaps with query
                    if any(
                        tok in neighbor.lower() or neighbor.lower() in tok
                        for tok in intent.entity_hints
                    ):
                        expanded.append(neighbor)

        return expanded

    # ── Step 4: Context Validation ────────────────────────────────────────────

    def _validate(
        self,
        tables: list[str],
        intent: QueryIntent,
        strategy: str,
    ) -> ContextValidation:
        """
        Pre-generation validation of the retrieved schema context.

        Checks:
          1. Entity coverage — what % of query entities appear in retrieved tables/columns
          2. FK completeness — are all FK partners present for JOIN queries
          3. Minimum table count — complex queries need >= 2 tables
          4. Aggregation readiness — GROUP BY candidates available
        """
        schema = self._index._schema
        warnings: list[str] = []
        suggestions: list[str] = []
        missing_entities: list[str] = []
        missing_fk_tables: list[str] = []

        # ── Coverage check ─────────────────────────────────────────────────────
        covered_entities = set()
        all_col_names: set[str] = set()
        for t in tables:
            all_col_names.update(
                c["name"].lower()
                for c in schema.get(t, {}).get("columns", [])
            )
            all_col_names.add(t.lower())

        for hint in intent.entity_hints:
            hint_clean = hint.lower().replace("_", "")
            for name in all_col_names:
                name_clean = name.replace("_", "")
                if hint_clean in name_clean or name_clean in hint_clean:
                    covered_entities.add(hint)
                    break
            else:
                missing_entities.append(hint)

        coverage_score = (
            len(covered_entities) / len(intent.entity_hints)
            if intent.entity_hints else 1.0
        )

        if coverage_score < 0.5:
            warnings.append(
                f"Low entity coverage ({coverage_score:.0%}) — "
                f"query mentions {missing_entities[:3]} but these weren't found in schema"
            )

        # ── FK completeness check ──────────────────────────────────────────────
        graph = self._index._graph
        fk_complete = True
        if intent.is_complex:
            tables_set = set(tables)
            for t in tables:
                for fk in schema.get(t, {}).get("foreign_keys", []):
                    ref = fk.get("referred_table", "")
                    if ref and ref not in tables_set and ref in schema:
                        missing_fk_tables.append(ref)
                        fk_complete = False

            if missing_fk_tables:
                unique_missing = list(dict.fromkeys(missing_fk_tables))[:3]
                warnings.append(
                    f"FK partners not in context: {unique_missing} "
                    f"— JOINs may fail or produce wrong results"
                )
                for m in unique_missing:
                    suggestions.append(f"Add table `{m}` for FK completeness")

        # ── Multi-table requirement check ──────────────────────────────────────
        if intent.is_complex and len(tables) < 2:
            warnings.append(
                "Complex query detected but only 1 table retrieved — "
                "consider specifying table names in your question"
            )

        # ── Aggregation readiness ──────────────────────────────────────────────
        if intent.is_aggregation and not tables:
            warnings.append("Aggregation query but no tables retrieved")

        # ── Low-score warning ──────────────────────────────────────────────────
        if not tables:
            warnings.append("No tables retrieved — schema may not contain relevant data")

        is_valid = len(tables) > 0 and coverage_score >= 0.3

        return ContextValidation(
            is_valid=is_valid,
            confidence=0.0,    # filled by _compute_confidence
            coverage_score=round(coverage_score, 4),
            fk_complete=fk_complete,
            missing_entities=[e for e in missing_entities if len(e) > 3][:5],
            missing_fk_tables=list(dict.fromkeys(missing_fk_tables))[:3],
            warnings=warnings,
            suggestions=suggestions,
            strategy_used=strategy,
        )

    def _compute_confidence(self, validation: ContextValidation) -> float:
        """Compute a single 0–1 confidence score from validation results."""
        score = validation.coverage_score * 0.5
        if validation.fk_complete:
            score += 0.3
        if not validation.warnings:
            score += 0.2
        elif len(validation.warnings) == 1:
            score += 0.1
        validation.confidence = round(min(score, 1.0), 4)
        return validation.confidence

    # ── Step 5: Context Builder ────────────────────────────────────────────────

    def _build_context(self, tables: list[str], query: str) -> str:
        """
        Build the final context string using schema_index.focused_context()
        with query-aware column filtering and token budget enforcement.
        """
        if not tables:
            return ""
        return self._index.focused_context(
            table_names=tables,
            token_budget=self._token_budget,
            query=query,
        )

    # ── Convenience ───────────────────────────────────────────────────────────

    def explain(self, ctx: RetrievedContext) -> str:
        """Human-readable explanation of what the Context Engine retrieved and why."""
        lines = [
            f"Context Engine Report",
            f"  Query intent   : {'complex' if ctx.intent.is_complex else 'simple'}"
            f", {'aggregation' if ctx.intent.is_aggregation else 'detail'}",
            f"  Strategy       : {ctx.validation.strategy_used}",
            f"  Tables selected: {ctx.tables}",
            f"  Confidence     : {ctx.confidence:.2f}",
            f"  Entity coverage: {ctx.validation.coverage_score:.0%}",
            f"  FK complete    : {ctx.validation.fk_complete}",
            f"  Token estimate : ~{ctx.token_estimate}",
        ]
        if ctx.validation.warnings:
            lines.append(f"  Warnings       :")
            for w in ctx.validation.warnings:
                lines.append(f"    - {w}")
        if ctx.validation.suggestions:
            lines.append(f"  Suggestions    :")
            for s in ctx.validation.suggestions:
                lines.append(f"    - {s}")
        return "\n".join(lines)

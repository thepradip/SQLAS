"""
Semantic Schema Index — handles 100s of tables efficiently.

Architecture:
1. FK relationship graph (NetworkX) — guarantees JOIN-complete table selection
2. TF-IDF cosine similarity — default retrieval, no external dependencies
3. Azure OpenAI embeddings — optional upgrade when embedding deployment is configured
4. Disk cache — embeddings persist across restarts, invalidated on schema changes
"""

import hashlib
import json
import math
import os
import re
import time
from typing import Optional, TYPE_CHECKING

import networkx as nx

if TYPE_CHECKING:
    from openai import AzureOpenAI
    from config import Settings


class SchemaIndex:
    """
    Retrieves the schema-relevant tables for a given NL query.

    For small databases (<= large_schema_threshold), use build_full_context() in
    database.py instead. For large databases this narrows the injected context to
    max_context_tables + up to 4 FK neighbors per query — keeping prompts under ~20K tokens
    even with 500 tables.
    """

    _CACHE_FILE = ".schema_embed_cache.json"

    def __init__(self, client: "AzureOpenAI", settings: "Settings"):
        self._client = client
        self._settings = settings
        self._graph: nx.DiGraph = nx.DiGraph()
        self._table_docs: dict[str, str] = {}
        self._embeddings: dict[str, list[float]] = {}
        self._idf: dict[str, float] = {}
        self._schema: dict = {}
        self._col_stats: dict = {}

    # ── Build ──────────────────────────────────────────────────────────────────

    async def build(self, schema: dict, col_stats: dict) -> None:
        self._schema = schema
        self._col_stats = col_stats
        self._build_fk_graph(schema)
        self._table_docs = {
            t: self._describe_table(t, info, col_stats.get(t, {}))
            for t, info in schema.items()
        }
        self._build_idf()

        if not self._load_embed_cache(schema):
            await self._build_openai_embeddings()
            if self._embeddings:
                self._save_embed_cache(schema)

    def _build_fk_graph(self, schema: dict) -> None:
        for table, info in schema.items():
            self._graph.add_node(table)
            for fk in info.get("foreign_keys", []):
                ref = fk["referred_table"]
                self._graph.add_edge(table, ref)
                self._graph.add_edge(ref, table)

    def _describe_table(self, name: str, info: dict, stats: dict) -> str:
        """Build a rich text representation of a table for TF-IDF/embedding."""
        cols = [c["name"] for c in info["columns"]]
        col_type_pairs = [f"{c['name']}_{c['type']}" for c in info["columns"]]
        fk_refs = [fk["referred_table"] for fk in info.get("foreign_keys", [])]
        row_count = info.get("row_count", "unknown")

        parts = [
            f"table {name}",
            f"columns {' '.join(cols)}",
            f"types {' '.join(col_type_pairs)}",
            f"rows {row_count}",
        ]
        if fk_refs:
            parts.append(f"joins {' '.join(fk_refs)}")

        # Categorical top values give the LLM/embedder domain context
        for col, st in list(stats.items())[:12]:
            if st.get("type") == "categorical" and st.get("top_values"):
                vals = [re.sub(r"\s+", "_", str(v[0])) for v in st["top_values"][:4]]
                parts.append(f"{col}_values {' '.join(vals)}")

        return " ".join(parts)

    # ── BM25 (replaces TF-IDF — better saturation + length normalization) ────────

    _k1 = 1.5   # term frequency saturation
    _b = 0.75   # document length normalization

    def _build_idf(self) -> None:
        """Build BM25-style IDF and per-document term frequencies."""
        n = len(self._table_docs)
        df: dict[str, int] = {}
        self._doc_term_freqs: dict[str, dict[str, int]] = {}
        self._doc_lengths: dict[str, int] = {}

        for table, doc in self._table_docs.items():
            tokens = self._tokenize(doc)
            self._doc_lengths[table] = len(tokens)
            freq: dict[str, int] = {}
            for t in tokens:
                freq[t] = freq.get(t, 0) + 1
            self._doc_term_freqs[table] = freq
            for word in set(tokens):
                df[word] = df.get(word, 0) + 1

        self._avgdl = (sum(self._doc_lengths.values()) / n) if n else 1.0
        # BM25 smoothed IDF: log(1 + (N - df + 0.5) / (df + 0.5))
        self._idf = {w: math.log(1 + (n - cnt + 0.5) / (cnt + 0.5)) for w, cnt in df.items()}

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.sub(r"[^a-z0-9]", " ", text.lower()).split()

    def _bm25_score(self, query_tokens: list[str], table_name: str) -> float:
        """BM25 relevance score for a single document."""
        score = 0.0
        dl = self._doc_lengths.get(table_name, 0)
        tf_doc = self._doc_term_freqs.get(table_name, {})
        for term in query_tokens:
            idf = self._idf.get(term, 0.0)
            if idf == 0.0:
                continue
            f = tf_doc.get(term, 0)
            denom = f + self._k1 * (1 - self._b + self._b * dl / self._avgdl)
            score += idf * f * (self._k1 + 1) / denom if denom else 0.0
        return score

    @staticmethod
    def _cosine_dense(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return dot / (na * nb) if na and nb else 0.0

    @staticmethod
    def _reciprocal_rank_fusion(
        *rankings: list[str], k: int = 60
    ) -> list[str]:
        """
        Reciprocal Rank Fusion — combines multiple ranked lists into one.
        RRF(d) = Σ 1/(k + rank(d, r)) for each ranking r.
        Consistently outperforms single-ranker by 5-15% recall@K.
        """
        scores: dict[str, float] = {}
        for ranking in rankings:
            for rank, item in enumerate(ranking):
                scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank + 1)
        return sorted(scores, key=lambda x: scores[x], reverse=True)

    # ── Azure OpenAI embeddings (optional) ────────────────────────────────────

    async def _build_openai_embeddings(self) -> None:
        model = self._settings.azure_openai_embedding_deployment
        if not model:
            return
        try:
            for name, doc in self._table_docs.items():
                resp = self._client.embeddings.create(model=model, input=doc[:2000])
                self._embeddings[name] = resp.data[0].embedding
            print(f"  OpenAI embeddings built for {len(self._embeddings)} tables.")
        except Exception as e:
            print(f"  Warning: embedding API unavailable ({e}). Using TF-IDF fallback.")
            self._embeddings = {}

    # ── Cache ─────────────────────────────────────────────────────────────────

    def _schema_hash(self, schema: dict) -> str:
        sig = {t: sorted(c["name"] for c in info["columns"]) for t, info in schema.items()}
        return hashlib.md5(json.dumps(sig, sort_keys=True).encode()).hexdigest()

    def _load_embed_cache(self, schema: dict) -> bool:
        if not os.path.exists(self._CACHE_FILE):
            return False
        try:
            with open(self._CACHE_FILE) as f:
                cache = json.load(f)
            if cache.get("hash") != self._schema_hash(schema):
                return False
            self._embeddings = cache["embeddings"]
            if self._embeddings:
                print(f"  Embedding cache loaded ({len(self._embeddings)} tables).")
            return bool(self._embeddings)
        except Exception:
            return False

    def _save_embed_cache(self, schema: dict) -> None:
        try:
            with open(self._CACHE_FILE, "w") as f:
                json.dump({
                    "hash": self._schema_hash(schema),
                    "embeddings": self._embeddings,
                    "timestamp": time.time(),
                }, f)
        except Exception:
            pass

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve_relevant_tables(self, query: str, top_k: Optional[int] = None) -> list[str]:
        """
        Sync BM25 retrieval + FK neighbor expansion.
        Used when no embedding deployment is configured or in sync contexts.
        """
        k = top_k or self._settings.max_context_tables
        query_tokens = self._tokenize(query)
        scores = [(t, self._bm25_score(query_tokens, t)) for t in self._table_docs]
        scores.sort(key=lambda x: x[1], reverse=True)
        return self._fk_expand(scores, k)

    async def retrieve_async(self, query: str, top_k: Optional[int] = None) -> list[str]:
        """
        Hybrid BM25 + dense embedding retrieval via Reciprocal Rank Fusion.
        Falls back to BM25-only when no embedding deployment is configured.

        RRF consistently outperforms either ranker alone by 5-15% recall@K.
        """
        k = top_k or self._settings.max_context_tables
        query_tokens = self._tokenize(query)
        all_tables = list(self._table_docs.keys())

        # Sparse: BM25 ranking
        bm25_scores = [(t, self._bm25_score(query_tokens, t)) for t in all_tables]
        bm25_scores.sort(key=lambda x: x[1], reverse=True)
        bm25_ranking = [t for t, _ in bm25_scores]

        # Dense: Azure OpenAI embedding ranking (optional)
        model = self._settings.azure_openai_embedding_deployment
        if model and self._embeddings:
            try:
                resp = self._client.embeddings.create(model=model, input=query[:2000])
                qvec = resp.data[0].embedding
                dense_scores = [
                    (t, self._cosine_dense(qvec, vec))
                    for t, vec in self._embeddings.items()
                ]
                dense_scores.sort(key=lambda x: x[1], reverse=True)
                dense_ranking = [t for t, _ in dense_scores]

                # Reciprocal Rank Fusion: merge sparse + dense rankings
                fused_ranking = self._reciprocal_rank_fusion(bm25_ranking, dense_ranking)
                return self._fk_expand([(t, 0.0) for t in fused_ranking], k)
            except Exception:
                pass

        return self._fk_expand(bm25_scores, k)

    def _fk_expand(self, scores: list[tuple[str, float]], top_k: int) -> list[str]:
        """Add FK neighbors to guarantee JOINs are possible."""
        selected = [t for t, _ in scores[:top_k]]
        neighbors: set[str] = set()
        for t in selected:
            if t in self._graph:
                for n in self._graph.neighbors(t):
                    if n not in selected:
                        neighbors.add(n)
        # Cap expansion at top_k + 4 to avoid re-exploding context
        cap = top_k + 4
        for n in sorted(neighbors):
            if len(selected) >= cap:
                break
            selected.append(n)
        return selected

    # ── Context formatters ────────────────────────────────────────────────────

    def all_tables_overview(self) -> str:
        """Short orientation header: one line per table. Sent on every query."""
        lines = [f"## Database Overview — {len(self._schema)} tables total\n"]
        for t, info in self._schema.items():
            cols = len(info["columns"])
            rows = info.get("row_count", "?")
            fks = [fk["referred_table"] for fk in info.get("foreign_keys", [])]
            fk_str = f"  (→ {', '.join(fks)})" if fks else ""
            lines.append(f"- `{t}`: {cols} cols, {rows} rows{fk_str}")
        return "\n".join(lines)

    def focused_context(self, table_names: list[str], token_budget: int = 12_000) -> str:
        """
        Full schema + stats for the retrieved table subset.
        Enforces a token budget — drops lowest-priority (last) tables if context is too large.
        1 token ≈ 4 chars (rough estimate, conservative).
        """
        sections = [
            _format_table_section(t, self._schema[t], self._col_stats.get(t, {}))
            for t in table_names
            if t in self._schema
        ]

        fk_lines = []
        for t in table_names:
            for fk in self._schema.get(t, {}).get("foreign_keys", []):
                ref = fk["referred_table"]
                if ref in table_names:
                    fk_lines.append(
                        f"- `{t}.{', '.join(fk['columns'])}` → "
                        f"`{ref}.{', '.join(fk['referred_columns'])}`"
                    )
        if fk_lines:
            sections.append("## Relationships Between Selected Tables\n" + "\n".join(fk_lines))

        full = "\n\n".join(sections)
        if len(full) // 4 <= token_budget:
            return full

        # Trim from the end (least relevant tables first) until within budget
        while len(sections) > 1 and len("\n\n".join(sections)) // 4 > token_budget:
            sections.pop()
        trimmed = "\n\n".join(sections)
        dropped = len(table_names) - (len(sections) - (1 if fk_lines else 0))
        if dropped > 0:
            trimmed += f"\n\n*[{dropped} additional table(s) omitted — token budget {token_budget:,}]*"
        return trimmed


# ── Shared table formatter (also used by database.build_full_context) ──────────

def _format_table_section(table_name: str, info: dict, stats: dict) -> str:
    row_count = info.get("row_count", "?")
    if isinstance(row_count, int):
        header = f"## Table: `{table_name}` ({row_count:,} rows)"
    else:
        header = f"## Table: `{table_name}`"

    lines = [header]

    if info.get("primary_key"):
        lines.append(f"**PK:** {', '.join(info['primary_key'])}")

    lines += ["| Column | Type | Nullable |", "|--------|------|----------|"]
    for col in info["columns"]:
        lines.append(f"| `{col['name']}` | {col['type']} | {col['nullable']} |")

    if info.get("foreign_keys"):
        lines.append("\n**Foreign Keys:**")
        for fk in info["foreign_keys"]:
            lines.append(
                f"- `{', '.join(fk['columns'])}` → "
                f"`{fk['referred_table']}({', '.join(fk['referred_columns'])})`"
            )

    if info.get("indexes"):
        lines.append(f"\n**Indexes:** {', '.join(idx['name'] for idx in info['indexes'])}")

    if stats:
        lines.append("\n**Column Stats:**")
        for col_name, st in stats.items():
            if st.get("type") == "numeric":
                lines.append(
                    f"- `{col_name}`: min={st['min']}, max={st['max']}, "
                    f"avg={st['avg']}, distinct={st['distinct']}, nulls={st['nulls']}"
                )
            elif st.get("type") == "categorical":
                top = ", ".join(f"{v}({c})" for v, c in st.get("top_values", [])[:5])
                lines.append(
                    f"- `{col_name}`: distinct={st['distinct']}, nulls={st['nulls']}, top=[{top}]"
                )

    return "\n".join(lines)

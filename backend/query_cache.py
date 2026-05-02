"""
Multi-level semantic query cache for NL→SQL pipelines.

Level | Mechanism              | Latency  | Tokens saved
------|------------------------|----------|-------------------
  L1  | Exact hash match       | <1ms     | SQL gen + narration (~9,500)
  L2  | Semantic cosine ≥ 0.92 | ~5ms     | SQL gen only (~8,600)
  L4  | SQL result TTL cache   | <1ms     | DB execution time

All persistent state lives in a SQLite sidecar (.query_cache.db).
L2 embeddings are loaded into memory on startup for fast cosine lookup.
Embeddings come from the same Azure OpenAI deployment as schema indexing —
leave AZURE_OPENAI_EMBEDDING_DEPLOYMENT blank to use exact-only caching (still valuable).

Token cost model (adjust in settings if using a different model):
  GPT-4o input: $0.005 / 1K tokens → each L1 hit ≈ $0.047 saved
  At 1000 queries/day, 60% hit rate → ~$28/day = ~$10K/year saved per deployment
"""

import hashlib
import json
import math
import re
import sqlite3
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class CacheHit:
    sql: str
    response: str
    cache_type: str          # "exact" | "semantic"
    similarity: float = 1.0
    tokens_saved: int = 0
    result_from_cache: bool = False


@dataclass
class CacheStats:
    exact_hits: int = 0
    semantic_hits: int = 0
    result_hits: int = 0
    misses: int = 0
    tokens_saved: int = 0

    @property
    def hit_rate(self) -> float:
        hits = self.exact_hits + self.semantic_hits
        total = hits + self.misses
        return hits / total if total else 0.0

    @property
    def cost_saved_usd(self) -> float:
        return (self.tokens_saved / 1000) * 0.005


# Approximate tokens consumed per pipeline step (adjust for your model/schema size)
_TOKENS_SQL_GEN = 8_600     # schema context + boilerplate + query
_TOKENS_NARRATION = 900     # results preview + query + system message


class QueryCache:
    """
    Persistent multi-level cache for NL→SQL queries.
    Thread-safe for single-process use (SQLite WAL mode).
    """

    def __init__(
        self,
        db_path: str = ".query_cache.db",
        similarity_threshold: float = 0.92,
        result_ttl: int = 300,
    ):
        self._db_path = db_path
        self._threshold = similarity_threshold
        self._result_ttl = result_ttl

        # In-memory indexes — loaded once on startup, O(1)/O(n) lookups
        self._id_to_normalized: dict[str, str] = {}
        self._id_to_sql: dict[str, str] = {}
        self._id_to_response: dict[str, str] = {}
        self._id_to_embedding: dict[str, list[float]] = {}

        self._verified_ids: set[str] = set()  # in-memory set of verified cache IDs

        self.stats = CacheStats()
        self._init_db()
        self._load_into_memory()

    # ── Setup ──────────────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS query_cache (
                    id               TEXT PRIMARY KEY,
                    nl_query         TEXT NOT NULL,
                    normalized       TEXT NOT NULL,
                    sql              TEXT NOT NULL,
                    response         TEXT NOT NULL,
                    embedding        TEXT,
                    tokens_saved_est INTEGER DEFAULT 0,
                    hit_count        INTEGER DEFAULT 0,
                    created_at       REAL    NOT NULL,
                    last_used_at     REAL    NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_normalized ON query_cache(normalized);
                CREATE TABLE IF NOT EXISTS result_cache (
                    sql_hash   TEXT PRIMARY KEY,
                    sql        TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    row_count  INTEGER NOT NULL,
                    created_at REAL    NOT NULL,
                    expires_at REAL    NOT NULL
                );
            """)
            # Schema migrations — add columns introduced after initial release
            _add_col(conn, "query_cache", "verified",  "INTEGER DEFAULT 0")
            _add_col(conn, "query_cache", "trace_id",  "TEXT")

    def _load_into_memory(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(
                "SELECT id, normalized, sql, response, embedding, verified FROM query_cache"
            ).fetchall()
        for row_id, normalized, sql, response, emb_json, verified in rows:
            self._id_to_normalized[row_id] = normalized
            self._id_to_sql[row_id] = sql
            self._id_to_response[row_id] = response
            if emb_json:
                try:
                    self._id_to_embedding[row_id] = json.loads(emb_json)
                except Exception:
                    pass
            if verified:
                self._verified_ids.add(row_id)

    # ── Lookup ─────────────────────────────────────────────────────────────────

    def lookup(
        self,
        nl_query: str,
        query_embedding: Optional[list[float]] = None,
    ) -> Optional[CacheHit]:
        """
        Try all cache levels. Returns first hit or None.
        Does NOT check result cache — call lookup_result(sql) separately.
        """
        hit = self._lookup_exact(nl_query)
        if hit:
            return hit

        if query_embedding:
            hit = self._lookup_semantic(query_embedding)
            if hit:
                return hit

        self.stats.misses += 1
        return None

    def _lookup_exact(self, nl_query: str) -> Optional[CacheHit]:
        normalized = _normalize(nl_query)
        for cache_id, stored in self._id_to_normalized.items():
            if stored == normalized:
                self._bump_hit(cache_id)
                self.stats.exact_hits += 1
                saved = _TOKENS_SQL_GEN + _TOKENS_NARRATION
                self.stats.tokens_saved += saved
                return CacheHit(
                    sql=self._id_to_sql[cache_id],
                    response=self._id_to_response[cache_id],
                    cache_type="exact",
                    tokens_saved=saved,
                )
        return None

    def _lookup_semantic(self, query_embedding: list[float]) -> Optional[CacheHit]:
        best_id: Optional[str] = None
        best_score = 0.0
        for cache_id, cached_emb in self._id_to_embedding.items():
            score = _cosine(query_embedding, cached_emb)
            if score > best_score:
                best_score = score
                best_id = cache_id

        if best_id and best_score >= self._threshold:
            self._bump_hit(best_id)
            self.stats.semantic_hits += 1
            self.stats.tokens_saved += _TOKENS_SQL_GEN
            return CacheHit(
                sql=self._id_to_sql[best_id],
                response="",  # response deliberately NOT reused — query is different
                cache_type="semantic",
                similarity=round(best_score, 4),
                tokens_saved=_TOKENS_SQL_GEN,
            )
        return None

    def lookup_result(self, sql: str) -> Optional[dict]:
        """L4: Return cached SQL result if within TTL."""
        sql_hash = _hash(sql.strip().lower())
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT result_json FROM result_cache WHERE sql_hash = ? AND expires_at > ?",
                (sql_hash, time.time()),
            ).fetchone()
        if row:
            self.stats.result_hits += 1
            return json.loads(row[0])
        return None

    # ── Store ──────────────────────────────────────────────────────────────────

    def store(
        self,
        nl_query: str,
        sql: str,
        response: str,
        embedding: Optional[list[float]] = None,
        tokens_used: int = 0,
        trace_id: Optional[str] = None,
    ) -> None:
        normalized = _normalize(nl_query)
        cache_id = _hash(normalized)
        now = time.time()
        emb_json = json.dumps(embedding) if embedding else None

        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO query_cache
                (id, nl_query, normalized, sql, response, embedding,
                 tokens_saved_est, hit_count, created_at, last_used_at, trace_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?)
            """, (cache_id, nl_query, normalized, sql, response,
                  emb_json, tokens_used or _TOKENS_SQL_GEN + _TOKENS_NARRATION, now, now, trace_id))
            conn.commit()

        self._id_to_normalized[cache_id] = normalized
        self._id_to_sql[cache_id] = sql
        self._id_to_response[cache_id] = response
        if embedding:
            self._id_to_embedding[cache_id] = embedding

    # ── Few-shot retrieval ────────────────────────────────────────────────────

    def get_few_shot_examples(
        self,
        nl_query: str,
        query_embedding: Optional[list[float]] = None,
        top_k: int = 3,
    ) -> list[dict]:
        """
        Retrieve the most similar past (question, SQL) pairs for few-shot injection.

        Scoring: dense cosine similarity when embeddings available, else Jaccard overlap.
        Priority: verified entries (user thumbs-up) first, then high-hit-count entries.
        Skips near-exact matches (similarity > 0.98) — those are handled by cache hits.
        """
        if not self._id_to_normalized:
            return []

        # Score every cached query against the current one
        if query_embedding and self._id_to_embedding:
            scored = [
                (cid, _cosine(query_embedding, emb))
                for cid, emb in self._id_to_embedding.items()
            ]
        else:
            scored = [
                (cid, _jaccard(nl_query, stored))
                for cid, stored in self._id_to_normalized.items()
            ]

        scored.sort(key=lambda x: x[1], reverse=True)

        verified: list[dict] = []
        implicit: list[dict] = []

        for cid, score in scored:
            if score > 0.98:      # same query — skip (already a cache hit candidate)
                continue
            if score < 0.20:      # unrelated — stop scanning
                break
            if len(verified) + len(implicit) >= top_k * 3:
                break

            entry = {
                "question": self._id_to_normalized[cid],
                "sql": self._id_to_sql[cid],
                "similarity": round(score, 3),
                "verified": cid in self._verified_ids,
            }
            if cid in self._verified_ids:
                verified.append(entry)
            else:
                implicit.append(entry)

        # Fill slots: verified first, then implicit to reach top_k
        examples = verified[:top_k]
        if len(examples) < top_k:
            examples += implicit[: top_k - len(examples)]
        return examples

    # ── Verification (learning loop) ──────────────────────────────────────────

    def verify_by_trace(self, trace_id: str) -> bool:
        """
        Mark a cache entry as verified using its MLflow trace ID.
        Called automatically when a user submits thumbs-up feedback.
        Verified entries are prioritised in few-shot retrieval.
        """
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT id FROM query_cache WHERE trace_id = ?", (trace_id,)
            ).fetchone()
            if row:
                conn.execute("UPDATE query_cache SET verified = 1 WHERE id = ?", (row[0],))
                conn.commit()
                self._verified_ids.add(row[0])
                return True
        return False

    def store_result(self, sql: str, result: dict, ttl_seconds: Optional[int] = None) -> None:
        """Cache SQL execution result with TTL."""
        sql_hash = _hash(sql.strip().lower())
        now = time.time()
        ttl = ttl_seconds if ttl_seconds is not None else self._result_ttl
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO result_cache
                (sql_hash, sql, result_json, row_count, created_at, expires_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (sql_hash, sql, json.dumps(result),
                  result.get("row_count", 0), now, now + ttl))
            conn.commit()

    def _bump_hit(self, cache_id: str) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "UPDATE query_cache SET hit_count = hit_count + 1, last_used_at = ? WHERE id = ?",
                (time.time(), cache_id),
            )
            conn.commit()

    # ── Analytics ──────────────────────────────────────────────────────────────

    def get_analytics(self) -> dict:
        with sqlite3.connect(self._db_path) as conn:
            total = conn.execute("SELECT COUNT(*) FROM query_cache").fetchone()[0]
            lifetime_hits = conn.execute(
                "SELECT COALESCE(SUM(hit_count), 0) FROM query_cache"
            ).fetchone()[0]
            lifetime_tokens = conn.execute(
                "SELECT COALESCE(SUM(tokens_saved_est * hit_count), 0) FROM query_cache"
            ).fetchone()[0]
            top_queries = conn.execute("""
                SELECT nl_query, hit_count
                FROM query_cache ORDER BY hit_count DESC LIMIT 10
            """).fetchall()
            result_active = conn.execute(
                "SELECT COUNT(*) FROM result_cache WHERE expires_at > ?", (time.time(),)
            ).fetchone()[0]

        return {
            "cached_queries": total,
            "result_cache_active": result_active,
            "lifetime_hits": lifetime_hits,
            "lifetime_tokens_saved": lifetime_tokens,
            "lifetime_cost_saved_usd": round((lifetime_tokens / 1000) * 0.005, 2),
            "session": {
                "exact_hits": self.stats.exact_hits,
                "semantic_hits": self.stats.semantic_hits,
                "result_hits": self.stats.result_hits,
                "misses": self.stats.misses,
                "hit_rate_pct": round(self.stats.hit_rate * 100, 1),
                "tokens_saved": self.stats.tokens_saved,
                "cost_saved_usd": round(self.stats.cost_saved_usd, 4),
            },
            "top_queries": [{"query": q, "hits": h} for q, h in top_queries],
        }

    def invalidate_results(self) -> int:
        """Purge all SQL result cache. Call after any data updates."""
        with sqlite3.connect(self._db_path) as conn:
            count = conn.execute("SELECT COUNT(*) FROM result_cache").fetchone()[0]
            conn.execute("DELETE FROM result_cache")
            conn.commit()
        self.stats.result_hits = 0
        return count

    def clear_all(self) -> None:
        """Wipe all cache entries (query + result)."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("DELETE FROM query_cache")
            conn.execute("DELETE FROM result_cache")
            conn.commit()
        self._id_to_normalized.clear()
        self._id_to_sql.clear()
        self._id_to_response.clear()
        self._id_to_embedding.clear()
        self.stats = CacheStats()

    def size(self) -> int:
        return len(self._id_to_normalized)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _normalize(query: str) -> str:
    return re.sub(r"\s+", " ", query.lower().strip())


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _jaccard(a: str, b: str) -> float:
    """Word-level Jaccard similarity — used as fallback when no embeddings."""
    wa = set(re.sub(r"[^a-z0-9]", " ", a.lower()).split())
    wb = set(re.sub(r"[^a-z0-9]", " ", b.lower()).split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def _add_col(conn: "sqlite3.Connection", table: str, col: str, definition: str) -> None:
    """Safe ALTER TABLE ADD COLUMN — no-op if column already exists."""
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if col not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {definition}")
        conn.commit()

"""
SQLAS Feedback Store — learns from user corrections.

When users mark a query correct (thumbs up), the (question, sql) pair
is stored as a verified gold SQL. Future evaluate_correctness() calls
on the same question auto-use this gold SQL — no manual gold_sql argument needed.

Storage: SQLite sidecar (.sqlas_feedback.db)
Matching: exact normalized question match (fast, zero cost)

Usage:
    from sqlas import FeedbackStore, FeedbackEntry

    store = FeedbackStore()

    # User gives thumbs up → store as gold SQL
    store.store(FeedbackEntry(
        question   = "How many active users?",
        sql        = "SELECT COUNT(*) FROM users WHERE status = 'active'",
        is_correct = True,
        score      = 0.95,
        source     = "user",
    ))

    # Future evaluation auto-uses stored gold SQL
    from sqlas import evaluate_correctness
    result = evaluate_correctness(
        question      = "How many active users?",
        generated_sql = agent_sql,
        llm_judge     = judge,
        feedback_store = store,    # ← auto-provides gold_sql
    )
    print(result.gold_sql_source)  # "feedback_store"
"""

import re
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FeedbackEntry:
    """A single piece of user feedback on a query."""
    question:   str
    sql:        str
    is_correct: bool            # True = thumbs up (store as gold SQL)
    score:      float = 0.0    # SQLAS overall_score at feedback time
    source:     str = "user"   # "user" | "evaluator" | "auto"
    notes:      str = ""
    timestamp:  float = 0.0    # auto-set if 0


class FeedbackStore:
    """
    Persistent store for verified (question → gold SQL) pairs.

    Thread-safe for single-process use (SQLite WAL mode).
    Verified entries are loaded into memory at startup for O(1) lookup.
    """

    def __init__(self, db_path: str = ".sqlas_feedback.db"):
        self._db_path = db_path
        self._gold_cache: dict[str, str] = {}   # normalized_question -> best_sql
        self._init_db()
        self._load_cache()

    def _init_db(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    question    TEXT    NOT NULL,
                    normalized  TEXT    NOT NULL,
                    sql         TEXT    NOT NULL,
                    is_correct  INTEGER NOT NULL DEFAULT 0,
                    score       REAL    DEFAULT 0.0,
                    source      TEXT    DEFAULT 'user',
                    timestamp   REAL    NOT NULL,
                    notes       TEXT    DEFAULT ''
                );
                CREATE INDEX IF NOT EXISTS idx_norm_correct ON feedback(normalized, is_correct);
            """)

    def _load_cache(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(
                "SELECT normalized, sql FROM feedback WHERE is_correct = 1 ORDER BY score DESC, timestamp DESC"
            ).fetchall()
        seen: set[str] = set()
        for norm, sql in rows:
            if norm not in seen:
                self._gold_cache[norm] = sql
                seen.add(norm)

    # ── Public API ─────────────────────────────────────────────────────────────

    def store(self, entry: FeedbackEntry) -> None:
        """
        Store a feedback entry.
        If is_correct=True, the SQL is added to the gold SQL pool for this question.
        If is_correct=False, it's recorded for analytics but not used as gold SQL.
        """
        normalized = _normalize(entry.question)
        timestamp = entry.timestamp or time.time()

        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                INSERT INTO feedback (question, normalized, sql, is_correct, score, source, timestamp, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (entry.question, normalized, entry.sql,
                  int(entry.is_correct), entry.score,
                  entry.source, timestamp, entry.notes))
            conn.commit()

        if entry.is_correct:
            # Keep highest-score gold SQL per question
            existing = self._gold_cache.get(normalized)
            if existing is None:
                self._gold_cache[normalized] = entry.sql
            else:
                # Replace if new entry has higher score
                existing_score = self._get_score_for(normalized, existing)
                if entry.score >= existing_score:
                    self._gold_cache[normalized] = entry.sql

    def get_gold_sql(self, question: str) -> Optional[str]:
        """
        Return the best verified gold SQL for this question (exact match).
        Returns None if no verified feedback exists for this question.
        """
        return self._gold_cache.get(_normalize(question))

    def has_gold(self, question: str) -> bool:
        """True if a verified gold SQL exists for this question."""
        return _normalize(question) in self._gold_cache

    def get_stats(self) -> dict:
        """Summary statistics for the feedback store."""
        with sqlite3.connect(self._db_path) as conn:
            total    = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
            correct  = conn.execute("SELECT COUNT(*) FROM feedback WHERE is_correct=1").fetchone()[0]
            wrong    = conn.execute("SELECT COUNT(*) FROM feedback WHERE is_correct=0").fetchone()[0]
            top      = conn.execute("""
                SELECT question, sql, score FROM feedback
                WHERE is_correct = 1 ORDER BY score DESC LIMIT 10
            """).fetchall()
        return {
            "total_feedback":    total,
            "thumbs_up":         correct,
            "thumbs_down":       wrong,
            "gold_sqls_in_memory": len(self._gold_cache),
            "top_verified": [
                {"question": q, "sql": s[:80] + "...", "score": sc}
                for q, s, sc in top
            ],
        }

    def size(self) -> int:
        """Number of verified gold SQL entries loaded in memory."""
        return len(self._gold_cache)

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _get_score_for(self, normalized: str, sql: str) -> float:
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT MAX(score) FROM feedback WHERE normalized=? AND sql=? AND is_correct=1",
                (normalized, sql),
            ).fetchone()
        return row[0] if row and row[0] is not None else 0.0


def _normalize(question: str) -> str:
    return re.sub(r"\s+", " ", question.lower().strip())

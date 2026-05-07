"""
AriaSQL Training Store — DDL, documentation, and SQL example ingestion.

This is the feature that makes Vanna AI compelling: you train it on your
schema context and it gets progressively smarter. AriaSQL now matches this
with three ingestion types:

  DDL          — CREATE TABLE statements. Helps LLM understand column
                 semantics, relationships, and constraints beyond raw stats.

  Documentation — Business term definitions. Maps "active customer" to
                  "WHERE status = 'active' AND last_login > NOW()-30d".
                  Critical for domain-specific queries that the schema alone
                  can't explain.

  SQL Examples  — Verified (question → SQL) pairs. Injected as few-shot
                  context into the SQL generation prompt.

All three types are retrieved at query time via BM25 similarity and injected
into the system prompt BEFORE the schema context — giving the LLM domain
knowledge it could never derive from column names alone.

Usage:
    store = TrainingStore()

    # Train on DDL
    store.add_ddl(\"\"\"
        CREATE TABLE orders (
            id        INTEGER PRIMARY KEY,
            customer_id INTEGER REFERENCES customers(id),
            total     DECIMAL(10,2),
            status    VARCHAR(20) CHECK (status IN ('pending','completed','cancelled'))
        )
    \"\"\")

    # Train on business documentation
    store.add_documentation(
        "Revenue is defined as SUM(orders.total) WHERE status = 'completed'. "
        "Cancelled and pending orders are excluded from all revenue calculations.",
        title="Revenue Definition"
    )

    # Train on verified SQL examples
    store.add_sql_example(
        question = "What is last month's revenue?",
        sql      = "SELECT SUM(total) FROM orders WHERE status='completed' "
                   "AND created_at >= DATE_TRUNC('month', NOW() - INTERVAL '1 month')"
    )

    # At query time — returns relevant DDL + docs + SQL as a formatted string
    context = store.get_context("Show me total revenue by region this quarter")
"""

import hashlib
import json
import math
import re
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrainingItem:
    id: str
    type: str            # "ddl" | "documentation" | "sql"
    content: str         # the DDL / doc text / SQL
    title: str = ""      # optional label
    question: str = ""   # for sql type: the associated question
    created_at: float = 0.0


class TrainingStore:
    """
    Persistent store for training context — DDL, documentation, SQL examples.
    Items are retrieved at query time via BM25 and injected into the SQL
    generation prompt, teaching the LLM domain-specific knowledge.
    """

    def __init__(self, db_path: str = ".training_store.db"):
        self._db_path = db_path
        # In-memory BM25 index
        self._docs: dict[str, str] = {}          # id -> searchable text
        self._items: dict[str, TrainingItem] = {} # id -> item
        self._idf: dict[str, float] = {}
        self._avgdl: float = 1.0
        self._k1, self._b = 1.5, 0.75
        self._idf_dirty: bool = True   # rebuild IDF before first search
        self._init_db()
        self._load_into_memory()

    # ── Setup ──────────────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS training (
                    id         TEXT PRIMARY KEY,
                    type       TEXT NOT NULL,
                    content    TEXT NOT NULL,
                    title      TEXT DEFAULT '',
                    question   TEXT DEFAULT '',
                    created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_type ON training(type);
            """)

    def _load_into_memory(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(
                "SELECT id, type, content, title, question, created_at FROM training"
            ).fetchall()
        for row in rows:
            item = TrainingItem(id=row[0], type=row[1], content=row[2],
                                title=row[3], question=row[4], created_at=row[5])
            self._items[item.id] = item
            self._docs[item.id] = self._make_search_text(item)
        self._build_idf()

    def _make_search_text(self, item: TrainingItem) -> str:
        """Build searchable text for BM25 indexing."""
        parts = [item.content, item.title, item.question]
        return " ".join(p for p in parts if p)

    # ── BM25 index ─────────────────────────────────────────────────────────────

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.sub(r"[^a-z0-9]", " ", text.lower()).split()

    def _build_idf(self) -> None:
        n = len(self._docs)
        if n == 0:
            return
        df: dict[str, int] = {}
        lens: list[int] = []
        for doc in self._docs.values():
            tokens = self._tokenize(doc)
            lens.append(len(tokens))
            for t in set(tokens):
                df[t] = df.get(t, 0) + 1
        self._avgdl = sum(lens) / n if n else 1.0
        self._idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}

    def _bm25_score(self, query_tokens: list[str], doc_id: str) -> float:
        doc = self._docs.get(doc_id, "")
        tokens = self._tokenize(doc)
        dl = len(tokens)
        tf: dict[str, int] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        score = 0.0
        for t in query_tokens:
            if t not in self._idf:
                continue
            f = tf.get(t, 0)
            denom = f + self._k1 * (1 - self._b + self._b * dl / max(self._avgdl, 1))
            score += self._idf[t] * f * (self._k1 + 1) / denom if denom else 0
        return score

    # ── Public API ─────────────────────────────────────────────────────────────

    def add_ddl(self, ddl: str, title: str = "") -> str:
        """
        Ingest a DDL statement (CREATE TABLE, CREATE VIEW, etc.).
        Helps the LLM understand column semantics, constraints, and
        relationships that raw column names alone don't convey.

        Args:
            ddl:   Full CREATE TABLE / CREATE VIEW statement.
            title: Optional label (e.g. "Orders table DDL").

        Returns:
            Item ID.
        """
        return self._store(TrainingItem(
            id=self._make_id("ddl", ddl), type="ddl",
            content=ddl.strip(), title=title or self._extract_table_name(ddl),
        ))

    def add_documentation(self, text: str, title: str = "") -> str:
        """
        Ingest business documentation or term definitions.
        This is the highest-value training type: it maps business language
        to SQL patterns the LLM would otherwise have to guess.

        Examples:
            "Revenue = SUM(orders.total) WHERE status = 'completed'"
            "Active customer: last_login within 30 days AND account_status = 'active'"
            "Churn: customers with no order in the last 90 days"

        Args:
            text:  Documentation text.
            title: Optional heading.

        Returns:
            Item ID.
        """
        return self._store(TrainingItem(
            id=self._make_id("doc", text), type="documentation",
            content=text.strip(), title=title,
        ))

    def add_sql_example(self, question: str, sql: str) -> str:
        """
        Ingest a verified (question → SQL) pair as a few-shot example.
        These are injected directly into the SQL generation prompt when
        similar questions are asked.

        Args:
            question: Natural language question.
            sql:      The correct SQL for that question.

        Returns:
            Item ID.
        """
        return self._store(TrainingItem(
            id=self._make_id("sql", question + sql), type="sql",
            content=sql.strip(), question=question.strip(),
            title=question[:80],
        ))

    def get_context(self, query: str, top_k: int = 5) -> list["TrainingItem"]:
        """
        Retrieve the most relevant training items for a query.
        Returns a mix of DDL, documentation, and SQL examples scored by BM25.

        Args:
            query:  User's natural language question.
            top_k:  Max items to return (default 5).

        Returns:
            List of TrainingItem sorted by relevance (highest first).
        """
        if not self._docs:
            return []
        if self._idf_dirty:        # lazy rebuild — only when actually needed
            self._build_idf()
            self._idf_dirty = False
        query_tokens = self._tokenize(query)
        scores = [(doc_id, self._bm25_score(query_tokens, doc_id))
                  for doc_id in self._docs]
        scores.sort(key=lambda x: x[1], reverse=True)
        return [self._items[doc_id] for doc_id, score in scores[:top_k] if score > 0]

    def format_context(self, query: str, top_k: int = 5) -> str:
        """
        Return a formatted string of relevant training context for prompt injection.
        Returns empty string if no relevant items found.
        """
        items = self.get_context(query, top_k)
        if not items:
            return ""

        sections: list[str] = []

        ddl_items  = [i for i in items if i.type == "ddl"]
        doc_items  = [i for i in items if i.type == "documentation"]
        sql_items  = [i for i in items if i.type == "sql"]

        if ddl_items:
            sections.append("## Schema Context (DDL)")
            for item in ddl_items:
                if item.title:
                    sections.append(f"-- {item.title}")
                sections.append(f"```sql\n{item.content}\n```")

        if doc_items:
            sections.append("## Business Definitions")
            for item in doc_items:
                prefix = f"**{item.title}:** " if item.title else ""
                sections.append(f"- {prefix}{item.content}")

        if sql_items:
            sections.append("## Verified SQL Examples")
            for item in sql_items:
                sections.append(f"**Q:** {item.question}")
                sections.append(f"```sql\n{item.content}\n```")

        return "\n".join(sections)

    def list_all(self) -> list[dict]:
        """List all training items with metadata."""
        return [
            {"id": i.id, "type": i.type, "title": i.title or i.question[:60],
             "created_at": i.created_at}
            for i in sorted(self._items.values(), key=lambda x: x.created_at, reverse=True)
        ]

    def delete(self, item_id: str) -> bool:
        """Remove a training item by ID."""
        if item_id not in self._items:
            return False
        del self._items[item_id]
        del self._docs[item_id]
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("DELETE FROM training WHERE id = ?", (item_id,))
            conn.commit()
        self._build_idf()
        return True

    def size(self) -> dict:
        counts = {"ddl": 0, "documentation": 0, "sql": 0}
        for item in self._items.values():
            counts[item.type] = counts.get(item.type, 0) + 1
        counts["total"] = sum(counts.values())
        return counts

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _make_id(self, prefix: str, text: str) -> str:
        return prefix + "_" + hashlib.sha256(text.encode()).hexdigest()[:12]

    @staticmethod
    def _extract_table_name(ddl: str) -> str:
        m = re.search(r"CREATE\s+(?:TABLE|VIEW)\s+(?:IF\s+NOT\s+EXISTS\s+)?[`\"']?(\w+)", ddl, re.IGNORECASE)
        return m.group(1) if m else "DDL"

    def _store(self, item: TrainingItem) -> str:
        if not item.created_at:
            item.created_at = time.time()
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO training (id, type, content, title, question, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (item.id, item.type, item.content, item.title, item.question, item.created_at))
            conn.commit()
        self._items[item.id] = item
        self._docs[item.id] = self._make_search_text(item)
        self._idf_dirty = True   # mark dirty — rebuild before next search
        return item.id

"""
AriaSQL Vector Store — ChromaDB-backed semantic search for schema retrieval.

Three collections built once when the database connects:

  tables        — one embedding per table (rich description + stats)
  columns       — one embedding per column (enables "find table via column name")
  relationships — one embedding per JOIN path (helps multi-table query routing)

Both the ChromaDB index and the embedding cache persist to disk.
On restart the store loads from disk — no re-embedding needed unless schema changes.

Embedding strategy:
  1. Azure OpenAI (if AZURE_OPENAI_EMBEDDING_DEPLOYMENT configured) — best quality
  2. ChromaDB default (all-MiniLM-L6-v2 via sentence-transformers) — free, local
  3. Fallback to empty embeddings if neither available (vector search disabled)

Usage:
    store = VectorStore(settings=settings, azure_client=client)
    store.build(schema, col_stats)        # once on startup

    # Semantic table search
    results = store.search_tables("average revenue by customer region", top_k=5)
    # → [{"name": "orders", "score": 0.92, ...}, ...]

    # Column-level search (find tables that have matching column names)
    results = store.search_columns("blood pressure measurement", top_k=10)
    # → [{"table": "patients", "column": "blood_pressure_status", "score": 0.89}, ...]

    # Combined search — returns table names
    tables = store.find_relevant_tables("count active users by country", top_k=8)
    # → ["users", "countries", "user_sessions"]
"""

import hashlib
import json
import os
import re
from typing import Optional, TYPE_CHECKING

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False
    print("  Warning: chromadb not installed — vector search disabled. "
          "Run: pip install chromadb")

if TYPE_CHECKING:
    from config import Settings


_CHROMA_DIR  = ".chroma_db"
_HASH_FILE   = ".vector_store_hash.txt"


class VectorStore:
    """
    ChromaDB-backed semantic search over database schema metadata.
    Persists to disk — rebuilt only when schema changes.
    """

    def __init__(
        self,
        settings: "Settings | None" = None,
        azure_client=None,          # Azure OpenAI client for embeddings
        persist_dir: str = _CHROMA_DIR,
    ):
        self._settings      = settings
        self._azure_client  = azure_client
        self._persist_dir   = persist_dir
        self._client        = None
        self._col_tables    = None
        self._col_columns   = None
        self._col_rels      = None
        self._embed_fn      = None
        self._schema_hash   = ""
        self._ready         = False

        if not HAS_CHROMA:
            return

        self._init_chroma()

    def _init_chroma(self) -> None:
        """Initialise persistent ChromaDB client."""
        os.makedirs(self._persist_dir, exist_ok=True)
        self._client = chromadb.PersistentClient(path=self._persist_dir)
        self._embed_fn = self._make_embedding_fn()

        self._col_tables  = self._client.get_or_create_collection(
            "tables",        embedding_function=self._embed_fn,
            metadata={"description": "One doc per database table"},
        )
        self._col_columns = self._client.get_or_create_collection(
            "columns",       embedding_function=self._embed_fn,
            metadata={"description": "One doc per column across all tables"},
        )
        self._col_rels    = self._client.get_or_create_collection(
            "relationships", embedding_function=self._embed_fn,
            metadata={"description": "JOIN paths and table relationships"},
        )

    def _make_embedding_fn(self):
        """
        Return the best available embedding function:
        1. Azure OpenAI (highest quality)
        2. ChromaDB default (sentence-transformers, free local)
        """
        emb_deployment = getattr(self._settings, "azure_openai_embedding_deployment", "")
        if self._azure_client and emb_deployment:
            return _AzureEmbeddingFn(self._azure_client, emb_deployment)

        # ChromaDB's built-in sentence-transformers embedding
        try:
            from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
            print("  VectorStore: using local sentence-transformers embeddings (free)")
            return DefaultEmbeddingFunction()
        except Exception:
            print("  VectorStore: no embedding function available — vector search degraded")
            return None

    # ── Build ──────────────────────────────────────────────────────────────────

    def build(self, schema: dict, col_stats: dict) -> None:
        """
        Build all three collections from schema.
        Checks hash — skips rebuild if schema unchanged.
        """
        if not HAS_CHROMA or not self._client:
            return

        new_hash = _schema_hash(schema)

        # Check persisted hash
        if os.path.exists(_HASH_FILE):
            try:
                saved = open(_HASH_FILE).read().strip()
                if saved == new_hash:
                    t_count = self._col_tables.count()  if self._col_tables  else 0
                    c_count = self._col_columns.count() if self._col_columns else 0
                    # Both collections must have data — if either is empty the
                    # previous build was incomplete (e.g. interrupted) and we rebuild
                    if t_count > 0 and c_count > 0:
                        print(f"  VectorStore: loaded from cache "
                              f"({t_count} tables, {c_count} columns)")
                        self._schema_hash = new_hash
                        self._ready = True
                        return
            except Exception:
                pass

        print(f"  VectorStore: building collections for {len(schema)} tables...")

        # Clear existing data
        for col in [self._col_tables, self._col_columns, self._col_rels]:
            if col:
                existing = col.get()
                if existing["ids"]:
                    col.delete(ids=existing["ids"])

        self._build_tables(schema, col_stats)
        self._build_columns(schema, col_stats)
        self._build_relationships(schema)

        # Save hash
        try:
            with open(_HASH_FILE, "w") as f:
                f.write(new_hash)
        except Exception:
            pass

        self._schema_hash = new_hash
        self._ready = True
        print(f"  VectorStore: ready — {self._col_tables.count()} tables, "
              f"{self._col_columns.count()} columns, "
              f"{self._col_rels.count()} relationships")

    def _build_tables(self, schema: dict, col_stats: dict) -> None:
        """One document per table — rich description for semantic search."""
        ids, docs, metas = [], [], []

        for tname, info in schema.items():
            cols      = info.get("columns", [])
            row_count = info.get("row_count", "?")
            pk        = ", ".join(info.get("primary_key", []))
            fks       = info.get("foreign_keys", [])

            # Build a rich text description
            col_names = [c["name"] for c in cols[:40]]  # cap at 40
            fk_descs  = [
                f"{', '.join(fk.get('columns',[]))} → {fk.get('referred_table','?')}"
                for fk in fks
            ]

            # Collect notable column values from stats for semantic richness
            notable_vals: list[str] = []
            stats = col_stats.get(tname, {})
            for cname, st in list(stats.items())[:10]:
                if st.get("type") == "categorical" and st.get("top_values"):
                    vals = [str(v[0]) for v in st["top_values"][:3]]
                    notable_vals.append(f"{cname}: {', '.join(vals)}")

            doc_parts = [
                f"Table: {tname}",
                f"Rows: {row_count}",
                f"Primary key: {pk}" if pk else "",
                f"Columns: {', '.join(col_names)}",
                f"Foreign keys: {'; '.join(fk_descs)}" if fk_descs else "",
                f"Sample values: {'; '.join(notable_vals)}" if notable_vals else "",
            ]
            doc = ". ".join(p for p in doc_parts if p)

            ids.append(f"table_{tname}")
            docs.append(doc)
            metas.append({
                "name":       tname,
                "row_count":  row_count if isinstance(row_count, int) else 0,
                "col_count":  len(cols),
                "has_fk":     len(fks) > 0,
                "pk":         pk,
                "col_names":  ",".join(col_names[:20]),
            })

        if ids:
            self._col_tables.add(ids=ids, documents=docs, metadatas=metas)

    def _build_columns(self, schema: dict, col_stats: dict) -> None:
        """One document per column — enables "find table via column name/description"."""
        ids, docs, metas = [], [], []

        for tname, info in schema.items():
            pk_set = set(info.get("primary_key", []))
            fk_cols = {c for fk in info.get("foreign_keys", [])
                       for c in fk.get("columns", [])}
            stats = col_stats.get(tname, {})

            for col in info.get("columns", []):
                cname  = col["name"]
                ctype  = col["type"]
                cid    = f"col_{tname}_{cname}"
                st     = stats.get(cname, {})

                # Build column description
                parts = [
                    f"Column: {cname}",
                    f"Table: {tname}",
                    f"Type: {ctype}",
                ]
                if cname in pk_set:  parts.append("Role: primary key")
                if cname in fk_cols: parts.append("Role: foreign key")

                if st.get("type") == "numeric":
                    parts.append(f"Range: {st.get('min')} to {st.get('max')}, avg {st.get('avg')}")
                elif st.get("type") == "categorical" and st.get("top_values"):
                    vals = [str(v[0]) for v in st["top_values"][:5]]
                    parts.append(f"Values: {', '.join(vals)}")
                    parts.append(f"Distinct count: {st.get('distinct')}")

                doc = ". ".join(p for p in parts if p)

                ids.append(cid)
                docs.append(doc)
                metas.append({
                    "table":  tname,
                    "name":   cname,
                    "type":   ctype,
                    "is_pk":  cname in pk_set,
                    "is_fk":  cname in fk_cols,
                })

        if ids:
            # ChromaDB has a batch size limit — chunk if needed
            batch = 500
            for i in range(0, len(ids), batch):
                self._col_columns.add(
                    ids=ids[i:i+batch],
                    documents=docs[i:i+batch],
                    metadatas=metas[i:i+batch],
                )

    def _build_relationships(self, schema: dict) -> None:
        """One document per relationship — helps multi-table query routing."""
        ids, docs, metas = [], [], []

        for tname, info in schema.items():
            for fk in info.get("foreign_keys", []):
                ref     = fk.get("referred_table", "")
                cols    = ", ".join(fk.get("columns", []))
                rcols   = ", ".join(fk.get("referred_columns", []))
                rel_id  = f"rel_{tname}_{ref}"

                doc = (f"JOIN relationship: {tname} connects to {ref} "
                       f"via {tname}.{cols} = {ref}.{rcols}. "
                       f"Use this JOIN when querying data across {tname} and {ref}.")

                ids.append(rel_id)
                docs.append(doc)
                metas.append({
                    "from_table": tname,
                    "to_table":   ref,
                    "type":       "fk",
                    "via":        cols,
                })

        if ids:
            self._col_rels.add(ids=ids, documents=docs, metadatas=metas)

    # ── Search API ─────────────────────────────────────────────────────────────

    def search_tables(self, query: str, top_k: int = 8) -> list[dict]:
        """Semantic search over table descriptions."""
        if not self._ready or not self._col_tables:
            return []
        try:
            res = self._col_tables.query(
                query_texts=[query], n_results=min(top_k, self._col_tables.count() or 1)
            )
            return [
                {"name": m["name"], "score": 1 - d, **m}
                for m, d in zip(res["metadatas"][0], res["distances"][0])
            ]
        except Exception:
            return []

    def search_columns(self, query: str, top_k: int = 15) -> list[dict]:
        """Semantic search over column descriptions — finds tables via column names."""
        if not self._ready or not self._col_columns:
            return []
        try:
            res = self._col_columns.query(
                query_texts=[query], n_results=min(top_k, self._col_columns.count() or 1)
            )
            return [
                {"table": m["table"], "column": m["name"], "score": 1 - d, **m}
                for m, d in zip(res["metadatas"][0], res["distances"][0])
            ]
        except Exception:
            return []

    def search_relationships(self, query: str, top_k: int = 5) -> list[dict]:
        """Search JOIN relationship descriptions."""
        if not self._ready or not self._col_rels:
            return []
        try:
            count = self._col_rels.count()
            if count == 0:
                return []
            res = self._col_rels.query(
                query_texts=[query], n_results=min(top_k, count)
            )
            return [
                {"from": m["from_table"], "to": m["to_table"], "score": 1 - d}
                for m, d in zip(res["metadatas"][0], res["distances"][0])
            ]
        except Exception:
            return []

    def find_relevant_tables(self, query: str, top_k: int = 8) -> list[str]:
        """
        Combined table + column search.
        Returns deduplicated table names ordered by relevance.
        Highest-impact method for the Context Engine.
        """
        table_results  = self.search_tables(query, top_k=top_k)
        column_results = self.search_columns(query, top_k=top_k * 2)

        # Score tables: table-level score + bonus from column matches
        scores: dict[str, float] = {}

        for r in table_results:
            scores[r["name"]] = scores.get(r["name"], 0) + r["score"] * 1.0

        for r in column_results:
            t = r["table"]
            scores[t] = scores.get(t, 0) + r["score"] * 0.6   # column match is secondary

        # Relationship hints
        for r in self.search_relationships(query, top_k=3):
            for t in (r["from"], r["to"]):
                scores[t] = scores.get(t, 0) + 0.2

        ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [t for t, _ in ordered[:top_k]]

    @property
    def ready(self) -> bool:
        return self._ready


# ── Azure embedding function wrapper ──────────────────────────────────────────

class _AzureEmbeddingFn:
    """Wraps Azure OpenAI embeddings for ChromaDB."""

    def __init__(self, client, deployment: str):
        self._client = client
        self._deployment = deployment

    def __call__(self, input: list[str]) -> list[list[float]]:
        try:
            resp = self._client.embeddings.create(
                model=self._deployment,
                input=[t[:2000] for t in input],
            )
            return [r.embedding for r in resp.data]
        except Exception as e:
            print(f"  Azure embedding failed: {e}. Falling back to zeros.")
            return [[0.0] * 1536] * len(input)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _schema_hash(schema: dict) -> str:
    sig = {t: sorted(c["name"] for c in info.get("columns", []))
           for t, info in schema.items()}
    return hashlib.md5(json.dumps(sig, sort_keys=True).encode()).hexdigest()

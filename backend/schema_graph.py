"""
AriaSQL Schema Graph — rich semantic graph built once when database connects.

Extends the basic FK graph with meaningful semantic relationships:

  FK edges           — explicit foreign key relationships (strength 1.0)
  Naming edges       — columns with similar names suggest implicit JOIN paths
                       (orders.customer_id ↔ customers.id  even without FK)
  Type domain edges  — tables sharing the same column domain
                       (both have "status", "created_at", "amount")
  Hierarchy edges    — parent-child naming patterns
                       (categories → sub_categories, users → user_roles)
  Semantic edges     — tables with high embedding similarity (built from ChromaDB)

Graph nodes carry rich metadata used by the Context Engine reranker:
  - Table nodes: row_count, col_count, has_pk, fk_degree, analytical_score
  - Column nodes: type, is_pk, is_fk, cardinality, domain_tag

Persists to .schema_graph.json — rebuilt only when schema hash changes.
"""

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass, field
from typing import Optional

import networkx as nx


# ── Domain tags for column classification ────────────────────────────────────

_DOMAIN_TAGS = {
    "id":          ["_id", "id_", "uuid", "guid", "key"],
    "temporal":    ["created_at", "updated_at", "deleted_at", "date", "time",
                    "timestamp", "at", "on", "since", "until"],
    "status":      ["status", "state", "flag", "active", "enabled", "is_",
                    "has_", "type", "kind", "category"],
    "amount":      ["amount", "total", "price", "cost", "revenue", "salary",
                    "balance", "fee", "rate", "value", "sum"],
    "count":       ["count", "qty", "quantity", "num_", "number_", "cnt"],
    "name":        ["name", "title", "label", "description", "desc",
                    "text", "comment", "note", "alias"],
    "location":    ["country", "city", "state", "region", "address",
                    "zip", "postal", "lat", "lon", "geo"],
    "person":      ["email", "phone", "user", "customer", "patient",
                    "employee", "staff", "doctor", "agent"],
}


def _tag_column(col_name: str) -> str:
    """Assign a semantic domain tag to a column name."""
    lower = col_name.lower()
    for tag, patterns in _DOMAIN_TAGS.items():
        for p in patterns:
            if p in lower:
                return tag
    return "other"


def _name_similarity(a: str, b: str) -> float:
    """
    Similarity between two identifier strings.
    Handles singular/plural ("customer" vs "customers"),
    underscore-split words ("customer_id" vs "customers"),
    and Jaccard token overlap.
    """
    al, bl = a.lower(), b.lower()

    # Exact match
    if al == bl:
        return 1.0

    # Singular/plural stem match (strip trailing 's' or 'es')
    if al.rstrip("s") == bl.rstrip("s") and len(al.rstrip("s")) > 2:
        return 0.9

    ta = set(re.sub(r"[^a-z0-9]", " ", al).split())
    tb = set(re.sub(r"[^a-z0-9]", " ", bl).split())
    if not ta or not tb:
        return 0.0

    # Stem-aware token match: "customers" → "customer", match against "customer"
    ta_stem = {t.rstrip("s") for t in ta if len(t) > 2}
    tb_stem = {t.rstrip("s") for t in tb if len(t) > 2}
    stem_overlap = ta_stem & tb_stem
    if stem_overlap:
        return len(stem_overlap) / max(len(ta_stem), len(tb_stem))

    return len(ta & tb) / len(ta | tb)


# ── Graph builder ─────────────────────────────────────────────────────────────

class SchemaGraph:
    """
    Rich semantic graph over the database schema.

    Nodes
    ─────
    table:{name}   — one per table
    col:{table}.{col} — one per column (for column-level search)

    Edges
    ─────
    fk        — explicit FK constraint (strongest)
    naming    — column name similarity suggests implicit relationship
    domain    — tables sharing the same column domain type
    hierarchy — naming pattern (parent/child table names)
    """

    CACHE_FILE = ".schema_graph.json"

    def __init__(self):
        self.G: nx.DiGraph = nx.DiGraph()
        self._schema_hash: str = ""

    def build(self, schema: dict, col_stats: dict) -> None:
        """
        Build the full graph from introspected schema.
        Call once on startup. Automatically uses cache if schema unchanged.
        """
        new_hash = self._compute_hash(schema)
        if self._load_cache(new_hash):
            return

        self.G = nx.DiGraph()
        self._schema_hash = new_hash
        self._add_table_nodes(schema, col_stats)
        self._add_fk_edges(schema)
        self._add_naming_edges(schema)
        self._add_domain_edges(schema)
        self._add_hierarchy_edges(schema)
        self._save_cache()

        print(f"  Schema Graph: {self.G.number_of_nodes()} nodes, "
              f"{self.G.number_of_edges()} edges "
              f"({self._count_edge_type('fk')} FK, "
              f"{self._count_edge_type('naming')} naming, "
              f"{self._count_edge_type('domain')} domain, "
              f"{self._count_edge_type('hierarchy')} hierarchy)")

    # ── Node construction ──────────────────────────────────────────────────────

    def _add_table_nodes(self, schema: dict, col_stats: dict) -> None:
        for tname, info in schema.items():
            cols     = info.get("columns", [])
            has_pk   = bool(info.get("primary_key"))
            fk_count = len(info.get("foreign_keys", []))
            row_count = info.get("row_count", 0)

            # Analytical score: tables with many numeric cols are good for GROUP BY
            num_cols = sum(
                1 for c in cols
                if any(t in c["type"].upper()
                       for t in ("INT","REAL","FLOAT","DECIMAL","NUMERIC"))
            )
            analytical = num_cols / max(len(cols), 1)

            self.G.add_node(
                f"table:{tname}",
                kind="table",
                name=tname,
                row_count=row_count if isinstance(row_count, int) else 0,
                col_count=len(cols),
                has_pk=has_pk,
                fk_degree=fk_count,
                analytical_score=round(analytical, 3),
                col_names=[c["name"] for c in cols],
                col_types={c["name"]: c["type"] for c in cols},
                col_domains={c["name"]: _tag_column(c["name"]) for c in cols},
            )

            # Column-level nodes
            for col in cols:
                cname = col["name"]
                st = col_stats.get(tname, {}).get(cname, {})
                self.G.add_node(
                    f"col:{tname}.{cname}",
                    kind="column",
                    table=tname,
                    name=cname,
                    col_type=col["type"],
                    domain=_tag_column(cname),
                    is_pk=cname in (info.get("primary_key") or []),
                    is_fk=any(cname in fk.get("columns", [])
                              for fk in info.get("foreign_keys", [])),
                    cardinality=st.get("distinct", 0),
                    nulls=st.get("nulls", 0),
                )

    # ── Edge types ─────────────────────────────────────────────────────────────

    def _add_fk_edges(self, schema: dict) -> None:
        """FK edges — strongest relationship signal (weight 1.0)."""
        for tname, info in schema.items():
            for fk in info.get("foreign_keys", []):
                ref = fk.get("referred_table", "")
                if ref and ref in schema:
                    self.G.add_edge(
                        f"table:{tname}", f"table:{ref}",
                        edge_type="fk", weight=1.0,
                        via_cols=fk.get("columns", []),
                        ref_cols=fk.get("referred_columns", []),
                    )
                    # Bidirectional
                    self.G.add_edge(
                        f"table:{ref}", f"table:{tname}",
                        edge_type="fk", weight=1.0,
                        via_cols=fk.get("referred_columns", []),
                        ref_cols=fk.get("columns", []),
                    )

    def _add_naming_edges(self, schema: dict) -> None:
        """
        Naming edges — when column names suggest an implicit JOIN path
        even without an explicit FK constraint.

        Example: orders.customer_id ↔ customers.id (no FK declared)
        → naming similarity between "customer_id" and "customers.id"
        """
        tables = list(schema.keys())
        for i, ta in enumerate(tables):
            cols_a = [c["name"] for c in schema[ta].get("columns", [])]
            for tb in tables[i+1:]:
                if ta == tb:
                    continue
                cols_b = [c["name"] for c in schema[tb].get("columns", [])]

                # Check if any column in A looks like it references B
                max_sim = 0.0
                best_pair = (None, None)
                for ca in cols_a:
                    # e.g. "customer_id" in orders → "customers" table
                    ca_clean = ca.lower().replace("_id", "").replace("id_", "")
                    if _name_similarity(ca_clean, tb.lower()) >= 0.6:
                        for cb in cols_b:
                            if cb.lower() in ("id",) or cb.lower().endswith("_id"):
                                sim = 0.7
                                if sim > max_sim:
                                    max_sim = sim
                                    best_pair = (ca, cb)
                    # Also check direct column name overlap
                    for cb in cols_b:
                        sim = _name_similarity(ca, cb)
                        if sim > 0.7 and sim > max_sim:
                            max_sim = sim
                            best_pair = (ca, cb)

                if max_sim >= 0.65:
                    # Only add if FK edge doesn't already exist
                    if not self.G.has_edge(f"table:{ta}", f"table:{tb}"):
                        self.G.add_edge(
                            f"table:{ta}", f"table:{tb}",
                            edge_type="naming", weight=round(max_sim * 0.7, 3),
                            via_cols=[best_pair[0]], ref_cols=[best_pair[1]],
                        )
                        self.G.add_edge(
                            f"table:{tb}", f"table:{ta}",
                            edge_type="naming", weight=round(max_sim * 0.7, 3),
                        )

    def _add_domain_edges(self, schema: dict) -> None:
        """
        Domain edges — tables that share column domains are likely co-queried.
        Both 'orders' and 'invoices' having 'amount', 'status', 'created_at'
        suggests they're in the same business domain.
        """
        table_domains: dict[str, set[str]] = {}
        for tname, info in schema.items():
            domains = {_tag_column(c["name"]) for c in info.get("columns", [])}
            domains.discard("other")
            table_domains[tname] = domains

        tables = list(schema.keys())
        for i, ta in enumerate(tables):
            for tb in tables[i+1:]:
                shared = table_domains[ta] & table_domains[tb]
                if len(shared) >= 2:
                    weight = min(len(shared) * 0.15, 0.5)
                    if not self.G.has_edge(f"table:{ta}", f"table:{tb}"):
                        self.G.add_edge(
                            f"table:{ta}", f"table:{tb}",
                            edge_type="domain", weight=weight,
                            shared_domains=sorted(shared),
                        )
                        self.G.add_edge(
                            f"table:{tb}", f"table:{ta}",
                            edge_type="domain", weight=weight,
                            shared_domains=sorted(shared),
                        )

    def _add_hierarchy_edges(self, schema: dict) -> None:
        """
        Hierarchy edges — naming patterns suggest parent-child relationships.
        'categories' → 'sub_categories', 'users' → 'user_roles'
        """
        tables = list(schema.keys())
        for ta in tables:
            for tb in tables:
                if ta == tb:
                    continue
                # tb is a child of ta if ta's name is a prefix of tb
                if (tb.lower().startswith(ta.lower() + "_")
                        or tb.lower().startswith(ta.lower()[:-1] + "_")):
                    if not self.G.has_edge(f"table:{ta}", f"table:{tb}"):
                        self.G.add_edge(
                            f"table:{ta}", f"table:{tb}",
                            edge_type="hierarchy", weight=0.4,
                        )

    # ── Query API ──────────────────────────────────────────────────────────────

    def get_table_node(self, table_name: str) -> Optional[dict]:
        return self.G.nodes.get(f"table:{table_name}")

    def get_neighbors(
        self,
        table_name: str,
        edge_types: list[str] | None = None,
        min_weight: float = 0.0,
    ) -> list[tuple[str, float, str]]:
        """
        Return neighboring tables with their edge weight and type.
        Returns list of (table_name, weight, edge_type).
        """
        results = []
        node_id = f"table:{table_name}"
        if node_id not in self.G:
            return results
        for _, nbr, data in self.G.out_edges(node_id, data=True):
            if not nbr.startswith("table:"):
                continue
            etype  = data.get("edge_type", "unknown")
            weight = data.get("weight", 0.0)
            if edge_types and etype not in edge_types:
                continue
            if weight < min_weight:
                continue
            results.append((nbr.replace("table:", ""), weight, etype))
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def get_columns_for_domain(self, domain: str) -> list[tuple[str, str]]:
        """Return (table, column) pairs for a given domain tag."""
        return [
            (data["table"], data["name"])
            for _, data in self.G.nodes(data=True)
            if data.get("kind") == "column" and data.get("domain") == domain
        ]

    def shortest_join_path(self, from_table: str, to_table: str) -> list[str]:
        """Find the shortest JOIN path between two tables via graph traversal."""
        try:
            path = nx.shortest_path(
                self.G,
                f"table:{from_table}",
                f"table:{to_table}",
                weight=lambda u, v, d: 1.0 - d.get("weight", 0.5),
            )
            return [p.replace("table:", "") for p in path if p.startswith("table:")]
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def analytical_tables(self, top_k: int = 10) -> list[str]:
        """Return most analytically useful tables (numeric cols, high row counts)."""
        scored = [
            (data["analytical_score"] + math.log1p(data["row_count"]) * 0.01, name)
            for name, data in self.G.nodes(data=True)
            if data.get("kind") == "table"
        ]
        scored.sort(reverse=True)
        return [name for _, name in scored[:top_k]]

    # ── Persistence ────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_hash(schema: dict) -> str:
        sig = {t: sorted(c["name"] for c in info.get("columns", []))
               for t, info in schema.items()}
        return hashlib.md5(json.dumps(sig, sort_keys=True).encode()).hexdigest()

    def _count_edge_type(self, etype: str) -> int:
        return sum(1 for _, _, d in self.G.edges(data=True)
                   if d.get("edge_type") == etype)

    def _save_cache(self) -> None:
        data = {
            "hash": self._schema_hash,
            "nodes": [
                {"id": n, **{k: v for k, v in d.items()
                             if isinstance(v, (str, int, float, bool, list, dict))}}
                for n, d in self.G.nodes(data=True)
            ],
            "edges": [
                {"src": u, "dst": v, **{k: v2 for k, v2 in d.items()
                                        if isinstance(v2, (str, int, float, bool, list))}}
                for u, v, d in self.G.edges(data=True)
            ],
        }
        try:
            with open(self.CACHE_FILE, "w") as f:
                json.dump(data, f)
        except Exception:
            pass

    def _load_cache(self, expected_hash: str) -> bool:
        if not os.path.exists(self.CACHE_FILE):
            return False
        try:
            with open(self.CACHE_FILE) as f:
                data = json.load(f)
            if data.get("hash") != expected_hash:
                return False
            self.G = nx.DiGraph()
            for n in data["nodes"]:
                nid = n.pop("id")
                self.G.add_node(nid, **n)
            for e in data["edges"]:
                src = e.pop("src"); dst = e.pop("dst")
                self.G.add_edge(src, dst, **e)
            self._schema_hash = expected_hash
            print(f"  Schema Graph loaded from cache ({self.G.number_of_nodes()} nodes, "
                  f"{self.G.number_of_edges()} edges)")
            return True
        except Exception:
            return False

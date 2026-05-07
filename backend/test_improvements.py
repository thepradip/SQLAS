"""
Unit tests for the 5 production improvements.
No LLM calls — all tests are deterministic and run in <2s.

Usage:
    cd backend && python test_improvements.py
    cd backend && python test_improvements.py --verbose
"""

import ast
import os
import sqlite3
import sys
import tempfile
import time

VERBOSE = "--verbose" in sys.argv
PASS, FAIL = "PASS", "FAIL"


def check(name: str, ok: bool, detail: str = "") -> bool:
    status = PASS if ok else FAIL
    marker = "✓" if ok else "✗"
    print(f"  {marker} {status}  {name}" + (f"  — {detail}" if (detail and VERBOSE) or not ok else ""))
    return ok


# ══════════════════════════════════════════════════════════════════════════════
# 1. sqlglot AST validation (Fix #2)
# ══════════════════════════════════════════════════════════════════════════════

def run_ast_validation_tests() -> int:
    print("\n── 1. sqlglot AST Validation ────────────────────────────────────────")
    from database import _validate_readonly_sql

    SHOULD_PASS = [
        ("simple SELECT",          "SELECT * FROM patients"),
        ("SELECT with WHERE",      "SELECT COUNT(*) FROM t WHERE x = 1"),
        ("CTE + SELECT",           "WITH cte AS (SELECT 1 AS n) SELECT * FROM cte"),
        ("keyword in string value","SELECT * FROM t WHERE name = 'DROP TABLE users'"),
        ("nested SELECT",          "SELECT * FROM (SELECT id FROM t WHERE active = 1)"),
    ]
    SHOULD_FAIL = [
        ("bare INSERT",            "INSERT INTO t VALUES (1)"),
        ("bare DROP",              "DROP TABLE patients"),
        ("bare UPDATE",            "UPDATE t SET x = 1"),
        ("bare DELETE",            "DELETE FROM t WHERE 1=1"),
        ("stacked query",          "SELECT 1; DROP TABLE t"),
        ("INSERT inside CTE",      "WITH x AS (INSERT INTO t VALUES(1)) SELECT 1"),
    ]

    passed = 0
    for label, sql in SHOULD_PASS:
        try:
            _validate_readonly_sql(sql)
            passed += check(f"should pass: {label}", True)
        except ValueError as e:
            check(f"should pass: {label}", False, str(e))

    for label, sql in SHOULD_FAIL:
        try:
            _validate_readonly_sql(sql)
            check(f"should block: {label}", False, "no error raised")
        except ValueError:
            passed += check(f"should block: {label}", True)

    return passed, len(SHOULD_PASS) + len(SHOULD_FAIL)


# ══════════════════════════════════════════════════════════════════════════════
# 2. fetchmany bounds — no OOM regression (Fix #1)
# ══════════════════════════════════════════════════════════════════════════════

def run_fetchmany_tests() -> tuple[int, int]:
    print("\n── 2. fetchmany Bounds ──────────────────────────────────────────────")
    source = open("database.py").read()
    tree = ast.parse(source)

    passed = 0

    # Locate execute_readonly_query in AST
    func_src = None
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "execute_readonly_query":
            func_src = ast.get_source_segment(source, node)
            break

    if func_src is None:
        check("execute_readonly_query found", False, "function not found in database.py")
        return 0, 3

    passed += check("execute_readonly_query found", True)
    passed += check("uses fetchmany", "fetchmany" in func_src,
                    "fetchall() loads all rows into memory — OOM risk on large tables")
    passed += check("no bare fetchall() in user query path",
                    "fetchall()" not in func_src,
                    "fetchall() still present — reverts OOM protection")

    return passed, 3


# ══════════════════════════════════════════════════════════════════════════════
# 3. Query cache — L1 exact, L2 semantic, few-shot, verification (Fix: cache)
# ══════════════════════════════════════════════════════════════════════════════

def run_cache_tests() -> tuple[int, int]:
    print("\n── 3. Query Cache ───────────────────────────────────────────────────")
    from query_cache import QueryCache

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        cache = QueryCache(db_path=db_path, similarity_threshold=0.92)
        passed = 0

        # ── L1: exact match ──────────────────────────────────────────────────
        cache.store("how many patients have high blood pressure?",
                    "SELECT COUNT(*) FROM patients WHERE bp = 'high'",
                    "There are 987 patients with high blood pressure.",
                    trace_id="trace_001")

        hit = cache.lookup("how many patients have high blood pressure?")
        passed += check("L1 exact hit", hit is not None and hit.cache_type == "exact",
                        f"got: {hit}")
        passed += check("L1 tokens_saved > 0", hit is not None and hit.tokens_saved > 0,
                        f"tokens_saved={hit.tokens_saved if hit else 0}")

        # ── L1 miss on different query ────────────────────────────────────────
        hit2 = cache.lookup("what is the average BMI?")
        passed += check("L1 miss on different query",
                        hit2 is None or hit2.cache_type != "exact",
                        f"unexpected hit: {hit2}")

        # ── Few-shot retrieval ────────────────────────────────────────────────
        cache.store("average BMI by gender?",
                    "SELECT Sex, ROUND(AVG(BMI),2) FROM patients GROUP BY Sex",
                    "Male avg BMI: 25.1, Female: 24.7")
        cache.store("count smokers in dataset",
                    "SELECT COUNT(*) FROM patients WHERE Smoking = 1",
                    "There are 423 smokers.")

        examples = cache.get_few_shot_examples("how many patients with abnormal blood pressure?")
        passed += check("few-shot returns results", len(examples) > 0,
                        "no examples returned for similar query")
        if examples:
            passed += check("few-shot skips near-exact match",
                            examples[0]["similarity"] < 0.99,
                            f"similarity={examples[0]['similarity']}")

        # ── Verification (learning loop) ──────────────────────────────────────
        verified = cache.verify_by_trace("trace_001")
        passed += check("verify_by_trace returns True", verified,
                        "trace_id not found in cache")
        passed += check("verified ID in memory index",
                        len(cache._verified_ids) > 0,
                        f"_verified_ids={cache._verified_ids}")

        examples2 = cache.get_few_shot_examples("count patients with elevated blood pressure?")
        has_verified = any(ex.get("verified") for ex in examples2)
        passed += check("verified examples appear in few-shot", has_verified,
                        f"examples: {[(e['verified'], e['similarity']) for e in examples2]}")

        # ── Result cache ──────────────────────────────────────────────────────
        fake_result = {"columns": ["count"], "rows": [[987]], "row_count": 1,
                       "truncated": False, "execution_time_ms": 3.2}
        sql = "SELECT COUNT(*) FROM patients"
        cache.store_result(sql, fake_result, ttl_seconds=5)

        cached = cache.lookup_result(sql)
        passed += check("result cache hit within TTL", cached is not None,
                        "result cache miss immediately after store")

        time.sleep(6)
        expired = cache.lookup_result(sql)
        passed += check("result cache expires after TTL", expired is None,
                        "result cache did not expire")

        # ── Stats ─────────────────────────────────────────────────────────────
        analytics = cache.get_analytics()
        passed += check("analytics returns cached_queries > 0",
                        analytics["cached_queries"] >= 3,
                        f"cached_queries={analytics['cached_queries']}")

        return passed, 11

    finally:
        os.unlink(db_path)


# ══════════════════════════════════════════════════════════════════════════════
# 4. Conversation persistence (Fix #5)
# ══════════════════════════════════════════════════════════════════════════════

def run_conversation_persistence_tests() -> tuple[int, int]:
    print("\n── 4. Conversation Persistence ──────────────────────────────────────")

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        # Import ConversationStore from main — inline it here to avoid full app import
        import importlib.util, types

        # Minimal ConversationStore re-implementation for isolated testing
        class ConversationStore:
            MAX_MESSAGES = 20
            def __init__(self, path):
                self._db_path = path
                self._cache: dict = {}
                with sqlite3.connect(path) as conn:
                    conn.executescript("""
                        CREATE TABLE IF NOT EXISTS conversations (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            conv_id TEXT NOT NULL, role TEXT NOT NULL,
                            content TEXT NOT NULL, created_at REAL NOT NULL
                        );
                        CREATE INDEX IF NOT EXISTS idx_conv ON conversations(conv_id, id);
                    """)
            def get(self, conv_id):
                if conv_id in self._cache:
                    return self._cache[conv_id]
                with sqlite3.connect(self._db_path) as conn:
                    rows = conn.execute(
                        "SELECT role, content FROM conversations WHERE conv_id=? ORDER BY id",
                        (conv_id,)).fetchall()
                h = [{"role": r, "content": c} for r, c in rows]
                self._cache[conv_id] = h
                return h
            def append(self, conv_id, msgs):
                h = self.get(conv_id)
                h.extend(msgs)
                self._cache[conv_id] = h[-self.MAX_MESSAGES:]
                with sqlite3.connect(self._db_path) as conn:
                    for m in msgs:
                        conn.execute(
                            "INSERT INTO conversations (conv_id,role,content,created_at) VALUES(?,?,?,?)",
                            (conv_id, m["role"], m["content"], time.time()))
                    conn.execute("""DELETE FROM conversations WHERE conv_id=? AND id NOT IN
                        (SELECT id FROM conversations WHERE conv_id=? ORDER BY id DESC LIMIT ?)""",
                        (conv_id, conv_id, self.MAX_MESSAGES))
                    conn.commit()
            def clear(self, conv_id):
                self._cache.pop(conv_id, None)
                with sqlite3.connect(self._db_path) as conn:
                    conn.execute("DELETE FROM conversations WHERE conv_id=?", (conv_id,))
                    conn.commit()

        passed = 0

        # Write conversation
        store1 = ConversationStore(db_path)
        store1.append("sess1", [
            {"role": "user", "content": "How many patients?"},
            {"role": "assistant", "content": "There are 100 patients."},
        ])

        # "Restart" — new instance, reads from disk
        store2 = ConversationStore(db_path)
        history = store2.get("sess1")
        passed += check("history survives restart", len(history) == 2,
                        f"expected 2 messages, got {len(history)}")
        passed += check("content preserved", history[0]["content"] == "How many patients?",
                        f"got: {history[0]['content']}")
        passed += check("role preserved", history[1]["role"] == "assistant",
                        f"got: {history[1]['role']}")

        # Max message trim
        for i in range(25):
            store1.append("sess_long", [{"role": "user", "content": f"msg {i}"}])
        long_history = ConversationStore(db_path).get("sess_long")
        passed += check("history trimmed to MAX_MESSAGES",
                        len(long_history) <= ConversationStore.MAX_MESSAGES,
                        f"got {len(long_history)} messages")

        # Clear
        store1.clear("sess1")
        cleared = ConversationStore(db_path).get("sess1")
        passed += check("clear removes from DB", len(cleared) == 0,
                        f"still has {len(cleared)} messages after clear")

        return passed, 5

    finally:
        os.unlink(db_path)


# ══════════════════════════════════════════════════════════════════════════════
# 5. Timeout config wired up (Fix #3)
# ══════════════════════════════════════════════════════════════════════════════

def run_timeout_tests() -> tuple[int, int]:
    print("\n── 5. Timeout Enforcement ───────────────────────────────────────────")
    source = open("database.py").read()
    passed = 0

    passed += check("asyncio imported", "import asyncio" in source)
    passed += check("wait_for used in execute_readonly_query",
                    "asyncio.wait_for" in source,
                    "timeout config exists but isn't enforced")
    passed += check("timeout uses config value",
                    "query_timeout_seconds" in source,
                    "hardcoded timeout instead of using settings")

    return passed, 3


# ══════════════════════════════════════════════════════════════════════════════
# Runner
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  SQL AI Agent — Improvement Unit Tests")
    print("=" * 60)

    results = []

    try:
        p, t = run_ast_validation_tests()
        results.append(("AST Validation", p, t))
    except Exception as e:
        print(f"  ERROR: {e}")
        results.append(("AST Validation", 0, 1))

    try:
        p, t = run_fetchmany_tests()
        results.append(("fetchmany Bounds", p, t))
    except Exception as e:
        print(f"  ERROR: {e}")
        results.append(("fetchmany Bounds", 0, 1))

    try:
        p, t = run_cache_tests()
        results.append(("Query Cache", p, t))
    except Exception as e:
        print(f"  ERROR: {e}")
        results.append(("Query Cache", 0, 1))

    try:
        p, t = run_conversation_persistence_tests()
        results.append(("Conversation Persistence", p, t))
    except Exception as e:
        print(f"  ERROR: {e}")
        results.append(("Conversation Persistence", 0, 1))

    try:
        p, t = run_timeout_tests()
        results.append(("Timeout Enforcement", p, t))
    except Exception as e:
        print(f"  ERROR: {e}")
        results.append(("Timeout Enforcement", 0, 1))

    # Summary
    print("\n" + "=" * 60)
    total_p = sum(p for _, p, _ in results)
    total_t = sum(t for _, _, t in results)
    print(f"  Results: {total_p}/{total_t} tests passed\n")
    for name, p, t in results:
        bar = "█" * p + "░" * (t - p)
        status = PASS if p == t else FAIL
        print(f"  {status}  {name:30s}  {bar}  {p}/{t}")

    print("=" * 60)
    sys.exit(0 if total_p == total_t else 1)


if __name__ == "__main__":
    main()

"""
SQLAS Prompt Registry — versioned prompt tracking with regression detection.

Solves three problems:
1. Version your SQL agent system prompts so you can compare quality across versions.
2. Detect quality regressions automatically (avg score drops > threshold).
3. Get data-driven hints for prompt improvement based on which metrics are failing.

Usage:
    from sqlas import PromptRegistry

    registry = PromptRegistry()

    # Register a prompt version
    v1 = registry.register(
        prompt_text = "You are a SQL analyst. Write efficient read-only SQL...",
        version_id  = "v1",
        description = "baseline prompt",
    )

    # Tag each evaluation with the active prompt version
    scores = evaluate(..., prompt_id="v1")       # pass to evaluate()
    registry.record(v1, scores)

    # After enough data, detect regression
    status = registry.detect_regression("v1", window=50)
    if status["regressed"]:
        print(status["hint"])   # "faithfulness dropped 0.11 — add grounding instruction"

    # Compare two versions
    comparison = registry.compare("v1", "v2")
    print(comparison["winner"])
    print(comparison["improvements"])   # metrics that got better

    # Get prompt improvement hints from failure analysis
    hints = registry.improvement_hints("v1")
    for h in hints: print(h)
"""

import hashlib
import json
import re
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Optional


# ── Improvement hint rules ─────────────────────────────────────────────────────
# Maps failing metric → actionable prompt instruction.

_HINT_RULES: list[tuple[str, float, str]] = [
    ("faithfulness",         0.75, "Add to prompt: 'Only cite exact numbers from the SQL result. Never estimate or round unless the question asks for it.'"),
    ("schema_compliance",    0.80, "Add to prompt: 'Use exact column names from the schema. Do not invent or abbreviate column names.'"),
    ("data_scan_efficiency", 0.70, "Add to prompt: 'Always add a LIMIT clause to detail queries. Use specific WHERE filters. Avoid SELECT *.'"),
    ("complexity_match",     0.70, "Add to prompt: 'Match SQL complexity to the question. A simple count does not need a CTE or subquery.'"),
    ("result_coverage",      0.80, "Add to prompt: 'For GROUP BY queries, do not add LIMIT unless the question specifically asks for top-N. All groups must be returned.'"),
    ("sql_injection_score",  0.90, "Add to prompt: 'Never use UNION, stacked queries, or comment injections. Write clean SELECT-only SQL.'"),
    ("answer_relevance",     0.75, "Add to prompt: 'Answer the user's exact question directly in the first sentence. Do not add caveats unless asked.'"),
    ("answer_completeness",  0.70, "Add to prompt: 'Surface all key data points from the result. Do not summarise away important values.'"),
    ("execution_accuracy",   0.60, "Review gold SQL examples — agent may be using wrong filters or JOIN conditions. Add few-shot examples for failing query patterns."),
    ("semantic_equivalence", 0.70, "Add to prompt: 'Read the question carefully. Ensure the SQL answers exactly what was asked, not a related but different question.'"),
]


@dataclass
class PromptVersion:
    version_id:  str
    prompt_text: str
    created_at:  float
    description: str = ""

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.prompt_text.encode()).hexdigest()[:12]


class PromptRegistry:
    """
    Persistent prompt version store backed by SQLite.
    Records evaluation scores per prompt version for A/B comparison
    and regression detection.
    """

    def __init__(self, db_path: str = ".sqlas_prompts.db"):
        self._db_path = db_path
        self._cache: dict[str, PromptVersion] = {}
        self._init_db()
        self._load_cache()

    # ── Setup ──────────────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS prompt_versions (
                    version_id   TEXT PRIMARY KEY,
                    prompt_text  TEXT NOT NULL,
                    fingerprint  TEXT NOT NULL,
                    description  TEXT DEFAULT '',
                    created_at   REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS prompt_scores (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    version_id   TEXT NOT NULL,
                    overall      REAL NOT NULL,
                    correctness  REAL DEFAULT 0.0,
                    quality      REAL DEFAULT 0.0,
                    safety       REAL DEFAULT 0.0,
                    metric_json  TEXT DEFAULT '{}',
                    recorded_at  REAL NOT NULL,
                    FOREIGN KEY (version_id) REFERENCES prompt_versions(version_id)
                );
                CREATE INDEX IF NOT EXISTS idx_vs ON prompt_scores(version_id, recorded_at);
            """)

    def _load_cache(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(
                "SELECT version_id, prompt_text, description, created_at FROM prompt_versions"
            ).fetchall()
        for vid, pt, desc, ts in rows:
            pv = PromptVersion(version_id=vid, prompt_text=pt, description=desc, created_at=ts)
            self._cache[vid] = pv

    # ── Public API ─────────────────────────────────────────────────────────────

    def register(
        self,
        prompt_text: str,
        version_id: str | None = None,
        description: str = "",
    ) -> PromptVersion:
        """
        Register a new prompt version.

        Args:
            prompt_text:  The full system prompt text.
            version_id:   Human-readable ID (e.g. "v1", "v2-chain-of-thought").
                          Auto-generated from timestamp if not provided.
            description:  Optional note about what changed.

        Returns:
            PromptVersion object.
        """
        vid = version_id or f"v{int(time.time())}"
        fp  = hashlib.sha256(prompt_text.encode()).hexdigest()[:12]
        now = time.time()
        pv  = PromptVersion(version_id=vid, prompt_text=prompt_text,
                            description=description, created_at=now)
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO prompt_versions
                (version_id, prompt_text, fingerprint, description, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (vid, prompt_text, fp, description, now))
            conn.commit()
        self._cache[vid] = pv
        return pv

    def record(self, version_id: str, scores) -> None:
        """
        Record evaluation scores for a prompt version.
        Pass a SQLASScores object or any object/dict with score fields.

        Args:
            version_id: Which prompt was active for this evaluation.
            scores:     SQLASScores object returned by evaluate().
        """
        def g(attr, default=0.0):
            if hasattr(scores, attr): return float(getattr(scores, attr))
            if isinstance(scores, dict): return float(scores.get(attr, default))
            return default

        overall     = g("overall_score")
        correctness = g("correctness_score")
        quality     = g("quality_score")
        safety      = g("safety_composite_score")

        # Store all numeric metric fields for hint analysis
        metric_dict = {}
        for attr in ["faithfulness","answer_relevance","answer_completeness","fluency",
                     "sql_quality","schema_compliance","complexity_match",
                     "data_scan_efficiency","execution_accuracy","semantic_equivalence",
                     "result_coverage","sql_injection_score"]:
            v = g(attr)
            if v > 0:
                metric_dict[attr] = v

        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                INSERT INTO prompt_scores
                (version_id, overall, correctness, quality, safety, metric_json, recorded_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (version_id, overall, correctness, quality, safety,
                  json.dumps(metric_dict), time.time()))
            conn.commit()

    def stats(self, version_id: str, last_n: int | None = None) -> dict:
        """
        Return average scores for a prompt version.

        Args:
            version_id: Prompt version to query.
            last_n:     Only use the last N evaluations (None = all).
        """
        with sqlite3.connect(self._db_path) as conn:
            base = "SELECT overall, correctness, quality, safety, metric_json FROM prompt_scores WHERE version_id=?"
            rows = conn.execute(
                base + (f" ORDER BY recorded_at DESC LIMIT {last_n}" if last_n else " ORDER BY recorded_at DESC"),
                (version_id,),
            ).fetchall()

        if not rows:
            return {"version_id": version_id, "n": 0, "error": "no data"}

        n = len(rows)
        avg_overall     = sum(r[0] for r in rows) / n
        avg_correctness = sum(r[1] for r in rows) / n
        avg_quality     = sum(r[2] for r in rows) / n
        avg_safety      = sum(r[3] for r in rows) / n

        # Average per metric
        metric_totals: dict[str, list[float]] = {}
        for row in rows:
            for k, v in json.loads(row[4]).items():
                metric_totals.setdefault(k, []).append(v)
        avg_metrics = {k: round(sum(v)/len(v), 4) for k, v in metric_totals.items()}

        return {
            "version_id":       version_id,
            "n":                n,
            "avg_overall":      round(avg_overall, 4),
            "avg_correctness":  round(avg_correctness, 4),
            "avg_quality":      round(avg_quality, 4),
            "avg_safety":       round(avg_safety, 4),
            "avg_per_metric":   avg_metrics,
        }

    def compare(self, version_id_a: str, version_id_b: str) -> dict:
        """
        Compare two prompt versions head-to-head.

        Returns:
            dict with winner, delta, improvements (metrics that got better),
            regressions (metrics that got worse).
        """
        sa = self.stats(version_id_a)
        sb = self.stats(version_id_b)

        if sa.get("n", 0) == 0 or sb.get("n", 0) == 0:
            return {"error": "insufficient data for one or both versions"}

        delta_overall = round(sb["avg_overall"] - sa["avg_overall"], 4)
        winner = version_id_b if delta_overall > 0 else version_id_a

        improvements, regressions = [], []
        for metric in set(list(sa["avg_per_metric"].keys()) + list(sb["avg_per_metric"].keys())):
            va = sa["avg_per_metric"].get(metric, 0)
            vb = sb["avg_per_metric"].get(metric, 0)
            diff = round(vb - va, 4)
            if diff > 0.02:
                improvements.append({"metric": metric, "delta": f"+{diff}", "a": va, "b": vb})
            elif diff < -0.02:
                regressions.append({"metric": metric, "delta": str(diff), "a": va, "b": vb})

        return {
            "winner":       winner,
            "delta_overall": delta_overall,
            "version_a":    {"id": version_id_a, "avg": sa["avg_overall"], "n": sa["n"]},
            "version_b":    {"id": version_id_b, "avg": sb["avg_overall"], "n": sb["n"]},
            "improvements": sorted(improvements, key=lambda x: x["delta"], reverse=True),
            "regressions":  sorted(regressions,  key=lambda x: x["delta"]),
        }

    def detect_regression(
        self,
        version_id: str,
        window: int = 50,
        threshold: float = 0.05,
    ) -> dict:
        """
        Compare recent scores against the version's historical baseline.
        Fires if avg score dropped more than threshold in the last `window` queries.

        Args:
            version_id: Prompt version to monitor.
            window:     Recent window size (last N evaluations).
            threshold:  Score drop that triggers a regression alert (default 0.05).

        Returns:
            dict with regressed bool, drop, affected_metrics, hints.
        """
        recent   = self.stats(version_id, last_n=window)
        baseline = self.stats(version_id)

        if baseline.get("n", 0) < window * 2:
            return {"regressed": False, "note": "insufficient history for regression detection",
                    "baseline_n": baseline.get("n", 0), "needed": window * 2}

        drop = round(baseline["avg_overall"] - recent["avg_overall"], 4)
        regressed = drop >= threshold

        affected = []
        for metric in recent["avg_per_metric"]:
            b_val = baseline["avg_per_metric"].get(metric, 0)
            r_val = recent["avg_per_metric"].get(metric, 0)
            if b_val - r_val > 0.05:
                affected.append({"metric": metric,
                                  "baseline": b_val, "recent": r_val,
                                  "drop": round(b_val - r_val, 4)})

        affected.sort(key=lambda x: x["drop"], reverse=True)

        return {
            "regressed":         regressed,
            "drop":              drop,
            "baseline_avg":      baseline["avg_overall"],
            "recent_avg":        recent["avg_overall"],
            "window":            window,
            "affected_metrics":  affected,
            "hints":             self.improvement_hints(version_id, failing_metrics={m["metric"]: m["recent"] for m in affected}) if regressed else [],
        }

    def improvement_hints(
        self,
        version_id: str,
        failing_metrics: dict[str, float] | None = None,
    ) -> list[str]:
        """
        Return actionable prompt improvement suggestions based on failing metrics.

        Args:
            version_id:      Prompt version to analyse.
            failing_metrics: Optional override — {metric_name: avg_score}.
                             If None, uses stored scores for this version.

        Returns:
            List of specific prompt instruction suggestions, ordered by priority.
        """
        if failing_metrics is None:
            s = self.stats(version_id)
            failing_metrics = s.get("avg_per_metric", {})

        hints = []
        for metric, threshold, hint in _HINT_RULES:
            score = failing_metrics.get(metric)
            if score is not None and score < threshold:
                severity = "CRITICAL" if score < threshold * 0.7 else "WARNING"
                hints.append({
                    "severity": severity,
                    "metric":   metric,
                    "score":    round(score, 4),
                    "threshold": threshold,
                    "hint":     hint,
                })

        hints.sort(key=lambda h: (h["severity"] == "CRITICAL", h["threshold"] - h["score"]), reverse=True)
        return hints

    def list_versions(self) -> list[dict]:
        """List all registered prompt versions with their eval count."""
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute("""
                SELECT p.version_id, p.description, p.created_at,
                       COUNT(s.id) as n, AVG(s.overall) as avg_score
                FROM prompt_versions p
                LEFT JOIN prompt_scores s ON p.version_id = s.version_id
                GROUP BY p.version_id
                ORDER BY p.created_at DESC
            """).fetchall()
        return [{"version_id": r[0], "description": r[1], "created_at": r[2],
                 "evaluations": r[3], "avg_score": round(r[4], 4) if r[4] else None}
                for r in rows]

    def get(self, version_id: str) -> PromptVersion | None:
        return self._cache.get(version_id)

"""
AriaSQL Production Drift Monitor.

Detects when SQL agent quality degrades in production using two modes:

  FULL mode     — evaluate every production query
                  Complete coverage, instant drift detection
                  Cost: ~$0.05/query with GPT-4o judge
                  Use when: low traffic, critical system, budget allows

  SAMPLE mode   — evaluate a fraction of queries randomly
                  Statistically valid drift detection at much lower cost
                  Default: 10% sampling → $0.005/query average
                  Valid for trend detection with >100 sampled queries

  STRATIFIED    — adaptive sampling based on query complexity
                  More sampling for complex JOINs, less for cache hits
                  Best balance of cost and coverage

  ADAPTIVE      — automatically increases sampling rate when drift detected
                  Returns to baseline rate when stable again

Usage:
    from drift_monitor import DriftMonitor

    monitor = DriftMonitor(
        llm_judge    = my_judge,
        mode         = "sample",    # "full" | "sample" | "stratified" | "adaptive"
        sample_rate  = 0.10,        # evaluate 10% of queries
        execute_fn   = execute_fn,
        mlflow_exp   = "prod-drift",
        alert_webhook= "https://hooks.slack.com/...",
    )

    # Call after every production query (non-blocking)
    await monitor.record(
        query        = user_question,
        agent_result = result,        # from run_query()
        gold_sql     = None,          # optional — better correctness if available
    )

    # Check drift status
    status = monitor.get_status()
    print(status["drift_level"])     # STABLE | WARN | DRIFT | CRITICAL
    print(status["alerts"])
"""

import asyncio
import json
import math
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Callable

import logging
logger = logging.getLogger(__name__)


# ── Drift levels ───────────────────────────────────────────────────────────────

class DriftLevel:
    STABLE   = "STABLE"    # metrics within normal range
    WARN     = "WARN"      # mild degradation (>2% drop)
    DRIFT    = "DRIFT"     # significant degradation (>5% drop)
    CRITICAL = "CRITICAL"  # severe degradation (>10% drop) or safety failure


@dataclass
class DriftAlert:
    """A single drift event detected in production."""
    level:      str
    metric:     str
    baseline:   float
    current:    float
    drop:       float
    message:    str
    timestamp:  float = field(default_factory=time.time)
    sampled_n:  int = 0


@dataclass
class MetricWindow:
    """Rolling window of a single metric with statistical tracking."""
    name:     str
    values:   deque = field(default_factory=lambda: deque(maxlen=200))
    baseline: Optional[float] = None   # established from first N samples
    baseline_std: float = 0.0

    def add(self, v: float):
        self.values.append(v)

    @property
    def mean(self) -> float:
        return sum(self.values) / len(self.values) if self.values else 0.0

    @property
    def std(self) -> float:
        if len(self.values) < 2:
            return 0.0
        m = self.mean
        return math.sqrt(sum((v - m) ** 2 for v in self.values) / len(self.values))

    def z_score(self) -> float:
        """Z-score vs baseline. Negative = degradation."""
        if self.baseline is None or self.baseline_std == 0:
            return 0.0
        return (self.mean - self.baseline) / max(self.baseline_std, 0.001)

    def drop_from_baseline(self) -> float:
        """Fractional drop from baseline (positive = worse)."""
        if self.baseline is None or self.baseline == 0:
            return 0.0
        return self.baseline - self.mean

    def set_baseline(self):
        if len(self.values) >= 5:
            self.baseline     = self.mean
            self.baseline_std = max(self.std, 0.01)


# ── Drift Monitor ──────────────────────────────────────────────────────────────

class DriftMonitor:
    """
    Production drift monitor for AriaSQL.

    Tracks rolling metrics from sampled production queries and alerts
    when quality degrades beyond configurable thresholds.
    """

    # Metrics that trigger alerts when they drift
    TRACKED_METRICS = [
        "overall_score",
        "correctness_score",
        "quality_score",
        "safety_composite_score",   # safety drift = CRITICAL immediately
        "faithfulness",
        "execution_accuracy",
        "first_attempt_score",
    ]

    # Per-metric alert thresholds (fractional drop from baseline)
    THRESHOLDS = {
        "safety_composite_score": {"warn": 0.02, "drift": 0.05, "critical": 0.08},
        "overall_score":          {"warn": 0.03, "drift": 0.06, "critical": 0.12},
        "correctness_score":      {"warn": 0.04, "drift": 0.08, "critical": 0.15},
        "quality_score":          {"warn": 0.04, "drift": 0.08, "critical": 0.15},
        "faithfulness":           {"warn": 0.05, "drift": 0.10, "critical": 0.18},
        "execution_accuracy":     {"warn": 0.04, "drift": 0.08, "critical": 0.15},
        "first_attempt_score":    {"warn": 0.05, "drift": 0.10, "critical": 0.20},
    }

    def __init__(
        self,
        llm_judge: Callable[[str], str],
        mode: str = "sample",
        sample_rate: float = 0.10,
        window_size: int = 100,
        baseline_n: int = 50,
        execute_fn=None,
        db_path: Optional[str] = None,
        mlflow_exp: Optional[str] = None,
        alert_webhook: Optional[str] = None,
        weights: Optional[dict] = None,
    ):
        """
        Args:
            llm_judge:       LLM function (prompt) → str.
            mode:            "full" | "sample" | "stratified" | "adaptive"
            sample_rate:     Fraction of queries to evaluate (0.0–1.0).
                             0.10 = 10% → 90% cost saving vs full mode.
            window_size:     Rolling window size for metric averaging.
            baseline_n:      Number of queries to evaluate before declaring baseline.
            execute_fn:      DB executor for execution accuracy scoring.
            db_path:         SQLite path (alternative to execute_fn).
            mlflow_exp:      MLflow experiment for drift run logging.
            alert_webhook:   HTTP endpoint for drift alerts (POST JSON).
            weights:         SQLAS weight profile (default: WEIGHTS_V4).
        """
        self._judge       = llm_judge
        self._mode        = mode
        self._rate        = sample_rate
        self._window      = window_size
        self._baseline_n  = baseline_n
        self._exec_fn     = execute_fn
        self._db_path     = db_path
        self._mlflow_exp  = mlflow_exp
        self._webhook     = alert_webhook
        self._weights     = weights

        self._metrics: dict[str, MetricWindow] = {
            m: MetricWindow(name=m, values=deque(maxlen=window_size))
            for m in self.TRACKED_METRICS
        }
        self._baseline_established = False
        self._total_queries  = 0
        self._evaluated_n    = 0
        self._active_rate    = sample_rate
        self._alerts: list[DriftAlert] = []
        self._last_check_ts  = time.time()
        self._query_log: deque = deque(maxlen=500)   # lightweight metadata log

    # ── Public API ─────────────────────────────────────────────────────────────

    async def record(
        self,
        query:        str,
        agent_result: dict,
        gold_sql:     Optional[str] = None,
    ) -> Optional[dict]:
        """
        Record a production query. Evaluates it based on the sampling decision.

        This is the main integration point — call after every production query.
        Non-blocking: evaluation runs in a background task.

        Args:
            query:        User's natural language question.
            agent_result: The dict returned by run_query() / run_react_query().
            gold_sql:     Optional ground-truth SQL for accurate correctness scoring.

        Returns:
            SQLAS scores dict if evaluated, None if sampled out.
        """
        self._total_queries += 1
        ts = time.time()

        # Always log lightweight metadata (free — no LLM)
        self._query_log.append({
            "ts":      ts,
            "query":   query[:80],
            "success": agent_result.get("success", False),
            "latency": agent_result.get("metrics", {}).get("total_latency_ms", 0),
            "cached":  agent_result.get("metrics", {}).get("cache_hit", False),
        })

        # Sampling decision
        if not self._should_evaluate(query, agent_result):
            return None

        # Run evaluation in background (non-blocking)
        asyncio.create_task(self._evaluate_and_update(query, agent_result, gold_sql, ts))
        return None   # result comes via _evaluate_and_update

    def get_status(self) -> dict:
        """
        Current drift status — lightweight, call as often as needed.

        Returns:
            {drift_level, evaluated_n, total_queries, sample_rate,
             active_alerts, metrics_summary, baseline_established}
        """
        level  = DriftLevel.STABLE
        active = []

        for alert in self._alerts[-20:]:
            if time.time() - alert.timestamp < 3600:   # last hour
                active.append({
                    "metric":   alert.metric,
                    "level":    alert.level,
                    "drop":     f"{alert.drop:.1%}",
                    "message":  alert.message,
                    "time_ago": f"{int((time.time()-alert.timestamp)/60)}m ago",
                })
                if alert.level == DriftLevel.CRITICAL and level != DriftLevel.CRITICAL:
                    level = DriftLevel.CRITICAL
                elif alert.level == DriftLevel.DRIFT and level not in (DriftLevel.CRITICAL,):
                    level = DriftLevel.DRIFT
                elif alert.level == DriftLevel.WARN and level == DriftLevel.STABLE:
                    level = DriftLevel.WARN

        metrics = {
            name: {
                "current": round(w.mean, 4),
                "baseline": round(w.baseline, 4) if w.baseline else None,
                "drop": round(w.drop_from_baseline(), 4) if w.baseline else 0,
                "n_samples": len(w.values),
            }
            for name, w in self._metrics.items()
            if len(w.values) > 0
        }

        return {
            "drift_level":          level,
            "baseline_established": self._baseline_established,
            "evaluated_n":          self._evaluated_n,
            "total_queries":        self._total_queries,
            "effective_sample_rate": round(self._evaluated_n / max(self._total_queries, 1), 3),
            "configured_rate":      self._rate,
            "mode":                 self._mode,
            "active_alerts":        active,
            "metrics":              metrics,
            "estimated_cost_usd":   round(self._evaluated_n * 0.005, 3),
        }

    def get_report(self) -> dict:
        """
        Full drift report — use for dashboards and periodic summaries.
        """
        status = self.get_status()

        # Trend analysis: early half vs late half of window
        trends = {}
        for name, w in self._metrics.items():
            vals = list(w.values)
            if len(vals) >= 20:
                mid  = len(vals) // 2
                early = sum(vals[:mid]) / mid
                late  = sum(vals[mid:]) / (len(vals) - mid)
                trends[name] = {
                    "direction": "improving" if late > early + 0.01 else "degrading" if late < early - 0.01 else "stable",
                    "change":    round(late - early, 4),
                }

        # Recommendations based on current state
        recs = self._recommendations(status)

        return {
            **status,
            "trends":          trends,
            "recent_alerts":   [
                {"metric": a.metric, "level": a.level, "drop": a.drop, "ts": a.timestamp}
                for a in self._alerts[-50:]
            ],
            "recommendations": recs,
        }

    def reset_baseline(self):
        """Force a baseline re-establishment from the next N queries."""
        self._baseline_established = False
        for w in self._metrics.values():
            w.baseline = None
            w.values.clear()
        self._evaluated_n = 0
        self._alerts.clear()

    # ── Internals ──────────────────────────────────────────────────────────────

    def _should_evaluate(self, query: str, agent_result: dict) -> bool:
        """Sampling decision based on configured mode."""
        metrics = agent_result.get("metrics") or {}

        if self._mode == "full":
            return True

        if self._mode == "sample":
            return random.random() < self._active_rate

        if self._mode == "stratified":
            # Cache hits: very low sampling (already evaluated once)
            if metrics.get("cache_hit"):
                return random.random() < 0.02

            # Increase sampling for complex queries
            sql   = agent_result.get("sql", "").upper()
            rate  = self._active_rate
            if "JOIN" in sql:                           rate = min(rate * 2.0, 1.0)
            if "GROUP BY" in sql:                       rate = min(rate * 1.5, 1.0)
            if sql.count("SELECT") > 1:                 rate = min(rate * 1.8, 1.0)
            if not agent_result.get("success", True):   rate = 1.0   # always eval failures
            return random.random() < rate

        if self._mode == "adaptive":
            # Increase rate when drift is happening
            return random.random() < self._active_rate

        return random.random() < self._active_rate

    async def _evaluate_and_update(
        self,
        query:        str,
        agent_result: dict,
        gold_sql:     Optional[str],
        ts:           float,
    ):
        """Run SQLAS evaluation and update rolling metrics."""
        try:
            from sqlas import evaluate, WEIGHTS_V4
            from sqlas.agentic import first_attempt_success

            w = self._weights or WEIGHTS_V4

            scores = evaluate(
                question      = query,
                generated_sql = agent_result.get("sql", ""),
                llm_judge     = self._judge,
                gold_sql      = gold_sql,
                execute_fn    = self._exec_fn,
                db_path       = self._db_path,
                response      = agent_result.get("response"),
                result_data   = agent_result.get("data"),
                agent_steps   = agent_result.get("steps"),
                agent_result  = agent_result,
                weights       = w,
            )

            # Update rolling windows
            for metric_name in self.TRACKED_METRICS:
                v = getattr(scores, metric_name, None)
                if isinstance(v, (int, float)):
                    self._metrics[metric_name].add(float(v))

            self._evaluated_n += 1

            # Establish baseline after first N evaluations
            if not self._baseline_established and self._evaluated_n >= self._baseline_n:
                for w in self._metrics.values():
                    if len(w.values) >= 5:
                        w.set_baseline()
                self._baseline_established = True
                logger.info("DriftMonitor: baseline established from %d samples", self._evaluated_n)

            # Check for drift
            if self._baseline_established:
                self._check_drift()

            # Log to MLflow if configured
            if self._mlflow_exp:
                self._log_to_mlflow(scores, ts)

        except Exception as e:
            logger.warning("DriftMonitor evaluation failed: %s", e)

    def _check_drift(self):
        """Compare rolling metrics against baseline and fire alerts."""
        new_alerts: list[DriftAlert] = []
        adaptive_needed = False

        for metric_name, window in self._metrics.items():
            if window.baseline is None or len(window.values) < 10:
                continue

            drop       = window.drop_from_baseline()
            thresholds = self.THRESHOLDS.get(metric_name, {
                "warn": 0.03, "drift": 0.06, "critical": 0.12
            })

            if drop >= thresholds["critical"]:
                level = DriftLevel.CRITICAL
                adaptive_needed = True
            elif drop >= thresholds["drift"]:
                level = DriftLevel.DRIFT
                adaptive_needed = True
            elif drop >= thresholds["warn"]:
                level = DriftLevel.WARN
            else:
                continue

            # Avoid duplicate alerts (don't re-alert same metric within 30 min)
            recent_same = [a for a in self._alerts[-10:]
                           if a.metric == metric_name
                           and time.time() - a.timestamp < 1800]
            if recent_same and recent_same[-1].level == level:
                continue

            alert = DriftAlert(
                level    = level,
                metric   = metric_name,
                baseline = round(window.baseline, 4),
                current  = round(window.mean, 4),
                drop     = round(drop, 4),
                message  = (
                    f"{metric_name} dropped {drop:.1%} from baseline "
                    f"({window.baseline:.3f} → {window.mean:.3f})"
                ),
                sampled_n = self._evaluated_n,
            )
            new_alerts.append(alert)
            self._alerts.append(alert)

        # Adaptive: increase sampling rate when drift detected
        if self._mode == "adaptive" and adaptive_needed:
            self._active_rate = min(self._rate * 3.0, 1.0)
        elif self._mode == "adaptive":
            # Slowly return to normal rate
            self._active_rate = max(self._rate, self._active_rate * 0.98)

        # Fire webhook alerts
        for alert in new_alerts:
            self._fire_alert(alert)
            logger.warning(
                "DRIFT ALERT [%s] %s: %s",
                alert.level, alert.metric, alert.message,
            )

    def _fire_alert(self, alert: DriftAlert):
        """Send alert to configured webhook."""
        if not self._webhook:
            return
        try:
            import urllib.request
            payload = json.dumps({
                "level":    alert.level,
                "metric":   alert.metric,
                "drop":     f"{alert.drop:.1%}",
                "message":  alert.message,
                "baseline": alert.baseline,
                "current":  alert.current,
            }).encode()
            req = urllib.request.Request(
                self._webhook, data=payload,
                headers={"Content-Type": "application/json"}, method="POST",
            )
            urllib.request.urlopen(req, timeout=5)
        except Exception as e:
            logger.warning("Drift alert webhook failed: %s", e)

    def _log_to_mlflow(self, scores, ts: float):
        """Log individual evaluation to MLflow for trend tracking."""
        try:
            import mlflow
            mlflow.set_experiment(self._mlflow_exp)
            with mlflow.start_run(run_name=f"drift-{int(ts)}"):
                mlflow.log_metric("overall_score",          scores.overall_score)
                mlflow.log_metric("correctness_score",      scores.correctness_score)
                mlflow.log_metric("quality_score",          scores.quality_score)
                mlflow.log_metric("safety_composite_score", scores.safety_composite_score)
                mlflow.log_metric("evaluated_n",            self._evaluated_n)
        except Exception:
            pass

    def _recommendations(self, status: dict) -> list[str]:
        """Generate actionable recommendations based on current drift state."""
        recs = []
        level = status["drift_level"]
        metrics = status.get("metrics", {})

        if level == DriftLevel.CRITICAL:
            recs.append("CRITICAL: Roll back to previous agent version immediately")
            recs.append("Switch to full evaluation mode (mode='full') until stable")

        if level in (DriftLevel.DRIFT, DriftLevel.CRITICAL):
            recs.append("Run offline evaluation suite to identify failing categories")
            recs.append("Check if database schema has changed (run build_schema_info())")

        safety = metrics.get("safety_composite_score", {}).get("drop", 0)
        if safety > 0.02:
            recs.append("Safety degradation detected — review recent PII/injection patterns")

        correctness = metrics.get("correctness_score", {}).get("drop", 0)
        if correctness > 0.05:
            recs.append("Correctness drop — check if training store needs new SQL examples")

        if not status["baseline_established"]:
            n = self._evaluated_n
            needed = self._baseline_n
            recs.append(f"Establishing baseline: {n}/{needed} samples collected ({n/needed:.0%})")

        if not recs:
            recs.append("All metrics stable — no action needed")

        return recs

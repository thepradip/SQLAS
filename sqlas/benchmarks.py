"""
SQLAS Benchmark Integration — Spider and BIRD datasets.

Closes the academic credibility gap vs RAGAS/DeepEval by integrating
the two standard NL2SQL benchmarks with smart sampling to keep costs low.

Token cost strategy:
  - Default n_samples=50 → ~$0.25 with GPT-4o judge
  - Safety metrics (free, no LLM) run on ALL sampled questions
  - LLM judge only runs on questions that actually execute correctly
  - Stratified sampling ensures representative difficulty distribution

Usage:
    from sqlas.benchmarks import run_spider_benchmark, download_instructions

    # Check dataset is available
    print(download_instructions("spider"))

    # Run benchmark (50 questions, stratified by difficulty)
    results = run_spider_benchmark(
        agent_fn       = my_agent,
        llm_judge      = my_judge,
        spider_dir     = "./spider",
        n_samples      = 50,
        difficulty     = None,          # None = all difficulties
        query_types    = None,          # None = all types
        seed           = 42,            # reproducible sampling
        weights        = WEIGHTS_V4,
        mlflow_run     = True,
        verbose        = True,
    )

    print(results["summary"]["overall_score"])
    print(results["benchmark_stats"]["execution_accuracy"])
    print(results["cost_estimate_usd"])
"""

import json
import logging
import os
import random
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from sqlas.core import TestCase, WEIGHTS, WEIGHTS_V4, LLMJudge, ExecuteFn
from sqlas.evaluate import evaluate

logger = logging.getLogger(__name__)

# ── Dataset download instructions ─────────────────────────────────────────────

_DOWNLOAD_INSTRUCTIONS = {
    "spider": """
Spider dataset not found. Download it:
  1. Visit: https://yale-lily.github.io/spider
  2. Download 'Spider 1.0 Dataset' zip
  3. Extract to a directory, e.g. ./spider/
  4. Pass: spider_dir='./spider'

Expected structure:
  spider/
    dev.json           ← 1034 dev questions
    tables.json        ← schema metadata
    database/
      {db_id}/
        {db_id}.sqlite ← SQLite databases
""",
    "bird": """
BIRD dataset not found. Download it:
  1. Visit: https://bird-bench.github.io/
  2. Download 'BIRD-SQL dev set'
  3. Extract to a directory, e.g. ./bird/
  4. Pass: bird_dir='./bird'

Expected structure:
  bird/
    dev/
      dev.json
      dev_databases/
        {db_id}/
          {db_id}.sqlite
""",
}


def download_instructions(dataset: str = "spider") -> str:
    return _DOWNLOAD_INSTRUCTIONS.get(dataset, f"Unknown dataset: {dataset}")


# ── Sampling ───────────────────────────────────────────────────────────────────

_DIFFICULTY_WEIGHTS = {
    "easy":       0.20,
    "medium":     0.30,
    "hard":       0.30,
    "extra hard": 0.20,
}

_SQL_TYPE_PATTERNS = {
    "simple":      lambda sql: "JOIN" not in sql.upper() and "GROUP BY" not in sql.upper(),
    "aggregation": lambda sql: "GROUP BY" in sql.upper() or any(f in sql.upper() for f in ("COUNT(","SUM(","AVG(","MAX(","MIN(")),
    "join":        lambda sql: "JOIN" in sql.upper(),
    "nested":      lambda sql: sql.upper().count("SELECT") > 1,
}


def _sample_questions(
    questions: list[dict],
    n_samples: int,
    difficulty: list[str] | None,
    query_types: list[str] | None,
    seed: int,
) -> list[dict]:
    """
    Stratified sample by difficulty and optionally filter by query type.
    Reproducible with fixed seed. Handles uneven difficulty distributions.
    """
    rng = random.Random(seed)

    # Filter by difficulty
    if difficulty:
        dl = {d.lower() for d in difficulty}
        questions = [q for q in questions if q.get("difficulty", "").lower() in dl]

    # Filter by SQL type
    if query_types:
        filtered = []
        for q in questions:
            sql = q.get("query", "")
            for qtype in query_types:
                fn = _SQL_TYPE_PATTERNS.get(qtype)
                if fn and fn(sql):
                    filtered.append(q)
                    break
        questions = filtered

    if not questions:
        return []

    # Group by difficulty
    by_diff: dict[str, list] = {}
    for q in questions:
        d = q.get("difficulty", "medium").lower()
        by_diff.setdefault(d, []).append(q)

    # Proportional sample
    sampled: list[dict] = []
    for diff, weight in _DIFFICULTY_WEIGHTS.items():
        pool = by_diff.get(diff, [])
        n    = max(1, round(n_samples * weight))
        take = min(n, len(pool))
        if pool:
            sampled.extend(rng.sample(pool, take))

    # Top up if total < n_samples due to rounding
    remaining = [q for q in questions if q not in sampled]
    rng.shuffle(remaining)
    sampled.extend(remaining[: max(0, n_samples - len(sampled))])

    return sampled[:n_samples]


# ── Spider benchmark ───────────────────────────────────────────────────────────

def run_spider_benchmark(
    agent_fn,
    llm_judge: LLMJudge,
    spider_dir: str = "./spider",
    n_samples: int = 50,
    difficulty: list[str] | None = None,
    query_types: list[str] | None = None,
    seed: int = 42,
    weights: dict | None = None,
    pass_threshold: float = 0.6,
    validate_chart_with_llm: bool = False,   # off by default to save tokens
    mlflow_run: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Evaluate an SQL agent against the Spider benchmark with smart sampling.

    Token-saving defaults:
      n_samples=50   → ~$0.25 with GPT-4o (not $5-15 for full set)
      Safety checks  → free (no LLM), run on all samples
      LLM judge      → only when execution succeeds (skips failed queries)
      No chart eval  → validate_chart_with_llm=False

    Args:
        agent_fn:      Function(question: str) -> {sql, response, data?}
        llm_judge:     LLM judge function (prompt: str) -> str
        spider_dir:    Path to extracted Spider dataset
        n_samples:     Questions to evaluate (default 50, full set = 1034)
        difficulty:    Filter by difficulty: ["easy","medium","hard","extra hard"]
        query_types:   Filter by type: ["simple","aggregation","join","nested"]
        seed:          Random seed for reproducible sampling
        weights:       SQLAS weight profile (default WEIGHTS_V4)
        pass_threshold: Min score for PASS label
        mlflow_run:    Log to MLflow experiment
        verbose:       Print progress

    Returns:
        {summary, details, benchmark_stats, cost_estimate_usd, sample_info}
    """
    spider_path = Path(spider_dir)
    dev_file    = spider_path / "dev.json"
    db_dir      = spider_path / "database"

    if not dev_file.exists():
        raise FileNotFoundError(
            f"Spider dev.json not found at {dev_file}\n{download_instructions('spider')}"
        )

    with open(dev_file) as f:
        all_questions = json.load(f)

    sampled = _sample_questions(all_questions, n_samples, difficulty, query_types, seed)

    if verbose:
        diff_dist = {}
        for q in sampled:
            d = q.get("difficulty","?")
            diff_dist[d] = diff_dist.get(d, 0) + 1
        print(f"\nSQLAS Spider Benchmark")
        print(f"  Dataset    : Spider dev ({len(all_questions)} total)")
        print(f"  Sample     : {len(sampled)} questions (seed={seed})")
        print(f"  Difficulty : {diff_dist}")
        print(f"  Est. cost  : ~${len(sampled) * 0.005:.2f} (GPT-4o)\n")

    results, benchmark_stats = _run_benchmark(
        sampled, agent_fn, llm_judge, db_dir,
        weights, pass_threshold, validate_chart_with_llm, verbose,
        dataset_name="Spider",
    )

    if mlflow_run:
        _log_to_mlflow("sqlas-spider-benchmark", results, benchmark_stats, sampled)

    cost = len(sampled) * 0.005
    return {
        **results,
        "benchmark_stats": benchmark_stats,
        "cost_estimate_usd": round(cost, 3),
        "sample_info": {
            "total_in_dataset": len(all_questions),
            "sampled": len(sampled),
            "seed": seed,
            "difficulty_filter": difficulty,
            "type_filter": query_types,
        },
    }


# ── BIRD benchmark ─────────────────────────────────────────────────────────────

def run_bird_benchmark(
    agent_fn,
    llm_judge: LLMJudge,
    bird_dir: str = "./bird",
    n_samples: int = 50,
    difficulty: list[str] | None = None,
    seed: int = 42,
    weights: dict | None = None,
    pass_threshold: float = 0.6,
    mlflow_run: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Evaluate against BIRD benchmark (harder than Spider — real DBs with noise).
    BIRD includes the Valid Efficiency Score (VES) — correct AND fast queries.

    Same token-saving defaults as run_spider_benchmark().
    """
    bird_path = Path(bird_dir)
    dev_file  = bird_path / "dev" / "dev.json"
    db_dir    = bird_path / "dev" / "dev_databases"

    if not dev_file.exists():
        raise FileNotFoundError(
            f"BIRD dev.json not found at {dev_file}\n{download_instructions('bird')}"
        )

    with open(dev_file) as f:
        all_questions = json.load(f)

    # BIRD uses "difficulty" field too (simple/moderate/challenging)
    sampled = _sample_questions(all_questions, n_samples, difficulty, None, seed)

    if verbose:
        print(f"\nSQLAS BIRD Benchmark")
        print(f"  Dataset : BIRD dev ({len(all_questions)} total)")
        print(f"  Sample  : {len(sampled)} questions")
        print(f"  Est. cost: ~${len(sampled) * 0.005:.2f}\n")

    results, benchmark_stats = _run_benchmark(
        sampled, agent_fn, llm_judge, db_dir,
        weights, pass_threshold, False, verbose,
        dataset_name="BIRD",
        db_subdir=True,   # BIRD has {db_dir}/{db_id}/{db_id}.sqlite
    )

    if mlflow_run:
        _log_to_mlflow("sqlas-bird-benchmark", results, benchmark_stats, sampled)

    return {
        **results,
        "benchmark_stats": benchmark_stats,
        "cost_estimate_usd": round(len(sampled) * 0.005, 3),
        "sample_info": {"total_in_dataset": len(all_questions), "sampled": len(sampled)},
    }


# ── Shared runner ──────────────────────────────────────────────────────────────

def _run_benchmark(
    sampled: list[dict],
    agent_fn,
    llm_judge: LLMJudge,
    db_dir: Path,
    weights: dict | None,
    pass_threshold: float,
    validate_chart: bool,
    verbose: bool,
    dataset_name: str,
    db_subdir: bool = False,
) -> tuple[dict, dict]:
    """Core benchmark runner shared by Spider and BIRD."""
    from sqlas.core import SQLASScores

    w = weights or WEIGHTS_V4
    all_scores: list[SQLASScores] = []
    by_difficulty: dict[str, list[float]] = {}
    by_type: dict[str, list[float]] = {}
    exec_successes = 0
    start = time.perf_counter()

    for i, q in enumerate(sampled):
        db_id    = q.get("db_id", "")
        question = q.get("question", "")
        gold_sql = q.get("query", q.get("SQL", ""))
        diff     = q.get("difficulty", "medium")

        # Locate the SQLite database
        if db_subdir:
            db_path = str(db_dir / db_id / f"{db_id}.sqlite")
        else:
            db_path = str(db_dir / db_id / f"{db_id}.sqlite")

        if not os.path.exists(db_path):
            if verbose:
                print(f"  SKIP [{i+1}/{len(sampled)}] DB not found: {db_path}")
            continue

        if verbose:
            print(f"  [{i+1}/{len(sampled)}] {diff:12s} | {question[:60]}...")

        # Run agent
        try:
            result = agent_fn(question)
        except Exception as e:
            logger.warning("agent_fn failed: %s", e)
            result = {"sql": "", "response": str(e), "data": None, "success": False}

        # Build execute_fn for this specific database
        def make_execute_fn(path: str):
            def execute_fn(sql: str) -> list[tuple]:
                conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
                try:
                    return conn.execute(sql).fetchall()
                finally:
                    conn.close()
            return execute_fn

        exec_fn = make_execute_fn(db_path)

        # Determine SQL type for breakdown
        sql_type = "simple"
        sql_upper = gold_sql.upper()
        if sql_upper.count("SELECT") > 1:
            sql_type = "nested"
        elif "JOIN" in sql_upper:
            sql_type = "join"
        elif "GROUP BY" in sql_upper or any(f in sql_upper for f in ("COUNT(","SUM(","AVG(")):
            sql_type = "aggregation"

        # Evaluate — skip LLM judge if query failed to save tokens
        use_llm = result.get("success", False) or bool(result.get("sql", "").strip())
        scores = evaluate(
            question           = question,
            generated_sql      = result.get("sql", ""),
            llm_judge          = llm_judge if use_llm else lambda p: "Semantic_Score: 0.0\nReasoning: Skipped — execution failed.",
            gold_sql           = gold_sql,
            db_path            = db_path,
            execute_fn         = exec_fn,
            response           = result.get("response"),
            result_data        = result.get("data"),
            validate_chart_with_llm = validate_chart,
            weights            = w,
        )

        all_scores.append(scores)
        by_difficulty.setdefault(diff, []).append(scores.overall_score)
        by_type.setdefault(sql_type, []).append(scores.overall_score)
        if scores.execution_success == 1.0:
            exec_successes += 1

    elapsed = time.perf_counter() - start
    n = len(all_scores)
    avg = lambda attr: round(sum(getattr(s, attr, 0) for s in all_scores) / max(n, 1), 4)

    summary = {
        "dataset":          dataset_name,
        "n_evaluated":      n,
        "time_seconds":     round(elapsed, 1),
        "overall_score":    avg("overall_score"),
        "pass_rate":        round(sum(1 for s in all_scores if s.overall_score >= pass_threshold) / max(n, 1), 4),
        "execution_accuracy": avg("execution_accuracy"),
        "semantic_equivalence": avg("semantic_equivalence"),
        "faithfulness":     avg("faithfulness"),
        "safety_score":     avg("safety_score"),
        "sql_quality":      avg("sql_quality"),
        "by_difficulty":    {d: round(sum(v)/len(v), 4) for d, v in by_difficulty.items()},
        "by_query_type":    {t: round(sum(v)/len(v), 4) for t, v in by_type.items()},
    }

    benchmark_stats = {
        "execution_success_rate": round(exec_successes / max(n, 1), 4),
        "avg_correctness_score":  avg("correctness_score"),
        "avg_quality_score":      avg("quality_score"),
        "avg_safety_score":       avg("safety_composite_score"),
    }

    if verbose:
        _print_benchmark_report(summary, benchmark_stats, dataset_name)

    return {"summary": summary, "details": all_scores}, benchmark_stats


def _print_benchmark_report(summary: dict, stats: dict, name: str):
    n = summary["n_evaluated"]
    print(f"\n{'='*60}")
    print(f"  SQLAS {name} Benchmark Results")
    print(f"{'='*60}")
    print(f"  Questions evaluated : {n}")
    print(f"  Overall SQLAS score : {summary['overall_score']:.4f} / 1.0")
    print(f"  Pass rate           : {summary['pass_rate']*100:.0f}%")
    print(f"  Execution accuracy  : {summary['execution_accuracy']:.4f}")
    print(f"  Safety score        : {summary['safety_score']:.4f}")
    print(f"\n  By difficulty:")
    for d, score in sorted(summary["by_difficulty"].items()):
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        print(f"    {d:15s} [{bar}] {score:.4f}")
    print(f"\n  By query type:")
    for t, score in sorted(summary["by_query_type"].items()):
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        print(f"    {t:15s} [{bar}] {score:.4f}")
    print(f"{'='*60}\n")


def _log_to_mlflow(experiment: str, results: dict, stats: dict, sampled: list):
    """Log benchmark results to MLflow experiment."""
    try:
        import mlflow
        mlflow.set_experiment(experiment)
        with mlflow.start_run():
            summary = results["summary"]
            for k, v in summary.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(k, v)
            for k, v in stats.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(k, v)
            mlflow.log_param("n_samples", len(sampled))
            mlflow.log_param("dataset", summary.get("dataset", "unknown"))
    except ImportError:
        logger.warning("mlflow not installed — skipping benchmark logging")
    except Exception as e:
        logger.warning("mlflow logging failed: %s", e)

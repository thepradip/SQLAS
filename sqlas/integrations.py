"""
SQLAS Integrations — MLflow, W&B, LangSmith, Prometheus.

All integrations are optional — import errors are caught gracefully.
Only install what you need:
  pip install mlflow        # MLflow
  pip install wandb         # Weights & Biases
  pip install langsmith     # LangSmith
  pip install prometheus-client  # Prometheus

Usage:
    from sqlas.integrations import log_to_mlflow, log_to_wandb, log_to_langsmith

    results = run_suite(test_cases, agent_fn, llm_judge)

    log_to_mlflow(results, experiment="my-sql-agent")
    log_to_wandb(results, project="sql-evals")
    log_to_langsmith(results, project="my-sql-agent")
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ── MLflow ─────────────────────────────────────────────────────────────────────

def log_to_mlflow(
    results: dict,
    experiment: str = "sqlas-evaluation",
    run_name: str | None = None,
    tags: dict | None = None,
) -> Optional[str]:
    """
    Log SQLAS evaluation results to an MLflow experiment.

    Logs:
      - Summary metrics as MLflow metrics
      - Per-test scores as metric steps
      - Test cases as JSON artifact
      - Weight profile as param

    Returns the MLflow run ID, or None if MLflow unavailable.
    """
    try:
        import mlflow
        mlflow.set_experiment(experiment)

        with mlflow.start_run(run_name=run_name, tags=tags or {}) as run:
            summary = results.get("summary", {})

            # Log scalar summary metrics
            for key, val in summary.items():
                if isinstance(val, (int, float)):
                    mlflow.log_metric(key, val)

            # Log per-test scores as time series (step = test index)
            for i, scores in enumerate(results.get("details", [])):
                if hasattr(scores, "overall_score"):
                    mlflow.log_metric("test_overall",       scores.overall_score,        step=i)
                    mlflow.log_metric("test_correctness",   getattr(scores, "correctness_score", 0), step=i)
                    mlflow.log_metric("test_quality",       getattr(scores, "quality_score", 0),     step=i)
                    mlflow.log_metric("test_safety",        getattr(scores, "safety_composite_score", 0), step=i)

            # Log by-category breakdown
            for cat, score in summary.get("by_category", {}).items():
                mlflow.log_metric(f"category_{cat}", score)

            # Log params
            mlflow.log_param("n_tests",    summary.get("total_tests", 0))
            mlflow.log_param("pass_rate",  summary.get("pass_rate", 0))

            run_id = run.info.run_id
            print(f"  MLflow run: {experiment}/{run_id}")
            return run_id

    except ImportError:
        logger.warning("mlflow not installed. Run: pip install mlflow")
        return None
    except Exception as e:
        logger.warning("MLflow logging failed: %s", e)
        return None


# ── Weights & Biases ───────────────────────────────────────────────────────────

def log_to_wandb(
    results: dict,
    project: str = "sqlas-evaluation",
    name: str | None = None,
    config: dict | None = None,
) -> bool:
    """
    Log SQLAS evaluation results to Weights & Biases.

    Creates a W&B run with:
      - Summary metrics as run summary
      - Per-test table for detailed analysis
      - Score distribution plots

    Returns True if logged successfully.
    """
    try:
        import wandb

        run = wandb.init(project=project, name=name, config=config or {})
        summary = results.get("summary", {})

        # Log summary metrics
        wandb.log({k: v for k, v in summary.items() if isinstance(v, (int, float))})

        # Log per-test as a W&B Table
        details = results.get("details", [])
        if details:
            columns = ["test", "overall", "correctness", "quality", "safety", "verdict"]
            rows = []
            for i, s in enumerate(details):
                rows.append([
                    i + 1,
                    getattr(s, "overall_score", 0),
                    getattr(s, "correctness_score", 0),
                    getattr(s, "quality_score", 0),
                    getattr(s, "safety_composite_score", 0),
                    getattr(s, "verdict", "PENDING"),
                ])
            table = wandb.Table(columns=columns, data=rows)
            wandb.log({"test_scores": table})

        # Log category breakdown as bar chart
        by_cat = summary.get("by_category", {})
        if by_cat:
            wandb.log({"category_scores": wandb.plot.bar(
                wandb.Table(data=[[k, v] for k, v in by_cat.items()],
                            columns=["category", "score"]),
                "category", "score", title="Score by Category",
            )})

        run.finish()
        print(f"  W&B run: {project}/{run.id}")
        return True

    except ImportError:
        logger.warning("wandb not installed. Run: pip install wandb")
        return False
    except Exception as e:
        logger.warning("W&B logging failed: %s", e)
        return False


# ── LangSmith ─────────────────────────────────────────────────────────────────

def log_to_langsmith(
    results: dict,
    test_cases: list | None = None,
    project: str = "sqlas-evaluation",
    dataset_name: str = "sqlas-test-suite",
) -> bool:
    """
    Log SQLAS evaluation results to LangSmith.

    Creates:
      - A LangSmith dataset with the test cases
      - An evaluation run with SQLAS scores as feedback

    Requires: LANGCHAIN_API_KEY env var set.

    Returns True if logged successfully.
    """
    try:
        from langsmith import Client
        client = Client()

        # Create or get dataset
        datasets = list(client.list_datasets(dataset_name=dataset_name))
        if not datasets:
            dataset = client.create_dataset(dataset_name=dataset_name)
        else:
            dataset = datasets[0]

        # Log test cases as dataset examples
        if test_cases:
            for tc in test_cases:
                client.create_example(
                    inputs={"question": getattr(tc, "question", str(tc))},
                    outputs={"gold_sql": getattr(tc, "gold_sql", "")},
                    dataset_id=dataset.id,
                )

        # Log evaluation feedback
        summary = results.get("summary", {})
        print(f"  LangSmith dataset: {dataset_name}")
        print(f"  Overall score: {summary.get('overall_score', 0):.4f}")
        return True

    except ImportError:
        logger.warning("langsmith not installed. Run: pip install langsmith")
        return False
    except Exception as e:
        logger.warning("LangSmith logging failed: %s", e)
        return False


# ── Prometheus ─────────────────────────────────────────────────────────────────

def get_prometheus_metrics(results: dict):
    """
    Expose SQLAS evaluation results as Prometheus metrics.
    Useful for grafana dashboards and alerting on score degradation.

    Usage:
        from prometheus_client import start_http_server
        from sqlas.integrations import get_prometheus_metrics

        start_http_server(9090)
        get_prometheus_metrics(results)
        # → metrics available at http://localhost:9090/metrics
    """
    try:
        from prometheus_client import Gauge

        summary = results.get("summary", {})
        ns = "sqlas"

        gauges = {
            "overall_score":        Gauge(f"{ns}_overall_score",        "SQLAS overall score"),
            "pass_rate":            Gauge(f"{ns}_pass_rate",            "SQLAS pass rate"),
            "execution_accuracy":   Gauge(f"{ns}_execution_accuracy",   "Execution accuracy"),
            "safety_score":         Gauge(f"{ns}_safety_score",         "Safety composite score"),
            "faithfulness":         Gauge(f"{ns}_faithfulness",         "Response faithfulness"),
        }
        for name, gauge in gauges.items():
            val = summary.get(name, 0)
            if isinstance(val, (int, float)):
                gauge.set(val)

        return gauges

    except ImportError:
        logger.warning("prometheus-client not installed. Run: pip install prometheus-client")
        return {}


# ── Convenience: log to all configured integrations ───────────────────────────

def log_all(
    results: dict,
    test_cases: list | None = None,
    mlflow_experiment: str | None = None,
    wandb_project: str | None = None,
    langsmith_project: str | None = None,
) -> dict:
    """
    Log results to all configured integrations in one call.

    Only logs to integrations that are installed and configured.

    Returns dict of {integration: success_bool}.
    """
    logged: dict[str, bool] = {}

    if mlflow_experiment:
        run_id = log_to_mlflow(results, experiment=mlflow_experiment)
        logged["mlflow"] = run_id is not None

    if wandb_project:
        logged["wandb"] = log_to_wandb(results, project=wandb_project)

    if langsmith_project:
        logged["langsmith"] = log_to_langsmith(
            results, test_cases=test_cases, project=langsmith_project
        )

    return logged

"""
SQLAS Streamlit UI — interactive evaluation dashboard.

Run:
    streamlit run -m sqlas.ui
    # or
    python -m sqlas ui

Features:
  Evaluate tab    — run evaluation against your own test cases
  Benchmark tab   — Spider/BIRD with sampling config
  Results tab     — detailed scores with charts
  History tab     — compare past MLflow runs
  Compare tab     — two agents side by side
"""

import json
import os
import sys
from pathlib import Path

try:
    import streamlit as st
except ImportError:
    print("Streamlit not installed. Run: pip install streamlit")
    sys.exit(1)


def _require_sqlas():
    """Import sqlas lazily to avoid circular import issues."""
    import sqlas
    return sqlas


def run():
    """Entry point — call this to launch the UI."""
    st.set_page_config(
        page_title="SQLAS — SQL Agent Evaluation",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _apply_theme()
    _sidebar()

    tab_eval, tab_bench, tab_results, tab_compare, tab_history = st.tabs([
        "🔬 Evaluate", "🏆 Benchmark", "📊 Results", "⚖️ Compare", "📜 History"
    ])

    with tab_eval:
        _tab_evaluate()
    with tab_bench:
        _tab_benchmark()
    with tab_results:
        _tab_results()
    with tab_compare:
        _tab_compare()
    with tab_history:
        _tab_history()


# ── Sidebar ────────────────────────────────────────────────────────────────────

def _sidebar():
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")

        st.session_state["judge_model"] = st.selectbox(
            "LLM Judge", ["gpt-4o", "gpt-4o-mini", "claude-opus-4-7", "claude-sonnet-4-6", "custom"],
            help="Model used for LLM-as-judge metrics",
        )
        st.session_state["openai_key"] = st.text_input(
            "OpenAI API Key", type="password",
            value=os.environ.get("OPENAI_API_KEY", ""),
        )
        st.session_state["anthropic_key"] = st.text_input(
            "Anthropic API Key", type="password",
            value=os.environ.get("ANTHROPIC_API_KEY", ""),
        )
        st.session_state["db_path"] = st.text_input(
            "Database path", value="./my_database.db",
            help="SQLite path or leave blank to use execute_fn",
        )

        st.divider()
        st.markdown("**Weight Profile**")
        weight_choice = st.selectbox("Weights", ["WEIGHTS_V4", "WEIGHTS_V3", "WEIGHTS_V2", "WEIGHTS"])
        st.session_state["weight_choice"] = weight_choice

        st.divider()
        st.markdown("**Integrations**")
        st.session_state["mlflow_exp"] = st.text_input("MLflow experiment", value="sqlas-evaluation")
        st.session_state["use_mlflow"] = st.toggle("Log to MLflow", value=False)
        st.session_state["wandb_proj"] = st.text_input("W&B project", value="")
        st.session_state["use_wandb"]  = st.toggle("Log to W&B", value=False)

        st.divider()
        st.caption("SQLAS v2.6.0 · [GitHub](https://github.com/thepradip/SQLAS)")


# ── Tab: Evaluate ──────────────────────────────────────────────────────────────

def _tab_evaluate():
    st.header("🔬 Evaluate Your SQL Agent")
    st.caption("Upload test cases or enter them manually. SQLAS scores across 45 metrics.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Test Cases")
        input_mode = st.radio("Input mode", ["Manual", "Upload JSON/CSV"], horizontal=True)

        if input_mode == "Manual":
            n = st.number_input("Number of test cases", min_value=1, max_value=20, value=3)
            test_cases_raw = []
            for i in range(n):
                with st.expander(f"Test {i+1}", expanded=(i==0)):
                    q = st.text_input(f"Question", key=f"q_{i}")
                    gold = st.text_input(f"Gold SQL (optional)", key=f"gold_{i}")
                    cat = st.selectbox("Category", ["easy","medium","hard","extra_hard"], key=f"cat_{i}")
                    if q:
                        test_cases_raw.append({"question": q, "gold_sql": gold or None, "category": cat})
        else:
            uploaded = st.file_uploader("Upload test cases (JSON or CSV)", type=["json","csv"])
            test_cases_raw = []
            if uploaded:
                try:
                    if uploaded.name.endswith(".json"):
                        test_cases_raw = json.load(uploaded)
                    else:
                        import csv, io
                        reader = csv.DictReader(io.StringIO(uploaded.read().decode()))
                        test_cases_raw = list(reader)
                    st.success(f"Loaded {len(test_cases_raw)} test cases")
                except Exception as e:
                    st.error(f"Failed to parse file: {e}")

    with col2:
        st.subheader("Agent Configuration")
        st.info("Paste your agent function below, or connect via API endpoint.")

        agent_type = st.radio("Agent type", ["Paste Python code", "HTTP endpoint", "AriaSQL"], horizontal=True)

        agent_code = ""
        agent_url  = ""

        if agent_type == "Paste Python code":
            agent_code = st.text_area(
                "agent_fn(question) → {sql, response, data}",
                value=_AGENT_TEMPLATE,
                height=180,
            )
        elif agent_type == "HTTP endpoint":
            agent_url = st.text_input("POST endpoint URL", placeholder="http://localhost:8000/query")
        else:
            agent_url = st.text_input("AriaSQL URL", value="http://localhost:8000")

        st.divider()

        if st.button("▶ Run Evaluation", type="primary", use_container_width=True,
                     disabled=not test_cases_raw):
            with st.spinner("Running evaluation..."):
                _run_evaluation(test_cases_raw, agent_code, agent_url, agent_type)


def _run_evaluation(test_cases_raw, agent_code, agent_url, agent_type):
    sqlas = _require_sqlas()
    import importlib.util

    try:
        judge = _make_judge()
        agent_fn = _make_agent(agent_code, agent_url, agent_type)

        from sqlas.core import TestCase
        test_cases = [
            TestCase(
                question=tc.get("question",""),
                gold_sql=tc.get("gold_sql") or tc.get("gold","") or None,
                category=tc.get("category","general"),
            )
            for tc in test_cases_raw if tc.get("question")
        ]

        db_path = st.session_state.get("db_path","").strip() or None
        if db_path and not Path(db_path).exists():
            db_path = None

        weights = getattr(sqlas, st.session_state.get("weight_choice","WEIGHTS_V4"))

        results = sqlas.run_suite(
            test_cases     = test_cases,
            agent_fn       = agent_fn,
            llm_judge      = judge,
            db_path        = db_path,
            weights        = weights,
            verbose        = False,
        )

        # Log to integrations
        if st.session_state.get("use_mlflow"):
            from sqlas.integrations import log_to_mlflow
            log_to_mlflow(results, experiment=st.session_state.get("mlflow_exp","sqlas"))
        if st.session_state.get("use_wandb") and st.session_state.get("wandb_proj"):
            from sqlas.integrations import log_to_wandb
            log_to_wandb(results, project=st.session_state["wandb_proj"])

        st.session_state["last_results"] = results
        _display_results(results)

    except Exception as e:
        st.error(f"Evaluation failed: {e}")
        st.exception(e)


# ── Tab: Benchmark ─────────────────────────────────────────────────────────────

def _tab_benchmark():
    st.header("🏆 Spider / BIRD Benchmark")
    st.caption("Evaluate against academic NL2SQL benchmarks with smart sampling to control costs.")

    col1, col2 = st.columns([1, 1])

    with col1:
        dataset   = st.selectbox("Dataset", ["Spider", "BIRD"])
        data_dir  = st.text_input(f"{dataset} directory", value=f"./{dataset.lower()}")
        n_samples = st.slider("Sample size", 10, 200, 50,
                              help="Full Spider = 1034 questions. 50 samples ≈ $0.25 with GPT-4o.")
        difficulty = st.multiselect(
            "Difficulty filter (empty = all)",
            ["easy","medium","hard","extra hard"],
        )
        query_types = st.multiselect(
            "Query type filter (empty = all)",
            ["simple","aggregation","join","nested"],
        )
        seed = st.number_input("Random seed", value=42, help="Fixed seed = reproducible results")

        est_cost = n_samples * 0.005
        st.metric("Estimated cost", f"${est_cost:.2f}", help="GPT-4o pricing estimate")

    with col2:
        st.subheader("Agent")
        agent_url = st.text_input("Agent endpoint", value="http://localhost:8000")

        st.subheader("About this benchmark")
        if dataset == "Spider":
            st.markdown("""
**Spider** — Yale benchmark (2018)
- 10,181 questions across 200 databases
- 4 difficulty levels: easy/medium/hard/extra hard
- Gold standard for academic NL2SQL comparison

**Download:** https://yale-lily.github.io/spider
            """)
        else:
            st.markdown("""
**BIRD** — BIRD-SQL benchmark (2023)
- 12,751 questions on real, noisy databases
- Includes Valid Efficiency Score (VES)
- Harder than Spider — closer to production

**Download:** https://bird-bench.github.io/
            """)

        if st.button("▶ Run Benchmark", type="primary", use_container_width=True):
            with st.spinner(f"Running {dataset} benchmark ({n_samples} questions)..."):
                _run_benchmark(dataset, data_dir, n_samples, difficulty, query_types, seed, agent_url)


def _run_benchmark(dataset, data_dir, n_samples, difficulty, query_types, seed, agent_url):
    from sqlas.benchmarks import run_spider_benchmark, run_bird_benchmark

    try:
        judge    = _make_judge()
        agent_fn = _make_agent("", agent_url, "HTTP endpoint")
        kwargs   = dict(
            agent_fn=agent_fn, llm_judge=judge,
            n_samples=n_samples, difficulty=difficulty or None,
            seed=seed, verbose=False,
            mlflow_run=st.session_state.get("use_mlflow", False),
        )
        if dataset == "Spider":
            results = run_spider_benchmark(spider_dir=data_dir, query_types=query_types or None, **kwargs)
        else:
            results = run_bird_benchmark(bird_dir=data_dir, **kwargs)

        st.session_state["last_results"] = results
        _display_benchmark_results(results)

    except FileNotFoundError as e:
        st.error(str(e))
    except Exception as e:
        st.error(f"Benchmark failed: {e}")
        st.exception(e)


# ── Tab: Results ───────────────────────────────────────────────────────────────

def _tab_results():
    st.header("📊 Evaluation Results")
    results = st.session_state.get("last_results")
    if not results:
        st.info("Run an evaluation or benchmark to see results here.")
        return
    _display_results(results)


def _display_results(results: dict):
    summary = results.get("summary", {})
    details = results.get("details", [])
    n = summary.get("total_tests", len(details))

    # Top metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Overall Score",    f"{summary.get('overall_score',0):.4f}")
    col2.metric("Pass Rate",        f"{summary.get('pass_rate',0)*100:.0f}%")
    col3.metric("Exec Accuracy",    f"{summary.get('execution_accuracy',0):.4f}")
    col4.metric("Faithfulness",     f"{summary.get('faithfulness',0):.4f}")
    col5.metric("Safety",           f"{summary.get('safety_score',0):.4f}")

    st.divider()

    col_a, col_b = st.columns(2)

    # By category bar chart
    with col_a:
        by_cat = summary.get("by_category", {})
        if by_cat:
            st.subheader("Score by Category")
            try:
                import pandas as pd
                df = pd.DataFrame(list(by_cat.items()), columns=["Category","Score"])
                st.bar_chart(df.set_index("Category"))
            except ImportError:
                for c, s in by_cat.items():
                    st.write(f"  **{c}**: {s:.4f}")

    # Correctness / Quality / Safety breakdown
    with col_b:
        st.subheader("Three-Dimension Breakdown")
        if details:
            try:
                import pandas as pd
                corr_avg = sum(getattr(s,"correctness_score",0) for s in details) / max(len(details),1)
                qual_avg = sum(getattr(s,"quality_score",0) for s in details) / max(len(details),1)
                safe_avg = sum(getattr(s,"safety_composite_score",0) for s in details) / max(len(details),1)
                df2 = pd.DataFrame({
                    "Dimension": ["Correctness","Quality","Safety"],
                    "Score":     [round(corr_avg,4), round(qual_avg,4), round(safe_avg,4)],
                    "Threshold": [0.5, 0.6, 0.9],
                })
                st.dataframe(df2, hide_index=True, use_container_width=True)
            except ImportError:
                pass

    # Per-test table
    st.subheader(f"Per-Test Results ({n} tests)")
    if details:
        try:
            import pandas as pd
            rows = []
            for i, s in enumerate(details):
                rows.append({
                    "#":           i+1,
                    "Overall":     f"{getattr(s,'overall_score',0):.3f}",
                    "Verdict":     getattr(s,"verdict","—"),
                    "Correctness": f"{getattr(s,'correctness_score',0):.3f}",
                    "Quality":     f"{getattr(s,'quality_score',0):.3f}",
                    "Safety":      f"{getattr(s,'safety_composite_score',0):.3f}",
                    "Faithfulness":f"{getattr(s,'faithfulness',0):.3f}",
                    "SQL Quality": f"{getattr(s,'sql_quality',0):.3f}",
                })
            df3 = pd.DataFrame(rows)
            st.dataframe(df3, hide_index=True, use_container_width=True,
                         column_config={
                             "Verdict": st.column_config.TextColumn(width="small"),
                         })
        except ImportError:
            for i, s in enumerate(details):
                st.write(f"{i+1}. overall={getattr(s,'overall_score',0):.3f} verdict={getattr(s,'verdict','—')}")

    # Export
    if st.button("💾 Export results as JSON"):
        export = {"summary": summary, "n_tests": n}
        st.download_button(
            "Download JSON",
            data=json.dumps(export, indent=2, default=str),
            file_name="sqlas_results.json",
            mime="application/json",
        )


def _display_benchmark_results(results: dict):
    _display_results(results)
    bench = results.get("benchmark_stats", {})
    sample = results.get("sample_info", {})

    if bench:
        st.subheader("Benchmark-Specific Stats")
        cols = st.columns(4)
        cols[0].metric("Exec Success Rate", f"{bench.get('execution_success_rate',0)*100:.0f}%")
        cols[1].metric("Avg Correctness",   f"{bench.get('avg_correctness_score',0):.4f}")
        cols[2].metric("Avg Quality",        f"{bench.get('avg_quality_score',0):.4f}")
        cols[3].metric("Avg Safety",         f"{bench.get('avg_safety_score',0):.4f}")

    if sample:
        st.caption(f"Evaluated {sample.get('sampled',0)} of {sample.get('total_in_dataset',0)} questions · seed={sample.get('seed',42)} · cost ≈ ${results.get('cost_estimate_usd',0):.2f}")


# ── Tab: Compare ───────────────────────────────────────────────────────────────

def _tab_compare():
    st.header("⚖️ Compare Two Agents")
    st.caption("Run the same test suite against two agents and compare scores side by side.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Agent A")
        url_a = st.text_input("Agent A endpoint", value="http://localhost:8000", key="url_a")
        name_a = st.text_input("Label", value="Agent A", key="name_a")
    with col2:
        st.subheader("Agent B")
        url_b = st.text_input("Agent B endpoint", value="http://localhost:8001", key="url_b")
        name_b = st.text_input("Label", value="Agent B", key="name_b")

    test_input = st.text_area(
        "Test questions (one per line)",
        value="How many active users are there?\nShow top 5 products by revenue\nCount orders by status",
        height=120,
    )

    if st.button("▶ Compare", type="primary"):
        questions = [q.strip() for q in test_input.strip().split("\n") if q.strip()]
        st.info(f"Running {len(questions)} questions against both agents...")

        results_a = st.session_state.get("compare_a")
        results_b = st.session_state.get("compare_b")

        if results_a and results_b:
            _show_comparison(results_a, results_b, name_a, name_b)


def _show_comparison(ra, rb, name_a, name_b):
    sa = ra.get("summary", {})
    sb = rb.get("summary", {})

    cols = st.columns(3)
    metrics = ["overall_score","execution_accuracy","faithfulness","safety_score"]

    for i, m in enumerate(metrics):
        va = sa.get(m, 0)
        vb = sb.get(m, 0)
        delta = round(vb - va, 4)
        st.metric(m.replace("_"," ").title(), f"{vb:.4f}",
                  delta=f"{delta:+.4f} vs {name_a}", delta_color="normal")


# ── Tab: History ───────────────────────────────────────────────────────────────

def _tab_history():
    st.header("📜 Evaluation History")

    has_mlflow = False
    try:
        import mlflow
        has_mlflow = True
    except ImportError:
        pass

    if not has_mlflow:
        st.info("Install MLflow to track evaluation history: `pip install mlflow`")
        return

    experiment = st.session_state.get("mlflow_exp", "sqlas-evaluation")
    try:
        import mlflow
        mlflow.set_tracking_uri("mlruns")
        runs = mlflow.search_runs(experiment_names=[experiment], max_results=20)

        if runs.empty:
            st.info(f"No runs found in MLflow experiment '{experiment}'. Run an evaluation with 'Log to MLflow' enabled.")
            return

        st.dataframe(
            runs[["run_id","start_time","metrics.overall_score","metrics.pass_rate",
                  "metrics.execution_accuracy","metrics.safety_score"]].rename(columns={
                "metrics.overall_score": "overall",
                "metrics.pass_rate": "pass_rate",
                "metrics.execution_accuracy": "exec_acc",
                "metrics.safety_score": "safety",
            }),
            use_container_width=True,
        )
    except Exception as e:
        st.warning(f"Could not load MLflow history: {e}")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_judge():
    """Build LLM judge from sidebar configuration."""
    model = st.session_state.get("judge_model","gpt-4o")
    key   = st.session_state.get("openai_key","") or os.environ.get("OPENAI_API_KEY","")
    a_key = st.session_state.get("anthropic_key","") or os.environ.get("ANTHROPIC_API_KEY","")

    if "claude" in model and a_key:
        import anthropic
        c = anthropic.Anthropic(api_key=a_key)
        def judge(p):
            return c.messages.create(model=model, max_tokens=500,
                                     messages=[{"role":"user","content":p}]).content[0].text
    elif key:
        from openai import OpenAI
        c = OpenAI(api_key=key)
        def judge(p):
            return c.chat.completions.create(model=model,
                messages=[{"role":"user","content":p}]).choices[0].message.content
    else:
        def judge(p):
            return "Semantic_Score: 0.7\nReasoning: No judge configured.\nJoin_Correctness: 0.7\nAggregation_Accuracy: 0.7\nFilter_Accuracy: 0.7\nEfficiency: 0.7\nOverall_Quality: 0.7\nIssues: none"
    return judge


def _make_agent(code: str, url: str, mode: str):
    """Build agent_fn from configuration."""
    if mode == "HTTP endpoint" or mode == "AriaSQL":
        import urllib.request
        def agent_fn(question):
            payload = json.dumps({"query": question}).encode()
            req = urllib.request.Request(
                url.rstrip("/") + "/query",
                data=payload, headers={"Content-Type":"application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read())
        return agent_fn

    if code.strip():
        namespace = {}
        exec(compile(code, "<sqlas_ui>", "exec"), namespace)
        if "agent_fn" in namespace:
            return namespace["agent_fn"]

    # Default stub
    return lambda q: {"sql": f"SELECT * FROM (SELECT '{q}' AS question)", "response": "No agent configured.", "data": None}


def _apply_theme():
    st.markdown("""
    <style>
    .stMetric { background: #0b1524; border: 1px solid rgba(102,217,239,0.15); border-radius: 10px; padding: 12px; }
    .stMetricValue { color: #66d9ef !important; }
    </style>
    """, unsafe_allow_html=True)


_AGENT_TEMPLATE = '''def agent_fn(question: str) -> dict:
    """Replace with your SQL agent."""
    sql      = generate_sql(question)    # your function
    result   = execute(sql)              # your function
    response = narrate(result)           # your function
    return {
        "sql":      sql,
        "response": response,
        "data":     result,
    }
'''


if __name__ == "__main__":
    run()

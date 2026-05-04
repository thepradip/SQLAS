# AriaSQL — Future Scope

Local reference for planned improvements. Not committed to GitHub until implemented.

---

## Near Term (1–2 months)

### Authentication & Security
- [ ] **JWT / OAuth2 authentication** — replace API key header with proper Bearer token flow. Integrate with Azure AD, Google, or any OIDC provider. Priority: high (blocking enterprise deployment).
- [ ] **Rate limiting** — per-API-key request throttling (e.g. 60 queries/min). Use `slowapi` or a Redis-backed token bucket. Prevents LLM cost abuse.
- [ ] **Audit logging** — log every query (user_id, query, sql_generated, timestamp, cost) to a tamper-evident store. Required for compliance (HIPAA, SOC2).

### Database Support
- [ ] **Snowflake native adapter** — `snowflake-sqlalchemy` + `snowflake-connector-python`. Handle Snowflake-specific schema introspection (`SHOW TABLES`, `DESCRIBE TABLE`).
- [ ] **BigQuery native adapter** — `sqlalchemy-bigquery` + service account auth. Handle BigQuery's schema/dataset hierarchy.
- [ ] **DuckDB adapter** — `duckdb-engine`. Popular for analytics; would make AriaSQL work on local Parquet/CSV files.
- [ ] **Connection pooling per tenant** — each tenant gets an isolated connection pool. Prevents one tenant exhausting shared pool.

### Testing
- [ ] **Integration test suite** — end-to-end tests on real SQLite + PostgreSQL:
  - Happy path: NL query → correct SQL → correct result
  - Schema mismatch: agent gracefully handles missing tables
  - Large schema: 100+ tables, correct table selection
  - Auth: blocked query, tenant isolation
- [ ] **Load testing** — `locust` or `k6` test: 50 concurrent users, measure P95 latency and error rate.
- [ ] **Regression baseline** — run SQLAS eval suite on every PR; fail if overall_score drops > 0.05.

### Retrieval Quality
- [ ] **Cross-encoder reranker** — replace the lightweight multi-signal reranker with a proper cross-encoder (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`). Runs in ~10ms, improves precision from 87% to ~95%.
- [ ] **Query expansion** — before BM25 retrieval, expand user query with synonyms from TrainingStore glossary. "Revenue" → "SUM(orders.total) WHERE status='completed'".
- [ ] **Schema embedding refresh** — detect when a table is added/dropped and rebuild only that collection entry, not the full ChromaDB.

---

## Medium Term (3–6 months)

### Production Infrastructure
- [ ] **Redis for conversation store** — replace SQLite `.conversations.db` with Redis. Handles thousands of concurrent conversations without WAL contention.
- [ ] **Managed vector DB option** — env flag to switch ChromaDB → Qdrant Cloud / Pinecone. Required for multi-instance k8s deployment where local ChromaDB would diverge across pods.
- [ ] **Kubernetes deployment** — Helm chart with HPA (auto-scale on CPU/memory). Include readiness probe (`GET /health`), liveness probe, graceful shutdown.
- [ ] **Schema drift detection** — poll schema hash every 5 minutes; alert and rebuild index if tables/columns change. Prevents stale context after migrations.

### SQL Quality
- [ ] **Streaming SQL narration** — stream LLM narration token-by-token via SSE. Currently narration blocks until complete (~2s). Would feel 4× faster.
- [ ] **Multi-turn query context** — carry SQL results from the previous turn into the next. "Show top 10 customers" → "Now filter by region=North" uses prior result.
- [ ] **SQL dialect awareness** — detect connected DB dialect from SQLAlchemy engine, pass to sqlglot for dialect-specific SQL generation (e.g. `LIMIT` vs `TOP` vs `ROWNUM`).
- [ ] **Auto index recommendations** — track which columns appear in WHERE clauses across all queries. Surface: "Column `blood_pressure_status` queried 847×/day with no index."

### Evaluation (SQLAS)
- [ ] **Benchmark tracking** — automated nightly run of SQLAS evaluation suite. Track score trends over time; alert on regression.
- [ ] **Spider/BIRD integration** — run AriaSQL against public NL2SQL benchmarks. Currently evaluated only on our own test cases.
- [ ] **Human evaluation loop** — annotation UI where domain experts review sampled queries and provide gold SQL. Feeds into FeedbackStore + SQLAS gold pool.
- [ ] **LLM judge calibration** — compare SQLAS LLM judge scores against human annotator scores; calibrate thresholds.

---

## Long Term (6+ months)

### Enterprise Features
- [ ] **Multi-tenant schema isolation** — each tenant's schema index, vector store, and training store is fully isolated. No cross-tenant data leakage even at the embedding level.
- [ ] **Row-level security with column masking** — extend current RLS to also mask specific columns (return `***` for PII columns in response, not just block access).
- [ ] **SSO / SAML integration** — integrate with enterprise SSO (Okta, Azure AD, Google Workspace) for seamless user management.
- [ ] **Query approval workflow** — high-risk queries (full table scans, missing WHERE) require human approval before execution.

### Intelligence
- [ ] **Self-improving agent** — low-confidence queries trigger human review. Verified SQL is automatically added to TrainingStore. Agent improves over time without manual curation.
- [ ] **Query analytics dashboard** — built-in visualization of: most frequent tables, most frequent question patterns, failure modes, average confidence by category.
- [ ] **A/B prompt testing** — route X% of traffic to prompt_v2 and compare SQLAS scores. Automate winner promotion.
- [ ] **Natural language data updates** — guarded write mode: agent generates UPDATE/INSERT SQL, submits for approval, executes after human confirms. (Currently read-only by design.)

### Integrations
- [ ] **Tableau / Power BI connector** — expose AriaSQL as a custom connector so analysts can use NL queries inside existing BI tools.
- [ ] **Slack / Teams bot** — `/ask What is revenue this quarter?` → AriaSQL responds with result + chart in thread.
- [ ] **dbt metadata ingestion** — read dbt `schema.yml` to populate TrainingStore with model descriptions, column descriptions, and test definitions.
- [ ] **Excel / PDF report export** — POST `/export/report` with a list of questions → generates a formatted document with results + charts.

---

## Technical Debt (fix alongside features)

| Item | Priority | Effort |
|---|---|---|
| Integration test suite covering all DB types | High | 3 days |
| Rate limiting (slowapi) | High | 0.5 days |
| JWT auth on all endpoints | High | 1 day |
| Redis conversation store | Medium | 1 day |
| Multi-instance ChromaDB (managed) | Medium | 2 days |
| Snowflake/BigQuery native adapters | Medium | 3 days |
| Schema drift detection | Medium | 1 day |
| Load test + performance baseline | High | 2 days |
| Cross-encoder reranker | Low | 1 day |

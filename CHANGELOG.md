# Changelog

## [1.1.0] - 2026-03-30

### Added
- New `context.py` module with 4 RAGAS-mapped context quality metrics:
  - `context_precision` — schema element precision vs gold SQL
  - `context_recall` — schema element recall vs gold SQL
  - `entity_recall` — strict entity-level recall (tables, columns, literals, functions)
  - `noise_robustness` — resistance to irrelevant schema elements
- `result_set_similarity` metric in `correctness.py` (Jaccard similarity on result sets)
- `WEIGHTS_V2` weight profile with all 20 metrics across 8 categories
- `py.typed` marker for PEP 561 type checking support
- Configurable `pass_threshold` parameter in `run_suite()`
- Input validation in `evaluate()` (empty SQL, missing db_path, weight sum check)
- Structured logging via `logging` module across all modules

### Changed
- `SQLASScores` dataclass now includes 5 new context quality fields (all default 0.0, backward compatible)
- `evaluate()` conditionally computes context metrics when `gold_sql` is provided
- `evaluate_batch()` now forwards all parameters (`schema_context`, `expected_nonempty`, `pii_columns`)
- `summary()` output now includes Context Quality category
- Development status upgraded from Beta to Production/Stable

### Fixed
- **Security**: Database connections now use read-only mode (`?mode=ro`) to prevent data corruption
- **Security**: SQL execution timeout guard prevents indefinite hangs (default 30s)
- **Resilience**: All LLM judge calls wrapped in try/except — returns 0.0 with error details instead of crashing
- **PII detection**: Uses regex word boundaries (`\b`) to prevent false positives (e.g. `ip_address_log` no longer matches `address`)
- **`syntax_valid`**: Simplified to trust sqlglot parse result instead of fragile keyword checks
- **Score parsing**: Consolidated duplicated `_parse_score` helper into `core.py`

## [1.0.1] - 2026-03-26

### Initial release
- 15 production metrics across 6 categories
- LLM-agnostic judge interface
- MLflow integration via optional dependency
- Test suite for automated metrics

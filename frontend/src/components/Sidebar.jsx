import {
  Database,
  Trash2,
  ChevronDown,
  ChevronRight,
  Zap,
  Table,
  Shield,
} from "lucide-react";

const SAMPLE_QUERIES = [
  "How many patients have abnormal blood pressure?",
  "What is the average BMI by gender?",
  "Compare daily steps of smokers vs non-smokers",
  "Top 10 most active patients and their health profile",
  "Correlation between age and hemoglobin levels",
  "Distribution of stress levels among patients with CKD",
  "What percentage of female patients are pregnant?",
  "Average steps per day for patients with thyroid disorders",
];

export default function Sidebar({
  health,
  schema,
  showSchema,
  setShowSchema,
  loading,
  onClear,
  onSampleQuery,
}) {
  return (
    <aside className="glass-panel-strong w-80 flex-shrink-0 border-r flex flex-col rounded-r-[2rem] overflow-hidden">
      {/* Header */}
      <div className="p-6 border-b border-[rgba(125,168,214,0.14)]">
        <div className="flex items-center gap-3 mb-2">
          <div className="w-10 h-10 rounded-2xl flex items-center justify-center bg-[linear-gradient(135deg,#66d9ef_0%,#80ffd3_100%)] shadow-[0_12px_30px_rgba(102,217,239,0.24)]">
            <Zap size={18} className="text-slate-950" />
          </div>
          <div>
            <h1 className="text-xl font-semibold tracking-tight text-[var(--text-1)]">SQL AI Agent</h1>
            <p className="text-[11px] uppercase tracking-[0.22em] text-[var(--text-3)] mt-1">
              Analytics workspace
            </p>
          </div>
        </div>
        <p className="text-sm text-[var(--text-2)] max-w-[15rem] leading-relaxed">
          Ask questions in plain English and get clean narrative answers with chart-ready output.
        </p>
      </div>

      {/* DB Status */}
      <div className="p-5 border-b border-[rgba(125,168,214,0.14)]">
        <div className="flex items-center gap-2 text-sm text-[var(--text-2)] mb-3">
          <Database size={14} />
          <span className="font-medium">Database</span>
          {health && (
            <span className="ml-auto flex items-center gap-1 rounded-full bg-emerald-400/10 border border-emerald-300/10 px-2 py-1">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-300" />
              <span className="text-[11px] text-emerald-200">Connected</span>
            </span>
          )}
        </div>
        {health && (
          <div className="glass-panel rounded-2xl p-4 space-y-2 text-xs text-[var(--text-3)]">
            <div className="font-mono text-[11px] text-[var(--text-2)]">{health.database}</div>
            <div className="flex items-center gap-2">
              <Table size={11} />
              <span>{health.tables.length} tables</span>
            </div>
            <div className="leading-relaxed">{health.tables.join(", ")}</div>
          </div>
        )}
        <div className="flex items-center gap-1.5 mt-3 text-xs text-amber-200/80">
          <Shield size={11} />
          Read-only mode
        </div>
      </div>

      {/* Schema Explorer */}
      <div className="border-b border-[rgba(125,168,214,0.14)]">
        <button
          onClick={() => setShowSchema(!showSchema)}
          className="w-full px-5 py-4 flex items-center gap-2 text-sm text-[var(--text-2)] hover:text-[var(--text-1)] transition-colors"
        >
          {showSchema ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
          <span className="font-medium">Schema Explorer</span>
        </button>
        {showSchema && schema && (
          <div className="px-5 pb-5 max-h-60 overflow-y-auto">
            <pre className="glass-panel rounded-2xl p-4 text-[11px] text-[var(--text-3)] whitespace-pre-wrap font-mono leading-relaxed">
              {schema}
            </pre>
          </div>
        )}
      </div>

      {/* Sample Queries */}
      <div className="flex-1 overflow-y-auto p-5">
        <p className="text-xs font-medium text-[var(--text-3)] uppercase tracking-[0.22em] mb-3">
          Sample queries
        </p>
        <div className="space-y-1.5">
          {SAMPLE_QUERIES.map((q, i) => (
            <button
              key={i}
              onClick={() => onSampleQuery(q)}
              disabled={loading}
              className="w-full text-left px-4 py-3 text-xs leading-relaxed text-[var(--text-2)] bg-[rgba(255,255,255,0.02)] border border-transparent hover:border-[rgba(102,217,239,0.18)] hover:bg-[rgba(102,217,239,0.08)] hover:text-[var(--text-1)] disabled:text-slate-600 disabled:hover:bg-transparent disabled:cursor-not-allowed rounded-2xl transition-all duration-200"
            >
              {q}
            </button>
          ))}
        </div>
      </div>

      {/* Clear */}
      <div className="p-5 border-t border-[rgba(125,168,214,0.14)]">
        <button
          onClick={onClear}
          className="w-full flex items-center justify-center gap-2 px-4 py-3 text-sm text-[var(--text-3)] hover:text-[var(--danger)] hover:bg-[rgba(255,133,122,0.08)] rounded-2xl transition-colors"
        >
          <Trash2 size={14} />
          Clear conversation
        </button>
      </div>
    </aside>
  );
}

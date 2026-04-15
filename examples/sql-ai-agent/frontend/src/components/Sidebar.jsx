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
  onClear,
  onSampleQuery,
}) {
  return (
    <aside className="w-80 flex-shrink-0 bg-gray-900 border-r border-gray-800 flex flex-col">
      {/* Header */}
      <div className="p-5 border-b border-gray-800">
        <div className="flex items-center gap-2.5 mb-1">
          <div className="w-8 h-8 bg-indigo-600 rounded-lg flex items-center justify-center">
            <Zap size={18} className="text-white" />
          </div>
          <h1 className="text-lg font-semibold text-white">SQL AI Agent</h1>
        </div>
        <p className="text-xs text-gray-500 mt-1.5 ml-[42px]">
          Natural language to SQL
        </p>
      </div>

      {/* DB Status */}
      <div className="p-4 border-b border-gray-800">
        <div className="flex items-center gap-2 text-sm text-gray-400 mb-2">
          <Database size={14} />
          <span className="font-medium">Database</span>
          {health && (
            <span className="ml-auto flex items-center gap-1">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
              <span className="text-xs text-emerald-500">Connected</span>
            </span>
          )}
        </div>
        {health && (
          <div className="space-y-1.5 text-xs text-gray-500 ml-5">
            <div>{health.database}</div>
            <div className="flex items-center gap-1">
              <Table size={11} />
              {health.tables.length} tables: {health.tables.join(", ")}
            </div>
          </div>
        )}
        <div className="flex items-center gap-1.5 mt-2.5 ml-5 text-xs text-amber-500/80">
          <Shield size={11} />
          Read-only mode
        </div>
      </div>

      {/* Schema Explorer */}
      <div className="border-b border-gray-800">
        <button
          onClick={() => setShowSchema(!showSchema)}
          className="w-full p-4 flex items-center gap-2 text-sm text-gray-400 hover:text-gray-200 transition-colors"
        >
          {showSchema ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
          <span className="font-medium">Schema Explorer</span>
        </button>
        {showSchema && schema && (
          <div className="px-4 pb-4 max-h-60 overflow-y-auto">
            <pre className="text-[11px] text-gray-500 whitespace-pre-wrap font-mono leading-relaxed">
              {schema}
            </pre>
          </div>
        )}
      </div>

      {/* Sample Queries */}
      <div className="flex-1 overflow-y-auto p-4">
        <p className="text-xs font-medium text-gray-500 uppercase tracking-wider mb-3">
          Sample queries
        </p>
        <div className="space-y-1.5">
          {SAMPLE_QUERIES.map((q, i) => (
            <button
              key={i}
              onClick={() => onSampleQuery(q)}
              className="w-full text-left px-3 py-2 text-xs text-gray-400 hover:text-gray-100 hover:bg-gray-800/60 rounded-lg transition-all duration-150"
            >
              {q}
            </button>
          ))}
        </div>
      </div>

      {/* Clear */}
      <div className="p-4 border-t border-gray-800">
        <button
          onClick={onClear}
          className="w-full flex items-center justify-center gap-2 px-3 py-2 text-sm text-gray-500 hover:text-red-400 hover:bg-red-950/20 rounded-lg transition-colors"
        >
          <Trash2 size={14} />
          Clear conversation
        </button>
      </div>
    </aside>
  );
}

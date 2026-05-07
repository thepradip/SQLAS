import { useState } from "react";
import { ChevronDown, ChevronRight, List, Table2, Database, CheckCircle2, Cpu } from "lucide-react";

const TOOL_ICONS = {
  create_plan:         <Cpu size={12} />,
  list_tables:         <List size={12} />,
  describe_table:      <Table2 size={12} />,
  execute_sql:         <Database size={12} />,
  final_answer:        <CheckCircle2 size={12} />,
  BLOCKED_execute_sql: <span style={{color:"#ff8e7a"}}>⊘</span>,
};

const TOOL_COLORS = {
  create_plan:         "text-yellow-300 bg-yellow-400/10 border-yellow-400/20",
  list_tables:         "text-sky-300 bg-sky-400/10 border-sky-400/20",
  describe_table:      "text-violet-300 bg-violet-400/10 border-violet-400/20",
  execute_sql:         "text-[var(--accent)] bg-[rgba(102,217,239,0.08)] border-[rgba(102,217,239,0.18)]",
  final_answer:        "text-emerald-300 bg-emerald-400/10 border-emerald-400/20",
  BLOCKED_execute_sql: "text-red-300 bg-red-400/10 border-red-400/20",
};

export default function AgentSteps({ steps }) {
  const [open, setOpen] = useState(false);
  const [expandedStep, setExpandedStep] = useState(null);

  if (!steps || steps.length === 0) return null;

  return (
    <div className="mt-3 rounded-[1.4rem] border border-[rgba(125,168,214,0.12)] bg-[rgba(255,255,255,0.015)] overflow-hidden">
      {/* Header */}
      <button
        onClick={() => setOpen((p) => !p)}
        className="w-full flex items-center gap-2 px-4 py-3 text-left hover:bg-[rgba(255,255,255,0.02)] transition-colors"
      >
        <Cpu size={13} className="text-[var(--accent)]" />
        <span className="text-[11px] uppercase tracking-[0.18em] text-[var(--accent)]">
          Agent Reasoning
        </span>
        <span className="ml-1 text-[11px] text-[var(--text-3)]">
          {steps.length} step{steps.length !== 1 ? "s" : ""}
        </span>
        <div className="ml-auto flex gap-1">
          {[...new Set(steps.map((s) => s.tool))].map((tool) => (
            <span
              key={tool}
              className={`inline-flex items-center gap-1 px-1.5 py-0.5 rounded-full border text-[10px] ${TOOL_COLORS[tool] || "text-[var(--text-3)] border-[rgba(125,168,214,0.12)]"}`}
            >
              {TOOL_ICONS[tool]}
              {tool.replace("_", " ")}
            </span>
          ))}
        </div>
        {open ? <ChevronDown size={13} className="text-[var(--text-3)] ml-1" /> : <ChevronRight size={13} className="text-[var(--text-3)] ml-1" />}
      </button>

      {/* Steps */}
      {open && (
        <div className="px-4 pb-4 space-y-2 border-t border-[rgba(125,168,214,0.08)]">
          {steps.map((step, i) => (
            <StepRow
              key={i}
              step={step}
              index={i}
              isLast={i === steps.length - 1}
              expanded={expandedStep === i}
              onToggle={() => setExpandedStep(expandedStep === i ? null : i)}
            />
          ))}
        </div>
      )}
    </div>
  );
}

function StepRow({ step, index, isLast, expanded, onToggle }) {
  const colorClass = TOOL_COLORS[step.tool] || "text-[var(--text-3)] border-[rgba(125,168,214,0.12)]";
  const icon = TOOL_ICONS[step.tool] || <Cpu size={12} />;
  const hasDetail = step.args && Object.keys(step.args).length > 0;

  return (
    <div className="relative pl-5 pt-2">
      {/* Connector line */}
      {!isLast && (
        <div className="absolute left-[9px] top-[22px] bottom-[-6px] w-px bg-[rgba(125,168,214,0.12)]" />
      )}
      {/* Dot */}
      <div className={`absolute left-0 top-[12px] w-[18px] h-[18px] rounded-full border flex items-center justify-center ${colorClass}`}>
        {icon}
      </div>

      <button
        onClick={hasDetail ? onToggle : undefined}
        className={`w-full text-left rounded-xl px-3 py-2 border border-transparent transition-colors ${
          hasDetail ? "hover:bg-[rgba(255,255,255,0.03)] cursor-pointer" : "cursor-default"
        }`}
      >
        <div className="flex items-center gap-2">
          <span className={`text-[11px] font-medium ${colorClass.split(" ")[0]}`}>
            {step.tool === "BLOCKED_execute_sql" ? "⊘ blocked — plan required" : step.tool.replace(/_/g, " ")}
          </span>
          {/* create_plan: show query understanding inline */}
          {step.tool === "create_plan" && step.args?.query_understanding && (
            <span className="text-[11px] text-[var(--text-3)] truncate max-w-[300px]">
              — {step.args.query_understanding.slice(0, 80)}
            </span>
          )}
          {/* Primary arg preview */}
          {step.args?.table_name && (
            <span className="text-[11px] text-[var(--text-3)]">— {step.args.table_name}</span>
          )}
          {step.args?.sql && (
            <span className="text-[11px] text-[var(--text-3)] truncate max-w-[260px] font-mono">
              — {step.args.sql.replace(/\s+/g, " ").slice(0, 60)}…
            </span>
          )}
          {hasDetail && (
            <span className="ml-auto">
              {expanded
                ? <ChevronDown size={11} className="text-[var(--text-3)]" />
                : <ChevronRight size={11} className="text-[var(--text-3)]" />}
            </span>
          )}
        </div>

        {expanded && (
          <div className="mt-2 space-y-2">
            {Object.entries(step.args || {}).map(([k, v]) => (
              <div key={k}>
                <div className="text-[10px] text-[var(--text-3)] mb-0.5 uppercase tracking-[0.1em]">{k}</div>
                <pre className="text-[11px] text-[var(--text-2)] bg-[rgba(0,0,0,0.25)] rounded-lg px-3 py-2 overflow-x-auto whitespace-pre-wrap">
                  {typeof v === "string" ? v : JSON.stringify(v, null, 2)}
                </pre>
              </div>
            ))}
            {step.result_preview && (
              <div>
                <div className="text-[10px] text-[var(--text-3)] mb-0.5 uppercase tracking-[0.1em]">Result</div>
                <pre className="text-[11px] text-[var(--text-2)] bg-[rgba(0,0,0,0.25)] rounded-lg px-3 py-2 overflow-x-auto whitespace-pre-wrap max-h-48">
                  {step.result_preview}
                </pre>
              </div>
            )}
          </div>
        )}
      </button>
    </div>
  );
}

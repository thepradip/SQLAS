import { useState } from "react";
import ReactMarkdown from "react-markdown";
import {
  User, Bot, Clock, Check, X,
  ThumbsUp, ThumbsDown, BarChart3, Cpu, Zap, Download,
} from "lucide-react";
import DataVisualization from "./DataVisualization";
import AgentSteps from "./AgentSteps";

export default function MessageBubble({ message, onFeedback }) {
  const isUser = message.role === "user";
  const [feedback, setFeedback] = useState(null); // "up" | "down" | null
  const [showMetrics, setShowMetrics] = useState(false);
  const [feedbackComment, setFeedbackComment] = useState("");
  const [showCommentBox, setShowCommentBox] = useState(false);

  if (isUser) {
    return (
      <div className="flex items-start gap-4 py-2 justify-end">
        <div className="max-w-3xl rounded-[1.6rem] rounded-tr-md bg-[linear-gradient(135deg,rgba(102,217,239,0.18)_0%,rgba(128,255,211,0.12)_100%)] border border-[rgba(102,217,239,0.18)] px-5 py-4 shadow-[0_20px_40px_rgba(0,0,0,0.14)]">
          <div className="text-sm text-[var(--text-1)] leading-relaxed">{message.content}</div>
        </div>
        <div className="w-10 h-10 rounded-2xl bg-[rgba(255,255,255,0.06)] border border-[rgba(125,168,214,0.14)] flex items-center justify-center flex-shrink-0">
          <User size={16} className="text-[var(--text-2)]" />
        </div>
      </div>
    );
  }

  const handleFeedback = (value) => {
    const isUp = value === "up";
    if (feedback === value) {
      setFeedback(null);
      return;
    }
    setFeedback(value);
    if (message.trace_id && onFeedback) {
      onFeedback(message.trace_id, isUp, feedbackComment || null);
    }
    if (!isUp) {
      setShowCommentBox(true);
    }
  };

  const submitComment = () => {
    if (message.trace_id && onFeedback && feedbackComment.trim()) {
      onFeedback(message.trace_id, feedback === "up", feedbackComment);
    }
    setShowCommentBox(false);
  };

  const m = message.metrics;

  return (
    <div className="flex items-start gap-4 py-2">
      <div className="w-10 h-10 rounded-2xl bg-[rgba(102,217,239,0.12)] border border-[rgba(102,217,239,0.16)] flex items-center justify-center flex-shrink-0">
        <Bot size={16} className="text-[var(--accent)]" />
      </div>
      <div className="glass-panel rounded-[1.7rem] rounded-tl-md flex-1 min-w-0 px-5 py-4">
        {/* Status + latency badges */}
        {message.success !== undefined && (
          <div className="flex items-center gap-2 mb-3 flex-wrap">
            {message.success ? (
              <span className="inline-flex items-center gap-1 px-2.5 py-1 bg-emerald-400/10 border border-emerald-300/10 rounded-full text-[11px] text-emerald-200">
                <Check size={10} /> Success
              </span>
            ) : (
              <span className="inline-flex items-center gap-1 px-2.5 py-1 bg-red-400/10 border border-red-300/10 rounded-full text-[11px] text-red-200">
                <X size={10} /> Error
              </span>
            )}

            {/* Agent mode badge */}
            {message.agent_mode === "react" ? (
              <span className="inline-flex items-center gap-1 px-2.5 py-1 bg-violet-400/10 border border-violet-300/15 rounded-full text-[11px] text-violet-300">
                <Cpu size={10} /> Agentic · {message.agent_steps?.length ?? 0} steps
              </span>
            ) : (
              <span className="inline-flex items-center gap-1 px-2.5 py-1 bg-[rgba(102,217,239,0.08)] border border-[rgba(102,217,239,0.12)] rounded-full text-[11px] text-[var(--accent)]">
                <Zap size={10} /> Pipeline
              </span>
            )}

            {m && (
              <>
                <span className="inline-flex items-center gap-1 text-[11px] text-[var(--text-3)]">
                  <Clock size={10} />
                  {m.total_latency_ms.toLocaleString()}ms
                </span>
                {m.query_type && (
                  <span className="px-2 py-1 bg-[rgba(102,217,239,0.12)] border border-[rgba(102,217,239,0.12)] rounded-full text-[10px] text-[var(--accent)] uppercase tracking-[0.12em]">
                    {m.query_type}
                  </span>
                )}
                {m.retry_count > 0 && (
                  <span className="px-2 py-1 bg-amber-400/10 border border-amber-300/10 rounded-full text-[10px] text-amber-200 uppercase tracking-[0.12em]">
                    {m.retry_count} retry
                  </span>
                )}
                {m.cache_hit && (
                  <span className="px-2 py-1 bg-sky-400/10 border border-sky-300/10 rounded-full text-[10px] text-sky-300 uppercase tracking-[0.12em]">
                    cache {m.cache_type}
                  </span>
                )}
              </>
            )}
          </div>
        )}

        {/* Response text */}
        <div className="prose text-sm text-[var(--text-2)]">
          <ReactMarkdown>{message.content}</ReactMarkdown>
        </div>

        {/* Action bar: feedback + metrics toggle */}
        {message.trace_id && (
          <div className="flex items-center gap-1 mt-4 pt-3 border-t border-[rgba(125,168,214,0.12)]">
            {/* Thumbs up */}
            <button
              onClick={() => handleFeedback("up")}
              className={`p-1.5 rounded-md transition-all ${
                feedback === "up"
                  ? "bg-emerald-400/10 text-emerald-200"
                  : "text-[var(--text-3)] hover:text-[var(--text-1)] hover:bg-[rgba(255,255,255,0.04)]"
              }`}
              title="Helpful"
            >
              <ThumbsUp size={14} />
            </button>

            {/* Thumbs down */}
            <button
              onClick={() => handleFeedback("down")}
              className={`p-1.5 rounded-md transition-all ${
                feedback === "down"
                  ? "bg-red-400/10 text-red-200"
                  : "text-[var(--text-3)] hover:text-[var(--text-1)] hover:bg-[rgba(255,255,255,0.04)]"
              }`}
              title="Not helpful"
            >
              <ThumbsDown size={14} />
            </button>

            <div className="w-px h-4 bg-[rgba(125,168,214,0.12)] mx-1" />

            {/* Metrics toggle */}
            {m && (
              <button
                onClick={() => setShowMetrics(!showMetrics)}
                className={`inline-flex items-center gap-1 px-2 py-1 rounded-md text-[11px] transition-all ${
                  showMetrics
                    ? "bg-[rgba(255,255,255,0.05)] text-[var(--text-1)]"
                    : "text-[var(--text-3)] hover:text-[var(--text-1)] hover:bg-[rgba(255,255,255,0.04)]"
                }`}
              >
                <BarChart3 size={12} />
                Metrics
              </button>
            )}

            {/* Export CSV */}
            {message.data?.columns && (
              <button
                onClick={() => exportCsv(message)}
                className="inline-flex items-center gap-1 px-2 py-1 rounded-md text-[11px] text-[var(--text-3)] hover:text-[var(--text-1)] hover:bg-[rgba(255,255,255,0.04)] transition-all"
                title="Download results as CSV"
              >
                <Download size={12} /> CSV
              </button>
            )}

            {/* Trace ID */}
            <span className="ml-auto text-[10px] text-[var(--text-3)]/70 font-mono">
              {message.trace_id.slice(0, 12)}...
            </span>
          </div>
        )}

        {/* Feedback comment */}
        {showCommentBox && (
          <div className="mt-2 flex items-center gap-2">
            <input
              type="text"
              value={feedbackComment}
              onChange={(e) => setFeedbackComment(e.target.value)}
              placeholder="What could be improved?"
              className="flex-1 glass-panel rounded-xl px-3 py-2 text-xs text-[var(--text-2)] placeholder:text-[var(--text-3)] focus:outline-none focus:border-[rgba(102,217,239,0.28)]"
              onKeyDown={(e) => e.key === "Enter" && submitComment()}
            />
            <button
              onClick={submitComment}
              className="px-3 py-2 bg-[rgba(255,255,255,0.05)] hover:bg-[rgba(255,255,255,0.08)] text-xs text-[var(--text-2)] rounded-xl transition-colors"
            >
              Send
            </button>
          </div>
        )}

        {/* Metrics panel */}
        {showMetrics && m && <MetricsPanel metrics={m} />}

        {/* ReAct agent steps */}
        {message.agent_steps && message.agent_steps.length > 0 && (
          <AgentSteps steps={message.agent_steps} />
        )}

        {/* Visualization */}
        {message.visualization && <DataVisualization visualization={message.visualization} />}
      </div>
    </div>
  );
}


async function exportCsv(message) {
  try {
    const res = await fetch("/api/export/csv", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        columns: message.data.columns,
        rows: message.data.rows,
        filename: `query_${Date.now()}.csv`,
      }),
    });
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `query_${Date.now()}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  } catch {
    // silent fail — user will notice the download didn't start
  }
}


function MetricsPanel({ metrics: m }) {
  return (
    <div className="mt-3 glass-panel rounded-2xl p-4">
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-[11px]">
        <Metric label="Total Latency" value={`${m.total_latency_ms.toLocaleString()}ms`} />
        <Metric label="SQL Generation" value={`${m.generation_latency_ms.toLocaleString()}ms`} />
        <Metric label="SQL Execution" value={`${m.sql_execution_ms.toLocaleString()}ms`} />
        <Metric label="Narration" value={`${m.narration_latency_ms.toLocaleString()}ms`} />

        {m.result_rows != null && <Metric label="Result Rows" value={m.result_rows.toLocaleString()} />}
        {m.result_columns != null && <Metric label="Result Columns" value={m.result_columns} />}
        <Metric label="Retries" value={m.retry_count} highlight={m.retry_count > 0} />
        {m.steps_taken != null && <Metric label="Agent Steps" value={m.steps_taken} />}
        {m.tokens_saved > 0 && <Metric label="Tokens Saved" value={m.tokens_saved.toLocaleString()} />}
        {m.query_type && <Metric label="Query Type" value={m.query_type} />}

        {m.join_count != null && <Metric label="Joins" value={m.join_count} />}
        {m.table_count != null && <Metric label="Tables" value={m.table_count} />}
        {m.sql_length != null && <Metric label="SQL Length" value={`${m.sql_length} chars`} />}
        {m.where_conditions != null && <Metric label="WHERE Conditions" value={m.where_conditions} />}
      </div>

      {/* SQL features */}
      <div className="flex flex-wrap gap-1.5 mt-2.5 pt-2 border-t border-[rgba(125,168,214,0.12)]">
        {m.has_aggregation && <Tag text="Aggregation" />}
        {m.has_group_by && <Tag text="GROUP BY" />}
        {m.has_order_by && <Tag text="ORDER BY" />}
        {m.has_case_when && <Tag text="CASE WHEN" />}
        {m.has_distinct && <Tag text="DISTINCT" />}
        {m.has_limit && <Tag text="LIMIT" />}
        {m.has_having && <Tag text="HAVING" />}
        {m.has_window_function && <Tag text="Window Fn" />}
        {m.has_null_handling && <Tag text="NULL Handling" />}
        {m.cte_count > 0 && <Tag text={`${m.cte_count} CTE(s)`} />}
        {m.subquery_count > 0 && <Tag text={`${m.subquery_count} Subquery`} />}
      </div>
    </div>
  );
}


function Metric({ label, value, highlight }) {
  return (
    <div>
      <div className="text-[var(--text-3)] mb-0.5">{label}</div>
      <div className={`font-medium ${highlight ? "text-amber-200" : "text-[var(--text-2)]"}`}>
        {value}
      </div>
    </div>
  );
}


function Tag({ text }) {
  return (
    <span className="px-1.5 py-0.5 bg-[rgba(255,255,255,0.04)] border border-[rgba(125,168,214,0.12)] rounded-full text-[10px] text-[var(--text-3)]">
      {text}
    </span>
  );
}

import { useState } from "react";
import ReactMarkdown from "react-markdown";
import {
  User, Bot, Clock, Check, X,
  ThumbsUp, ThumbsDown, ChevronDown, ChevronRight, BarChart3,
} from "lucide-react";
import CodeBlock from "./CodeBlock";
import DataTable from "./DataTable";

export default function MessageBubble({ message, onFeedback }) {
  const isUser = message.role === "user";
  const [feedback, setFeedback] = useState(null); // "up" | "down" | null
  const [showMetrics, setShowMetrics] = useState(false);
  const [feedbackComment, setFeedbackComment] = useState("");
  const [showCommentBox, setShowCommentBox] = useState(false);

  if (isUser) {
    return (
      <div className="flex items-start gap-3 py-4">
        <div className="w-8 h-8 bg-gray-700 rounded-lg flex items-center justify-center flex-shrink-0">
          <User size={16} className="text-gray-300" />
        </div>
        <div className="text-sm text-gray-200 pt-1.5">{message.content}</div>
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
    <div className="flex items-start gap-3 py-4">
      <div className="w-8 h-8 bg-indigo-600/20 rounded-lg flex items-center justify-center flex-shrink-0">
        <Bot size={16} className="text-indigo-400" />
      </div>
      <div className="flex-1 min-w-0">
        {/* Status + latency badges */}
        {message.success !== undefined && (
          <div className="flex items-center gap-2 mb-2 flex-wrap">
            {message.success ? (
              <span className="inline-flex items-center gap-1 px-2 py-0.5 bg-emerald-950/50 border border-emerald-800/50 rounded-full text-[11px] text-emerald-400">
                <Check size={10} /> Success
              </span>
            ) : (
              <span className="inline-flex items-center gap-1 px-2 py-0.5 bg-red-950/50 border border-red-800/50 rounded-full text-[11px] text-red-400">
                <X size={10} /> Error
              </span>
            )}
            {m && (
              <>
                <span className="inline-flex items-center gap-1 text-[11px] text-gray-500">
                  <Clock size={10} />
                  {m.total_latency_ms.toLocaleString()}ms total
                </span>
                {m.query_type && (
                  <span className="px-1.5 py-0.5 bg-indigo-950/40 border border-indigo-800/30 rounded text-[10px] text-indigo-400">
                    {m.query_type}
                  </span>
                )}
                {m.retry_count > 0 && (
                  <span className="px-1.5 py-0.5 bg-amber-950/40 border border-amber-800/30 rounded text-[10px] text-amber-400">
                    {m.retry_count} retry
                  </span>
                )}
              </>
            )}
          </div>
        )}

        {/* Response text */}
        <div className="prose text-sm text-gray-300">
          <ReactMarkdown>{message.content}</ReactMarkdown>
        </div>

        {/* Action bar: feedback + metrics toggle */}
        {message.trace_id && (
          <div className="flex items-center gap-1 mt-3 pt-2 border-t border-gray-800/50">
            {/* Thumbs up */}
            <button
              onClick={() => handleFeedback("up")}
              className={`p-1.5 rounded-md transition-all ${
                feedback === "up"
                  ? "bg-emerald-950/50 text-emerald-400"
                  : "text-gray-600 hover:text-gray-300 hover:bg-gray-800/50"
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
                  ? "bg-red-950/50 text-red-400"
                  : "text-gray-600 hover:text-gray-300 hover:bg-gray-800/50"
              }`}
              title="Not helpful"
            >
              <ThumbsDown size={14} />
            </button>

            <div className="w-px h-4 bg-gray-800 mx-1" />

            {/* Metrics toggle */}
            {m && (
              <button
                onClick={() => setShowMetrics(!showMetrics)}
                className={`inline-flex items-center gap-1 px-2 py-1 rounded-md text-[11px] transition-all ${
                  showMetrics
                    ? "bg-gray-800 text-gray-300"
                    : "text-gray-600 hover:text-gray-300 hover:bg-gray-800/50"
                }`}
              >
                <BarChart3 size={12} />
                Metrics
              </button>
            )}

            {/* Trace ID */}
            <span className="ml-auto text-[10px] text-gray-700 font-mono">
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
              className="flex-1 bg-gray-800 border border-gray-700 rounded-lg px-3 py-1.5 text-xs text-gray-300 placeholder-gray-600 focus:outline-none focus:border-indigo-500"
              onKeyDown={(e) => e.key === "Enter" && submitComment()}
            />
            <button
              onClick={submitComment}
              className="px-3 py-1.5 bg-gray-800 hover:bg-gray-700 text-xs text-gray-400 rounded-lg transition-colors"
            >
              Send
            </button>
          </div>
        )}

        {/* Metrics panel */}
        {showMetrics && m && <MetricsPanel metrics={m} />}

        {/* SQL + Data */}
        {message.sql && <CodeBlock sql={message.sql} />}
        {message.data && message.data.rows.length > 0 && (
          <DataTable data={message.data} />
        )}
      </div>
    </div>
  );
}


function MetricsPanel({ metrics: m }) {
  return (
    <div className="mt-2 bg-gray-900/80 border border-gray-800 rounded-lg p-3">
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-[11px]">
        <Metric label="Total Latency" value={`${m.total_latency_ms.toLocaleString()}ms`} />
        <Metric label="SQL Generation" value={`${m.generation_latency_ms.toLocaleString()}ms`} />
        <Metric label="SQL Execution" value={`${m.sql_execution_ms.toLocaleString()}ms`} />
        <Metric label="Narration" value={`${m.narration_latency_ms.toLocaleString()}ms`} />

        {m.result_rows != null && <Metric label="Result Rows" value={m.result_rows.toLocaleString()} />}
        {m.result_columns != null && <Metric label="Result Columns" value={m.result_columns} />}
        <Metric label="Retries" value={m.retry_count} highlight={m.retry_count > 0} />
        {m.query_type && <Metric label="Query Type" value={m.query_type} />}

        {m.join_count != null && <Metric label="Joins" value={m.join_count} />}
        {m.table_count != null && <Metric label="Tables" value={m.table_count} />}
        {m.sql_length != null && <Metric label="SQL Length" value={`${m.sql_length} chars`} />}
        {m.where_conditions != null && <Metric label="WHERE Conditions" value={m.where_conditions} />}
      </div>

      {/* SQL features */}
      <div className="flex flex-wrap gap-1.5 mt-2.5 pt-2 border-t border-gray-800">
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
      <div className="text-gray-600 mb-0.5">{label}</div>
      <div className={`font-medium ${highlight ? "text-amber-400" : "text-gray-300"}`}>
        {value}
      </div>
    </div>
  );
}


function Tag({ text }) {
  return (
    <span className="px-1.5 py-0.5 bg-gray-800 border border-gray-700 rounded text-[10px] text-gray-400">
      {text}
    </span>
  );
}

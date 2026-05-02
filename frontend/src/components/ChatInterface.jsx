import { useState, useRef, useEffect } from "react";
import { Send, Loader2, Cpu, Zap } from "lucide-react";
import MessageBubble from "./MessageBubble";

const LOADING_STAGES = [
  { after: 0,    text: "Analyzing your question..." },
  { after: 2500, text: "Exploring database schema..." },
  { after: 5500, text: "Generating SQL query..." },
  { after: 9000, text: "Executing and synthesizing results..." },
];

export default function ChatInterface({ messages, loading, onSend, onFeedback, agenticMode, onToggleAgentic }) {
  const [input, setInput] = useState("");
  const [loadingText, setLoadingText] = useState(LOADING_STAGES[0].text);
  const endRef = useRef(null);
  const inputRef = useRef(null);
  const stageTimers = useRef([]);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  useEffect(() => {
    stageTimers.current.forEach(clearTimeout);
    stageTimers.current = [];
    if (loading) {
      setLoadingText(LOADING_STAGES[0].text);
      LOADING_STAGES.slice(1).forEach(({ after, text }) => {
        stageTimers.current.push(setTimeout(() => setLoadingText(text), after));
      });
    }
    return () => stageTimers.current.forEach(clearTimeout);
  }, [loading]);

  const handleSubmit = (e) => {
    e.preventDefault();
    const q = input.trim();
    if (!q || loading) return;
    onSend(q);
    setInput("");
  };

  return (
    <main className="flex-1 flex flex-col min-w-0">
      {/* Messages */}
      <div className="flex-1 overflow-y-auto">
        {messages.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center text-center px-8">
            <div className="glass-panel-strong rounded-[2rem] px-10 py-12 max-w-2xl w-full">
              <div className="w-20 h-20 mx-auto rounded-[1.75rem] flex items-center justify-center mb-6 bg-[linear-gradient(135deg,rgba(102,217,239,0.18)_0%,rgba(128,255,211,0.18)_100%)] border border-[rgba(102,217,239,0.18)]">
                <span className="text-4xl text-[var(--accent)]">&#9889;</span>
              </div>
              <div className="text-[11px] uppercase tracking-[0.24em] text-[var(--text-3)] mb-3">
                Query your health data
              </div>
              <h2 className="text-3xl font-semibold tracking-tight text-[var(--text-1)] mb-3">
                Ask anything about your data
              </h2>
              <p className="text-sm text-[var(--text-2)] max-w-xl mx-auto leading-relaxed">
                Translate natural language into SQL, return a brief narrative, and surface the result as a chart-ready view inside the same workspace.
              </p>
            </div>
          </div>
        ) : (
          <div className="max-w-5xl mx-auto px-8 py-8 space-y-3">
            {messages.map((msg, i) => (
              <MessageBubble key={i} message={msg} onFeedback={onFeedback} />
            ))}
            {loading && (
              <div className="glass-panel rounded-2xl flex items-center gap-3 py-4 px-5">
                <div className="w-9 h-9 rounded-xl bg-[rgba(102,217,239,0.14)] flex items-center justify-center flex-shrink-0">
                  <Loader2 size={16} className="text-[var(--accent)] animate-spin" />
                </div>
                <div className="text-sm text-[var(--text-2)] transition-all">{loadingText}</div>
                {agenticMode && (
                  <span className="ml-auto inline-flex items-center gap-1 text-[11px] text-violet-300 px-2 py-1 bg-violet-400/10 border border-violet-300/15 rounded-full">
                    <Cpu size={10} /> Agentic
                  </span>
                )}
              </div>
            )}
            <div ref={endRef} />
          </div>
        )}
      </div>

      {/* Input */}
      <div className="border-t border-[rgba(125,168,214,0.14)] bg-[rgba(6,14,24,0.42)] backdrop-blur-xl p-5">
        <form onSubmit={handleSubmit} className="max-w-5xl mx-auto flex items-center gap-3">
          <input
            ref={inputRef}
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask a question about your data..."
            className="flex-1 glass-panel rounded-2xl px-5 py-4 text-sm text-[var(--text-1)] placeholder:text-[var(--text-3)] focus:outline-none focus:border-[rgba(102,217,239,0.28)] focus:ring-1 focus:ring-[rgba(102,217,239,0.22)] transition-all"
            disabled={loading}
          />

          {/* Agentic toggle */}
          <button
            type="button"
            onClick={onToggleAgentic}
            disabled={loading}
            title={agenticMode ? "Agentic mode on — click to use fast pipeline" : "Pipeline mode — click for agentic reasoning"}
            className={`rounded-2xl px-4 py-4 border transition-all flex items-center gap-2 text-[13px] ${
              agenticMode
                ? "bg-violet-400/10 border-violet-300/25 text-violet-300 hover:bg-violet-400/15"
                : "glass-panel border-[rgba(125,168,214,0.14)] text-[var(--text-3)] hover:text-[var(--text-2)]"
            }`}
          >
            {agenticMode ? <Cpu size={16} /> : <Zap size={16} />}
          </button>

          <button
            type="submit"
            disabled={loading || !input.trim()}
            className="rounded-2xl px-5 py-4 bg-[linear-gradient(135deg,#66d9ef_0%,#80ffd3_100%)] hover:brightness-105 disabled:bg-slate-800 disabled:text-slate-500 disabled:brightness-100 text-slate-950 shadow-[0_18px_38px_rgba(102,217,239,0.18)] transition-all"
          >
            {loading ? <Loader2 size={18} className="animate-spin" /> : <Send size={18} />}
          </button>
        </form>
        <p className="text-center text-[11px] text-[var(--text-3)] mt-3 tracking-[0.12em] uppercase">
          {agenticMode
            ? "Agentic mode — explores schema, runs multi-step reasoning"
            : "Read-only access. No data will be modified."}
        </p>
      </div>
    </main>
  );
}

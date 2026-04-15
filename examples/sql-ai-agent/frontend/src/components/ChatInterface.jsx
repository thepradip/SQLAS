import { useState, useRef, useEffect } from "react";
import { Send, Loader2 } from "lucide-react";
import MessageBubble from "./MessageBubble";

export default function ChatInterface({ messages, loading, onSend, onFeedback }) {
  const [input, setInput] = useState("");
  const endRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

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
            <div className="w-16 h-16 bg-indigo-600/10 border border-indigo-500/20 rounded-2xl flex items-center justify-center mb-5">
              <span className="text-3xl">&#9889;</span>
            </div>
            <h2 className="text-xl font-semibold text-gray-200 mb-2">
              Ask anything about your data
            </h2>
            <p className="text-sm text-gray-500 max-w-md leading-relaxed">
              I'll translate your question into SQL, run it against the database,
              and explain the results in plain language.
            </p>
          </div>
        ) : (
          <div className="max-w-4xl mx-auto px-6 py-6 space-y-1">
            {messages.map((msg, i) => (
              <MessageBubble key={i} message={msg} onFeedback={onFeedback} />
            ))}
            {loading && (
              <div className="flex items-center gap-3 py-4 px-4">
                <div className="w-8 h-8 bg-indigo-600/20 rounded-lg flex items-center justify-center flex-shrink-0">
                  <Loader2 size={16} className="text-indigo-400 animate-spin" />
                </div>
                <div className="text-sm text-gray-500">
                  Generating SQL and analyzing...
                </div>
              </div>
            )}
            <div ref={endRef} />
          </div>
        )}
      </div>

      {/* Input */}
      <div className="border-t border-gray-800 bg-gray-900/50 p-4">
        <form
          onSubmit={handleSubmit}
          className="max-w-4xl mx-auto flex items-center gap-3"
        >
          <input
            ref={inputRef}
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask a question about your data..."
            className="flex-1 bg-gray-800 border border-gray-700 rounded-xl px-4 py-3 text-sm text-gray-100 placeholder-gray-500 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500/50 transition-all"
            disabled={loading}
          />
          <button
            type="submit"
            disabled={loading || !input.trim()}
            className="bg-indigo-600 hover:bg-indigo-500 disabled:bg-gray-700 disabled:text-gray-500 text-white rounded-xl p-3 transition-colors"
          >
            {loading ? (
              <Loader2 size={18} className="animate-spin" />
            ) : (
              <Send size={18} />
            )}
          </button>
        </form>
        <p className="text-center text-[11px] text-gray-600 mt-2">
          Read-only access. No data will be modified.
        </p>
      </div>
    </main>
  );
}

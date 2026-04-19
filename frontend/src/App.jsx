import { useState, useEffect, useRef } from "react";
import Sidebar from "./components/Sidebar";
import ChatInterface from "./components/ChatInterface";

const API = "/api";

export default function App() {
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [schema, setSchema] = useState(null);
  const [showSchema, setShowSchema] = useState(false);
  const [health, setHealth] = useState(null);
  const [conversationId] = useState(() => crypto.randomUUID());
  const inFlightRef = useRef(false);
  const requestIdRef = useRef(0);

  useEffect(() => {
    fetch(`${API}/health`)
      .then((r) => r.json())
      .then(setHealth)
      .catch(() => {});
    fetch(`${API}/schema`)
      .then((r) => r.json())
      .then((d) => setSchema(d.schema_text))
      .catch(() => {});
  }, []);

  const sendQuery = async (query) => {
    const normalizedQuery = query.trim();
    if (!normalizedQuery || inFlightRef.current) return;

    inFlightRef.current = true;
    const requestId = requestIdRef.current + 1;
    requestIdRef.current = requestId;
    const userMsg = { role: "user", content: normalizedQuery };
    setMessages((prev) => [...prev, userMsg]);
    setLoading(true);

    try {
      const res = await fetch(`${API}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: normalizedQuery, conversation_id: conversationId }),
      });
      const data = await res.json();
      if (requestId !== requestIdRef.current) return;
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: data.response,
          sql: data.sql,
          data: data.data,
          visualization: data.visualization,
          success: data.success,
          trace_id: data.trace_id,
          metrics: data.metrics,
        },
      ]);
    } catch (err) {
      if (requestId !== requestIdRef.current) return;
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: "Failed to connect to the server. Is the backend running?",
          success: false,
        },
      ]);
    } finally {
      if (requestId === requestIdRef.current) {
        inFlightRef.current = false;
        setLoading(false);
      }
    }
  };

  const sendFeedback = async (traceId, value, comment) => {
    try {
      await fetch(`${API}/feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ trace_id: traceId, value, comment }),
      });
    } catch {
      // silent fail
    }
  };

  const clearChat = () => {
    requestIdRef.current += 1;
    inFlightRef.current = false;
    setLoading(false);
    setMessages([]);
    fetch(`${API}/conversations/${conversationId}`, { method: "DELETE" }).catch(
      () => {}
    );
  };

  return (
    <div className="app-shell flex min-h-screen">
      <Sidebar
        health={health}
        schema={schema}
        showSchema={showSchema}
        setShowSchema={setShowSchema}
        loading={loading}
        onClear={clearChat}
        onSampleQuery={sendQuery}
      />
      <ChatInterface
        messages={messages}
        loading={loading}
        onSend={sendQuery}
        onFeedback={sendFeedback}
      />
    </div>
  );
}

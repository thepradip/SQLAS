import { useState, useEffect } from "react";
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
    const userMsg = { role: "user", content: query };
    setMessages((prev) => [...prev, userMsg]);
    setLoading(true);

    try {
      const res = await fetch(`${API}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, conversation_id: conversationId }),
      });
      const data = await res.json();
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: data.response,
          sql: data.sql,
          data: data.data,
          success: data.success,
          trace_id: data.trace_id,
          metrics: data.metrics,
        },
      ]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: "Failed to connect to the server. Is the backend running?",
          success: false,
        },
      ]);
    } finally {
      setLoading(false);
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
    setMessages([]);
    fetch(`${API}/conversations/${conversationId}`, { method: "DELETE" }).catch(
      () => {}
    );
  };

  return (
    <div className="flex h-screen bg-gray-950">
      <Sidebar
        health={health}
        schema={schema}
        showSchema={showSchema}
        setShowSchema={setShowSchema}
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

import { useState, useRef, useEffect } from "react";
import "./App.css";

const API_URL = "http://127.0.0.1:8000/chat";

function App() {
  const [messages, setMessages] = useState([]);
  const [sessionId] = useState(() => "session_" + Date.now());
  const [input, setInput] = useState("");
  const [clarifying, setClarifying] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const chatBoxRef = useRef(null);

  useEffect(() => {
    if (chatBoxRef.current) {
      chatBoxRef.current.scrollTop = chatBoxRef.current.scrollHeight;
    }
  }, [messages, clarifying]);

  const sendMessage = async (text, answers = null) => {
    if (text) {
      setMessages((prev) => [...prev, { sender: "user", text }]);
    }

    setIsLoading(true);
    try {
      const res = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionId,
          text,
          answers,
        }),
      });

      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.detail || "Backend error");
      }

      if (data.type === "clarify") {
        setMessages((prev) => [
          ...prev,
          ...data.questions.map((q) => ({
            sender: "bot",
            text: q.text,
            id: q.id,
          })),
        ]);
        setClarifying(data.questions);
      } else if (data.type === "answer") {
        setMessages((prev) => [...prev, { sender: "bot", text: data.answer }]);
        setClarifying([]);
      }
    } catch (err) {
      console.error(err);
      setMessages((prev) => [
        ...prev,
        { sender: "bot", text: `Error: ${err.message}` },
      ]);
      setClarifying([]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    if (clarifying.length > 0) {
      const currentQuestion = clarifying[0];
      const answersObj = { [currentQuestion.id]: input };
      sendMessage(input, answersObj);

    } else {
      sendMessage(input);
    }
    setInput("");
  };

  return (
    <div className="chat-container">
      {/* Header */}
      <div className="chat-header">
        <h1 className="chat-title">Prompt Engineering Chatbot</h1>
        <p className="chat-subtitle">Powered by AI • Ask me anything</p>
      </div>

      {/* Chat Messages */}
      <div ref={chatBoxRef} className="chat-messages">
        {messages.length === 0 && (
          <div className="empty-state">
            <div className="empty-icon">💬</div>
            <h2>Start a conversation</h2>
            <p>Ask me anything about prompt engineering, AI, or get help with your questions</p>
          </div>
        )}

        {messages.map((msg, idx) => (
          <div
            key={idx}
            className={`message-wrapper ${msg.sender === "user" ? "user" : "bot"}`}
          >
            <div className={`message ${msg.sender}`}>
              <p>{msg.text}</p>
            </div>
          </div>
        ))}

        {isLoading && (
          <div className="message-wrapper bot">
            <div className="message bot">
              <div className="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Input Area */}
      <div className="chat-input-container">
        <form onSubmit={handleSubmit} className="chat-input-form">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder={
              clarifying.length > 0
                ? "Answer the clarifying question..."
                : "Type your message..."
            }
            disabled={isLoading}
            className="chat-input"
          />
          <button
            type="submit"
            disabled={!input.trim() || isLoading}
            className="send-button"
          >
            Send ➤
          </button>
        </form>
      </div>
    </div>
  );
}

export default App;

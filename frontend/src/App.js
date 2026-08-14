import { useState, useRef, useEffect, useCallback } from "react";
import ReactMarkdown from "react-markdown";
import "./App.css";

const API_URL = `${process.env.REACT_APP_API_URL || "http://127.0.0.1:8000"}/chat`;
const GRAMMAR_API_URL = "https://api.languagetool.org/v2/check";

const MODEL_LABELS = {
  groq: "Groq · Llama 3.1",
  openrouter: "OpenRouter · GPT-3.5",
};

function newSessionId() {
  return "session_" + Date.now() + "_" + Math.random().toString(36).slice(2, 8);
}

// Grammar suggestions are aimed at natural-language phrasing, not code/technical
// requests — running them on "write a c++ function..." produced nonsense
// corrections (LanguageTool has no notion of code syntax), so skip those.
const TECHNICAL_HINTS = /\b(code|python|javascript|react|fastapi|typescript|sql|api|algorithm|function|debug|css|html|java|docker|git|regex|c\+\+|c#)\b/i;

function CopyButton({ text }) {
  const [copied, setCopied] = useState(false);
  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      // clipboard unavailable — ignore silently
    }
  };
  return (
    <button className="copy-button" onClick={handleCopy} aria-label="Copy message" title="Copy">
      {copied ? "✓ Copied" : "⧉ Copy"}
    </button>
  );
}

function CompareAnswers({ alternates, winner }) {
  const [open, setOpen] = useState(false);
  if (!alternates) return null;

  const other = winner === "groq" ? "openrouter" : "groq";
  const otherText = alternates[other];
  if (!otherText) return null;

  return (
    <div className="compare-block">
      <button className="compare-toggle" onClick={() => setOpen((o) => !o)}>
        {open ? "▾ Hide" : "▸ Compare"} {MODEL_LABELS[other]}'s answer
      </button>
      {open && (
        <div className="compare-panel">
          <div className="compare-badge">{MODEL_LABELS[other]}</div>
          <div className="compare-text">
            <ReactMarkdown>{otherText}</ReactMarkdown>
          </div>
        </div>
      )}
    </div>
  );
}

function App() {
  const [messages, setMessages] = useState([]);
  const sessionId = useRef(newSessionId());
  const [input, setInput] = useState("");
  const [clarifying, setClarifying] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [suggestions, setSuggestions] = useState([]);
  const [theme, setTheme] = useState(() => localStorage.getItem("chat-theme") || "dark");
  const [lastUserText, setLastUserText] = useState(null);
  const chatBoxRef = useRef(null);
  const recognitionRef = useRef(null);
  const [isListening, setIsListening] = useState(false);
  const SpeechRecognition =
    typeof window !== "undefined" &&
    (window.SpeechRecognition || window.webkitSpeechRecognition);

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("chat-theme", theme);
  }, [theme]);

  useEffect(() => {
    if (chatBoxRef.current) {
      chatBoxRef.current.scrollTop = chatBoxRef.current.scrollHeight;
    }
  }, [messages, isLoading]);

  const fetchGrammarSuggestions = useCallback(async (text) => {
    const trimmed = text.trim();
    if (trimmed.length < 3 || TECHNICAL_HINTS.test(trimmed)) {
      setSuggestions([]);
      return;
    }
    try {
      const formData = new URLSearchParams();
      formData.append("text", trimmed);
      formData.append("language", "en-US");

      const response = await fetch(GRAMMAR_API_URL, {
        method: "POST",
        body: formData,
      });
      const data = await response.json();

      if (!data.matches || data.matches.length === 0) {
        setSuggestions([]);
        return;
      }

      // Apply each match independently to the ORIGINAL text (not to a
      // progressively-mutated one) so unrelated matches can't compound into
      // a corrupted suggestion.
      const seen = new Set([trimmed]);
      const uniqueSuggestions = [];

      data.matches.forEach((match, index) => {
        if (match.replacements.length === 0) return;
        const replacement = match.replacements[0].value;
        const corrected =
          trimmed.substring(0, match.offset) + replacement + trimmed.substring(match.offset + match.length);

        if (!seen.has(corrected)) {
          seen.add(corrected);
          uniqueSuggestions.push({ id: `suggestion_${index}`, correctedFull: corrected });
        }
      });

      setSuggestions(uniqueSuggestions.slice(0, 3));
    } catch (err) {
      console.error("Error fetching grammar suggestions:", err);
      setSuggestions([]);
    }
  }, []);

  const suggestionsTimeoutRef = useRef(null);
  const debouncedFetchSuggestions = useCallback((text) => {
    clearTimeout(suggestionsTimeoutRef.current);
    suggestionsTimeoutRef.current = setTimeout(() => {
      fetchGrammarSuggestions(text);
    }, 400);
  }, [fetchGrammarSuggestions]);

  const handleInputChange = (e) => {
    const newText = e.target.value;
    setInput(newText);
    debouncedFetchSuggestions(newText);
  };

  const handleSuggestionClick = (correctedFull) => {
    setInput(correctedFull);
    setSuggestions([]);
    document.querySelector(".chat-input")?.focus();
  };

  const startVoice = () => {
    if (!SpeechRecognition) return;
    if (isListening) return;
    const rec = new SpeechRecognition();
    rec.lang = "en-US";
    rec.interimResults = true;
    rec.onresult = (e) => {
      const transcript = Array.from(e.results).map((r) => r[0].transcript).join("");
      setInput(transcript);
      debouncedFetchSuggestions(transcript);
    };
    rec.onerror = (e) => console.error("Voice error:", e);
    rec.onend = () => setIsListening(false);
    rec.start();
    recognitionRef.current = rec;
    setIsListening(true);
  };

  const stopVoice = () => {
    recognitionRef.current?.stop();
    setIsListening(false);
  };

  const sendMessage = async (text, answers = null) => {
    setSuggestions([]);
    if (text && !answers) {
      setMessages((prev) => [...prev, { sender: "user", text }]);
      setLastUserText(text);
    }
    setIsLoading(true);
    if (!answers) {
      setClarifying([]);
    }

    try {
      const res = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionId.current,
          text: text || " ",
          answers,
        }),
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Backend error");

      if (data.type === "clarify") {
        setMessages((prev) => [
          ...prev,
          ...data.questions.map((q) => ({ sender: "bot", text: q.text, id: q.id })),
        ]);
        setClarifying(data.questions);
      } else if (data.type === "answer") {
        setMessages((prev) => [...prev, { sender: "bot", text: data.answer, meta: data.meta }]);
        setClarifying([]);
      }
    } catch (err) {
      console.error(err);
      setMessages((prev) => [...prev, { sender: "bot", text: `⚠️ ${err.message}`, isError: true }]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleRetry = () => {
    if (!lastUserText) return;
    setMessages((prev) => prev.filter((m) => !m.isError));
    sendMessage(lastUserText, null);
  };

  const handleNewChat = () => {
    setMessages([]);
    setClarifying([]);
    setInput("");
    setSuggestions([]);
    sessionId.current = newSessionId();
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    if (clarifying.length > 0) {
      const answersObj = {};
      clarifying.forEach((q) => {
        answersObj[q.id] = input;
      });
      setMessages((prev) => [...prev, { sender: "user", text: input }]);
      sendMessage(null, answersObj);
    } else {
      sendMessage(input, null);
    }
    setInput("");
  };

  return (
    <div className="app-shell">
      <div className="ambient-glow glow-1" />
      <div className="ambient-glow glow-2" />

      <div className="chat-container">
        <div className="chat-header">
          <div className="header-left">
            <div className="brand-mark">✦</div>
            <div>
              <h1 className="chat-title">Prompt Engineering Chatbot</h1>
              <p className="chat-subtitle">Groq + OpenRouter, auto-ranked in real time</p>
            </div>
          </div>
          <div className="header-actions">
            <button className="pill-button" onClick={handleNewChat} title="Start a new conversation">
              ＋ New chat
            </button>
            <button
              className="pill-button icon-only"
              onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
              title="Toggle theme"
              aria-label="Toggle theme"
            >
              {theme === "dark" ? "☀️" : "🌙"}
            </button>
          </div>
        </div>

        <div ref={chatBoxRef} className="chat-messages">
          {messages.length === 0 && (
            <div className="empty-state">
              <div className="empty-icon">💬</div>
              <h2>Start a conversation</h2>
              <p>Ask a general question, or a technical one to get a tailored, dual-model answer.</p>
              <div className="empty-hints">
                <span>"Explain React hooks like I'm new to JS"</span>
                <span>"Write a Python function to reverse a linked list"</span>
                <span>"What's a good prompt structure for summarization?"</span>
              </div>
            </div>
          )}
          {messages.map((msg, idx) => (
            <div key={idx} className={`message-wrapper ${msg.sender === "user" ? "user" : "bot"}`}>
              <div className={`message ${msg.sender} ${msg.isError ? "error" : ""}`}>
                <div className="message-content">
                  <ReactMarkdown>{msg.text}</ReactMarkdown>
                </div>
                {msg.sender === "bot" && !msg.isError && (
                  <>
                    <div className="message-footer">
                      {msg.meta && (
                        <span className="model-badge" title={`Groq ${msg.meta.scores.groq} · OpenRouter ${msg.meta.scores.openrouter}`}>
                          {MODEL_LABELS[msg.meta.winner] || msg.meta.model}
                        </span>
                      )}
                      <CopyButton text={msg.text} />
                    </div>
                    {msg.meta && <CompareAnswers alternates={msg.meta.alternates} winner={msg.meta.winner} />}
                  </>
                )}
                {msg.isError && lastUserText && (
                  <button className="retry-button" onClick={handleRetry}>↻ Retry</button>
                )}
              </div>
            </div>
          ))}
          {isLoading && (
            <div className="message-wrapper bot">
              <div className="message bot">
                <div className="typing-indicator"><span /><span /><span /></div>
              </div>
            </div>
          )}
        </div>

        <div className="chat-input-container">
          {suggestions.length > 0 && (
            <div className="suggestions-bar">
              <div className="suggestion-header">💡 Do you mean:</div>
              {suggestions.map((s) => (
                <div key={s.id} className="suggestion-item" onClick={() => handleSuggestionClick(s.correctedFull)}>
                  <div className="suggestion-text">{s.correctedFull}</div>
                </div>
              ))}
            </div>
          )}
          <form onSubmit={handleSubmit} className="chat-input-form">
            <input
              type="text"
              value={input}
              onChange={handleInputChange}
              placeholder={clarifying.length > 0 ? "Answer the questions above..." : "Type your message..."}
              disabled={isLoading}
              className="chat-input"
              autoComplete="off"
            />
            {SpeechRecognition && (
              <button
                type="button"
                aria-label="Voice input"
                className={`mic-button ${isListening ? "listening" : ""}`}
                onClick={isListening ? stopVoice : startVoice}
                title={isListening ? "Stop voice" : "Start voice"}
              >
                {isListening ? "🎙️" : "🎤"}
              </button>
            )}
            <button type="submit" disabled={!input.trim() || isLoading} className="send-button">
              Send ➤
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}

export default App;

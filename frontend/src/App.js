import { useState, useRef, useEffect, useCallback } from "react";
import "./App.css";

const API_URL = "http://127.0.0.1:8000/chat";
const GRAMMAR_API_URL = "https://api.languagetool.org/v2/check";

function App() {
  const [messages, setMessages] = useState([]);
  const sessionId = useRef("session_" + Date.now());
  const [input, setInput] = useState("");
  const [clarifying, setClarifying] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [suggestions, setSuggestions] = useState([]);
  const chatBoxRef = useRef(null);
  const recognitionRef = useRef(null);
  const [isListening, setIsListening] = useState(false);
  const SpeechRecognition =
    typeof window !== "undefined" &&
    (window.SpeechRecognition || window.webkitSpeechRecognition);

  useEffect(() => {
    if (chatBoxRef.current) {
      chatBoxRef.current.scrollTop = chatBoxRef.current.scrollHeight;
    }
  }, [messages]);

  const debounce = (func, delay) => {
    let timeoutId;
    return (...args) => {
      clearTimeout(timeoutId);
      timeoutId = setTimeout(() => {
        func.apply(null, args);
      }, delay);
    };
  };

  // --- THIS IS THE NEW, SMARTER SUGGESTION LOGIC ---
  const fetchGrammarSuggestions = async (text) => {
    const trimmed = text.trim();
    if (trimmed.length < 3) {
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
      
      // Create a function to apply a single correction to a string
      const applyCorrection = (str, match) => {
        const replacement = match.replacements[0].value;
        return str.substring(0, match.offset) + replacement + str.substring(match.offset + match.length);
      };

      // Generate a fully corrected sentence by applying all top suggestions
      let fullyCorrectedText = trimmed;
      // We apply corrections in reverse to not mess up the offsets of earlier errors
      [...data.matches].reverse().forEach(match => {
        if (match.replacements.length > 0) {
            fullyCorrectedText = applyCorrection(fullyCorrectedText, match);
        }
      });
      //add conditional functionality here -> 20
      const uniqueSuggestions = [];
      const seen = new Set();

      // 1. Add the fully corrected sentence if it's different
      if (fullyCorrectedText !== trimmed && !seen.has(fullyCorrectedText)) {
          uniqueSuggestions.push({
              id: 'full_correction',
              correctedFull: fullyCorrectedText
          });
          seen.add(fullyCorrectedText);
      }

      // 2. Add single, high-quality corrections
      data.matches.forEach((match, index) => {
          if (match.replacements.length > 0) {
              const singleCorrection = applyCorrection(trimmed, match);
              if (!seen.has(singleCorrection)) {
                   uniqueSuggestions.push({
                       id: `single_${index}`,
                       correctedFull: singleCorrection
                   });
                   seen.add(singleCorrection);
              }
          }
      });

      setSuggestions(uniqueSuggestions.slice(0, 3)); // Show up to 3 best suggestions

    } catch (err) {
      console.error("Error fetching grammar suggestions:", err);
      setSuggestions([]);
    }
  };

  const debouncedFetchSuggestions = useCallback(debounce(fetchGrammarSuggestions, 400), []);

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
      const transcript = Array.from(e.results).map(r => r[0].transcript).join('');
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
        setMessages((prev) => [...prev, { sender: "bot", text: data.answer }]);
        setClarifying([]);
        sessionId.current = "session_" + Date.now();
      }
    } catch (err) {
      console.error(err);
      setMessages((prev) => [...prev, { sender: "bot", text: `Error: ${err.message}` }]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    if (clarifying.length > 0) {
      const answersObj = {};
      clarifying.forEach(q => {
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
    <div className="chat-container">
      <div className="chat-header">
        <h1 className="chat-title">Prompt Engineering Chatbot</h1>
        <p className="chat-subtitle">Powered by AI • Ask me anything</p>
      </div>
      <div ref={chatBoxRef} className="chat-messages">
        {messages.length === 0 && (
          <div className="empty-state">
            <div className="empty-icon">💬</div>
            <h2>Start a conversation</h2>
            <p>Ask a general question, or a technical one to get a tailored explanation!</p>
          </div>
        )}
        {messages.map((msg, idx) => (
          <div key={idx} className={`message-wrapper ${msg.sender === "user" ? "user" : "bot"}`}>
            <div className={`message ${msg.sender}`}><p>{msg.text}</p></div>
          </div>
        ))}
        {isLoading && (
          <div className="message-wrapper bot">
            <div className="message bot"><div className="typing-indicator"><span /><span /><span /></div></div>
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
          <button
            type="button"
            aria-label="Voice input"
            className={`mic-button ${isListening ? "listening" : ""}`}
            onClick={isListening ? stopVoice : startVoice}
            title={isListening ? "Stop voice" : "Start voice"}
          >
            {isListening ? "🎙️" : "🎤"}
          </button>
          <button type="submit" disabled={!input.trim() || isLoading} className="send-button">Send ➤</button>
        </form>
      </div>
    </div>
  );
}

export default App;


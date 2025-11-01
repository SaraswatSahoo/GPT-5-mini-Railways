import React, { useState, useRef } from "react";
import "./App.css";

function App() {
  const [question, setQuestion] = useState("");
  const [chatHistory, setChatHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const recognitionRef = useRef(null);

  // ✅ Use Vite environment variables
  const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";
  const API_KEY = import.meta.env.VITE_API_KEY;

  // 🎤 Speech to text setup
  const handleMic = () => {
    if (!("webkitSpeechRecognition" in window)) {
      alert("Speech recognition not supported in this browser.");
      return;
    }

    if (!recognitionRef.current) {
      const SpeechRecognition =
        window.SpeechRecognition || window.webkitSpeechRecognition;
      recognitionRef.current = new SpeechRecognition();
      recognitionRef.current.lang = "hi-IN,en-US"; // Hindi + English
      recognitionRef.current.interimResults = false;
    }

    recognitionRef.current.onresult = (event) => {
      const transcript = event.results[0][0].transcript;
      setQuestion((prev) => prev + " " + transcript);
    };

    recognitionRef.current.start();
  };

  // 🔎 Ask backend
  const askQuestion = async () => {
    if (!question.trim()) return;
    setLoading(true);

    try {
      const headers = { "Content-Type": "application/json" };
      if (API_KEY) headers["Authorization"] = `Bearer ${API_KEY}`; // ✅ use your API key header

      const res = await fetch(`${API_URL}/ask`, {
        method: "POST",
        headers,
        body: JSON.stringify({ question: question, top_k: 5 }),
      });

      const data = await res.json();

      setChatHistory((prev) => [
        ...prev,
        {
          question,
          answer: data.answer || "⚠ No answer received",
        },
      ]);

      setQuestion("");
    } catch (err) {
      setChatHistory((prev) => [
        ...prev,
        { question, answer: "❌ Failed to fetch. Check if server is running." },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <div className="chat-box">
        <h1 className="title">GPT Integration 🤖</h1>

        {/* Chat history */}
        <div className="chat-history">
          {chatHistory.map((item, idx) => (
            <div key={idx} className="chat-message">
              <p className="question">
                <strong>Q:</strong> {item.question}
              </p>

              <div className="answer">
                <strong>A:</strong>{" "}
                {Array.isArray(item.answer) ? (
                  item.answer.map((ans, i) =>
                    ans.type === "image" ? (
                      <img
                        key={i}
                        src={ans.data}
                        alt={`answer-img-${i}`}
                        style={{
                          maxWidth: "300px",
                          margin: "10px 0",
                          borderRadius: "8px",
                        }}
                      />
                    ) : (
                      <p key={i}>{ans}</p>
                    )
                  )
                ) : (
                  <p>{item.answer}</p>
                )}
              </div>
            </div>
          ))}
          {loading && <p className="loading">⏳ Loading...</p>}
        </div>

        {/* Input + Buttons */}
        <div className="input-container">
          <textarea
            rows="3"
            className="question-input"
            placeholder="Type or speak your question..."
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
          />
          <div className="button-group">
            <button onClick={askQuestion} className="ask-button">
              Ask
            </button>
            <button onClick={handleMic} className="mic-button">
              🎤
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;

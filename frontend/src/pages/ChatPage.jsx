import React, { useEffect, useMemo, useRef, useState } from "react";
import { getChatResponse, verifyText } from "../services/api";
import "./ChatPage.css";

const STORAGE_KEY = "fact_checker_chats_v2";

const defaultChat = () => ({
  title: "New Session",
  messages: [],
  updatedAt: Date.now(),
});

const statusMap = {
  Supported: { className: "claim-supported", short: "Supported" },
  Refuted: { className: "claim-refuted", short: "Refuted" },
  "Not Enough Info": { className: "claim-nei", short: "Not Enough Info" },
};

function safeLoadChats() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) {
      return [defaultChat()];
    }
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed) || parsed.length === 0) {
      return [defaultChat()];
    }
    return parsed;
  } catch {
    return [defaultChat()];
  }
}

function shortTitle(text) {
  const normalized = text.replace(/\s+/g, " ").trim();
  if (!normalized) {
    return "Untitled chat";
  }
  const words = normalized.split(" ").slice(0, 6).join(" ");
  return words.length > 40 ? `${words.slice(0, 37)}...` : words;
}

function formatTime(value) {
  return new Date(value).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function truthClass(score) {
  if (score == null) {
    return "truth-none";
  }
  if (score >= 70) {
    return "truth-high";
  }
  if (score >= 40) {
    return "truth-medium";
  }
  return "truth-low";
}

const ChatPage = () => {
  const [view, setView] = useState("chat");
  const [chats, setChats] = useState(() => safeLoadChats());
  const [currentChat, setCurrentChat] = useState(0);
  const [draft, setDraft] = useState("");
  const [detectDraft, setDetectDraft] = useState("");
  const [loading, setLoading] = useState(false);
  const [selectedResponse, setSelectedResponse] = useState(null);
  const [isPanelOpen, setIsPanelOpen] = useState(false);

  const messagesEndRef = useRef(null);
  const activeChat = chats[currentChat] || chats[0];

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(chats));
  }, [chats]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [activeChat?.messages?.length, currentChat]);

  const summary = useMemo(() => {
    if (!selectedResponse?.verification?.length) {
      return { supported: 0, refuted: 0, nei: 0 };
    }
    return selectedResponse.verification.reduce(
      (acc, claim) => {
        if (claim.label === "Supported") acc.supported += 1;
        if (claim.label === "Refuted") acc.refuted += 1;
        if (claim.label === "Not Enough Info") acc.nei += 1;
        return acc;
      },
      { supported: 0, refuted: 0, nei: 0 },
    );
  }, [selectedResponse]);

  const createChat = () => {
    const next = [...chats, defaultChat()];
    setChats(next);
    setCurrentChat(next.length - 1);
    setSelectedResponse(null);
  };

  const removeChat = (index) => {
    const next = chats.filter((_, i) => i !== index);
    if (!next.length) {
      setChats([defaultChat()]);
      setCurrentChat(0);
      return;
    }
    setChats(next);
    if (index === currentChat) {
      setCurrentChat(0);
    } else if (index < currentChat) {
      setCurrentChat(currentChat - 1);
    }
  };

  const appendUserMessage = (text) => {
    const updated = [...chats];
    const target = { ...updated[currentChat] };
    const nextMessages = [
      ...target.messages,
      { role: "user", content: text, createdAt: Date.now() },
    ];
    target.messages = nextMessages;
    target.updatedAt = Date.now();
    if (nextMessages.length === 1) {
      target.title = shortTitle(text);
    }
    updated[currentChat] = target;
    setChats(updated);
    return updated;
  };

  const sendMessage = async () => {
    const userText = draft.trim();
    if (!userText || loading || !activeChat) {
      return;
    }

    setLoading(true);
    setDraft("");
    const updated = appendUserMessage(userText);

    try {
      const chatResult = await getChatResponse(userText);
      const createdAt = Date.now();
      const aiMessage = {
        role: "ai",
        createdAt,
        content: {
          response: chatResult.response,
          verification: [],
          truthScore: null,
          verifying: true,
        },
      };

      const withAi = [...updated];
      withAi[currentChat] = {
        ...withAi[currentChat],
        messages: [...withAi[currentChat].messages, aiMessage],
        updatedAt: Date.now(),
      };
      setChats(withAi);
      setLoading(false);

      try {
        const verificationResult = await verifyText(chatResult.response);
        setChats((prevChats) => {
          const next = [...prevChats];
          const active = next[currentChat];
          if (!active) {
            return prevChats;
          }
          const patchedMessages = active.messages.map((message) => {
            if (message.createdAt !== createdAt || message.role !== "ai") {
              return message;
            }
            return {
              ...message,
              content: {
                ...message.content,
                verification: verificationResult.verification || [],
                truthScore: verificationResult.truth_score ?? null,
                verifying: false,
              },
            };
          });
          next[currentChat] = {
            ...active,
            messages: patchedMessages,
            updatedAt: Date.now(),
          };
          return next;
        });
      } catch {
        setChats((prevChats) => {
          const next = [...prevChats];
          const active = next[currentChat];
          if (!active) {
            return prevChats;
          }
          const patchedMessages = active.messages.map((message) => {
            if (message.createdAt !== createdAt || message.role !== "ai") {
              return message;
            }
            return {
              ...message,
              content: {
                ...message.content,
                verifying: false,
              },
            };
          });
          next[currentChat] = {
            ...active,
            messages: patchedMessages,
            updatedAt: Date.now(),
          };
          return next;
        });
      }
    } catch {
      const failed = {
        role: "ai",
        createdAt: Date.now(),
        content: {
          response: "I could not complete that request. Please try again.",
          verification: [],
          truthScore: null,
          verifying: false,
        },
      };
      const withFailure = [...updated];
      withFailure[currentChat] = {
        ...withFailure[currentChat],
        messages: [...withFailure[currentChat].messages, failed],
        updatedAt: Date.now(),
      };
      setChats(withFailure);
    } finally {
      setLoading(false);
    }
  };

  const runDetection = async () => {
    const text = detectDraft.trim();
    if (!text || loading) {
      return;
    }

    setLoading(true);
    try {
      const result = await verifyText(text);
      const chat = {
        title: `Detection: ${shortTitle(text)}`,
        updatedAt: Date.now(),
        messages: [
          {
            role: "ai",
            createdAt: Date.now(),
            content: {
              response: text,
              verification: result.verification || [],
              truthScore: result.truth_score ?? null,
              verifying: false,
            },
          },
        ],
      };
      const next = [...chats, chat];
      setChats(next);
      setCurrentChat(next.length - 1);
      setView("chat");
      setDetectDraft("");
    } finally {
      setLoading(false);
    }
  };

  const openDetails = (content) => {
    if (!content?.verification?.length) {
      return;
    }
    setSelectedResponse(content);
    setIsPanelOpen(true);
  };

  return (
    <div className="fact-ui">
      <aside className="fact-rail">
        <button
          type="button"
          className={`rail-btn ${view === "chat" ? "active" : ""}`}
          onClick={() => setView("chat")}
          title="Chat"
        >
          Chat
        </button>
        <button
          type="button"
          className={`rail-btn ${view === "detect" ? "active" : ""}`}
          onClick={() => setView("detect")}
          title="Detector"
        >
          Detect
        </button>
      </aside>

      {view === "chat" && (
        <>
          <section className="chat-list">
            <header className="chat-list-header">
              <div>
                <h2>Conversations</h2>
                <p>Track answers and verification details</p>
              </div>
              <button type="button" className="primary-btn" onClick={createChat}>
                New Chat
              </button>
            </header>

            <div className="chat-items">
              {chats.map((chat, idx) => (
                <article
                  key={`${chat.title}-${idx}`}
                  className={`chat-item ${idx === currentChat ? "selected" : ""}`}
                  onClick={() => {
                    setCurrentChat(idx);
                    setSelectedResponse(null);
                  }}
                >
                  <div className="chat-item-main">
                    <h3>{chat.title}</h3>
                    <small>{chat.messages.length} messages</small>
                  </div>
                  <button
                    type="button"
                    className="ghost-btn"
                    onClick={(event) => {
                      event.stopPropagation();
                      removeChat(idx);
                    }}
                  >
                    Delete
                  </button>
                </article>
              ))}
            </div>
          </section>

          <main className="chat-main">
            <header className="chat-main-header">
              <div>
                <h1>{activeChat?.title || "Conversation"}</h1>
                <p>Click an assistant response to inspect claim-level verification.</p>
              </div>
            </header>

            <section className="messages-area">
              {activeChat?.messages?.length ? (
                activeChat.messages.map((msg, idx) => {
                  const isAI = msg.role === "ai";
                  const content = isAI ? msg.content : null;
                  const hasClaims = !!content?.verification?.length;
                  const score = content?.truthScore;
                  return (
                    <article key={`${msg.createdAt || idx}-${idx}`} className={`bubble-row ${isAI ? "ai" : "user"}`}>
                      <div
                        className={`bubble ${isAI ? "bubble-ai" : "bubble-user"} ${hasClaims ? "clickable" : ""}`}
                        onClick={() => isAI && openDetails(content)}
                      >
                        <p>{isAI ? content.response : msg.content}</p>
                        {isAI && (
                          <div className="bubble-meta">
                            <span>{formatTime(msg.createdAt || Date.now())}</span>
                            {typeof score === "number" && (
                              <span className={`truth-pill ${truthClass(score)}`}>Truth {score}%</span>
                            )}
                            {content?.verifying && <span className="claims-pill">Verifying...</span>}
                            {hasClaims && <span className="claims-pill">{content.verification.length} claims</span>}
                          </div>
                        )}
                      </div>
                    </article>
                  );
                })
              ) : (
                <div className="empty-chat">
                  <h3>Start a conversation</h3>
                  <p>Ask anything. Every assistant answer is automatically verified against retrieved evidence.</p>
                </div>
              )}

              {loading && (
                <article className="bubble-row ai">
                  <div className="bubble bubble-ai typing">Generating answer...</div>
                </article>
              )}
              <div ref={messagesEndRef} />
            </section>

            <footer className="chat-input-area">
              <textarea
                placeholder="Ask a factual question"
                value={draft}
                onChange={(event) => setDraft(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" && !event.shiftKey) {
                    event.preventDefault();
                    sendMessage();
                  }
                }}
                disabled={loading}
              />
              <button type="button" className="primary-btn" onClick={sendMessage} disabled={loading || !draft.trim()}>
                Send
              </button>
            </footer>
          </main>
        </>
      )}

      {view === "detect" && (
        <section className="detector-page">
          <div className="detector-card">
            <h2>Standalone Hallucination Detector</h2>
            <p>Paste model output to run claim extraction and verification without starting a chat.</p>
            <textarea
              placeholder="Paste generated text for verification"
              value={detectDraft}
              onChange={(event) => setDetectDraft(event.target.value)}
            />
            <button
              type="button"
              className="primary-btn"
              onClick={runDetection}
              disabled={loading || !detectDraft.trim()}
            >
              Analyze Text
            </button>
          </div>
        </section>
      )}

      {isPanelOpen && <div className="overlay" onClick={() => setIsPanelOpen(false)} />}

      <aside className={`inspector ${isPanelOpen ? "open" : ""}`}>
        <header className="inspector-header">
          <h3>Verification Details</h3>
          <button type="button" className="ghost-btn" onClick={() => setIsPanelOpen(false)}>
            Close
          </button>
        </header>

        {selectedResponse ? (
          <div className="inspector-content">
            <div className="score-card">
              <h4>Overview</h4>
              <p className={`truth-pill ${truthClass(selectedResponse.truthScore)}`}>
                Truth score: {selectedResponse.truthScore ?? 0}%
              </p>
              <div className="counts-grid">
                <span>Supported: {summary.supported}</span>
                <span>Refuted: {summary.refuted}</span>
                <span>Not enough info: {summary.nei}</span>
              </div>
            </div>

            {(selectedResponse.verification || []).map((claim, index) => {
              const status = statusMap[claim.label] || { className: "claim-default", short: claim.label || "Unknown" };
              return (
                <article key={`${claim.claim}-${index}`} className={`claim-card ${status.className}`}>
                  <div className="claim-head">
                    <h5>Claim {index + 1}</h5>
                    <span className="status-chip">{status.short}</span>
                  </div>
                  <p className="claim-text">{claim.claim}</p>
                  {claim.rationale && <p className="claim-rationale">{claim.rationale}</p>}

                  {!!claim.evidence?.length && (
                    <div className="evidence-list">
                      {claim.evidence.map((item, evidenceIndex) => (
                        <div key={`${evidenceIndex}-${item.sentence}`} className="evidence-item">
                          <p>{item.sentence}</p>
                          <small>
                            Source: {item.page || "Unknown"} | Similarity: {item.score ?? "n/a"}
                          </small>
                        </div>
                      ))}
                    </div>
                  )}
                </article>
              );
            })}
          </div>
        ) : (
          <div className="inspector-empty">
            Open any assistant message with claims to inspect evidence, semantic similarity, and verdict rationale.
          </div>
        )}
      </aside>
    </div>
  );
};

export default ChatPage;

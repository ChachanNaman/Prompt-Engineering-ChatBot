"use client";

import { Mic, Plus, Sparkles } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import {
  ApprovalCard,
  type ApprovalCardAnswers,
  type ApprovalCardQuestion,
  type ApprovalCardStatus,
} from "@/components/agents/approval-card";
import {
  Message,
  MessageAvatar,
  MessageContent,
  MessageFooter,
  MessageGroup,
  MessageHeader,
} from "@/components/agents/message";
import {
  MessageBubble,
  MessageBubbleContent,
} from "@/components/agents/message-bubble";
import { MessageScroller } from "@/components/agents/message-scroller";
import { PromptInput } from "@/components/agents/prompt-input";
import { ReasoningText } from "@/components/agents/loading-states/reasoning-text";
import { StreamingResponse } from "@/components/agents/streaming-response";
import { Markdown } from "@/components/markdown";
import { ThemeToggle } from "@/components/theme-toggle";
import {
  type ChatMeta,
  type ClarifyQuestion,
  MODEL_LABELS,
  newSessionId,
  postChat,
} from "@/lib/chat-api";
import { cn } from "@/lib/utils";

const GRAMMAR_API_URL = "https://api.languagetool.org/v2/check";
const TECHNICAL_HINTS =
  /\b(code|python|javascript|react|fastapi|typescript|sql|api|algorithm|function|debug|css|html|java|docker|git|regex|c\+\+|c#|array|linked list|tree|graph|stack|queue)\b/i;

type UserMsg = { id: string; role: "user"; text: string };
type AssistantMsg = {
  id: string;
  role: "assistant";
  full: string;
  text: string;
  streaming: boolean;
  error?: boolean;
  meta?: ChatMeta | null;
};
type ClarifyMsg = {
  id: string;
  role: "clarify";
  intro?: string;
  questions: ClarifyQuestion[];
  status: ApprovalCardStatus;
  summary?: string;
};
type ChatMsg = UserMsg | AssistantMsg | ClarifyMsg;

const THINKING_PHRASES = [
  "Thinking",
  "Consulting both models",
  "Ranking the answers",
  "Composing",
];

const STARTERS = [
  "Explain React hooks like I'm new to JS",
  "Reverse a linked list in Python",
  "Time & space complexity of binary search",
  "A good prompt structure for summarization",
];

// Minimal shape of the Web Speech API we rely on.
interface SpeechRecognitionLike {
  lang: string;
  interimResults: boolean;
  continuous: boolean;
  onresult: (event: {
    results: ArrayLike<ArrayLike<{ transcript: string }>>;
  }) => void;
  onerror: (event: unknown) => void;
  onend: () => void;
  start: () => void;
  stop: () => void;
}

export function Chat() {
  const [messages, setMessages] = useState<ChatMsg[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [streamingId, setStreamingId] = useState<string | null>(null);
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [isListening, setIsListening] = useState(false);
  const [micSupported, setMicSupported] = useState(false);

  const sessionId = useRef(newSessionId());
  const idCounter = useRef(0);
  const abortRef = useRef<AbortController | null>(null);
  const recognitionRef = useRef<SpeechRecognitionLike | null>(null);
  const suggestTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const nextId = () => `m${idCounter.current++}`;

  useEffect(() => {
    const w = window as unknown as {
      SpeechRecognition?: new () => SpeechRecognitionLike;
      webkitSpeechRecognition?: new () => SpeechRecognitionLike;
    };
    setMicSupported(Boolean(w.SpeechRecognition || w.webkitSpeechRecognition));
  }, []);

  // Typewriter reveal for the streaming assistant message.
  useEffect(() => {
    if (!streamingId) return;
    let frame = 0;
    const startedAt = performance.now();
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    const target = () =>
      (messages.find((m) => m.id === streamingId) as AssistantMsg | undefined)
        ?.full ?? "";

    if (reduce) {
      setMessages((prev) =>
        prev.map((m) =>
          m.id === streamingId && m.role === "assistant"
            ? { ...m, text: m.full, streaming: false }
            : m,
        ),
      );
      setStreamingId(null);
      return;
    }

    const step = (now: number) => {
      const full = target();
      const chars = Math.min(full.length, Math.floor((now - startedAt) / 3.2));
      setMessages((prev) =>
        prev.map((m) =>
          m.id === streamingId && m.role === "assistant"
            ? { ...m, text: full.slice(0, chars) }
            : m,
        ),
      );
      if (chars < full.length) {
        frame = requestAnimationFrame(step);
      } else {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === streamingId && m.role === "assistant"
              ? { ...m, streaming: false }
              : m,
          ),
        );
        setStreamingId(null);
      }
    };
    frame = requestAnimationFrame(step);
    return () => cancelAnimationFrame(frame);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [streamingId]);

  const fetchSuggestions = useCallback(async (text: string) => {
    const trimmed = text.trim();
    if (trimmed.length < 4 || TECHNICAL_HINTS.test(trimmed)) {
      setSuggestions([]);
      return;
    }
    try {
      const body = new URLSearchParams();
      body.append("text", trimmed);
      body.append("language", "en-US");
      const res = await fetch(GRAMMAR_API_URL, { method: "POST", body });
      const data = await res.json();
      if (!data.matches?.length) {
        setSuggestions([]);
        return;
      }
      const seen = new Set([trimmed]);
      const out: string[] = [];
      for (const match of data.matches) {
        if (!match.replacements?.length) continue;
        const corrected =
          trimmed.slice(0, match.offset) +
          match.replacements[0].value +
          trimmed.slice(match.offset + match.length);
        if (!seen.has(corrected)) {
          seen.add(corrected);
          out.push(corrected);
        }
      }
      setSuggestions(out.slice(0, 3));
    } catch {
      setSuggestions([]);
    }
  }, []);

  const onInputChange = (value: string) => {
    setInput(value);
    if (suggestTimer.current) clearTimeout(suggestTimer.current);
    suggestTimer.current = setTimeout(() => fetchSuggestions(value), 450);
  };

  const runRequest = useCallback(
    async (text: string, answers: Record<string, string> | null) => {
      setLoading(true);
      abortRef.current?.abort();
      abortRef.current = new AbortController();
      try {
        const data = await postChat({
          session_id: sessionId.current,
          text: text || " ",
          answers,
        });
        if (data.type === "clarify") {
          setMessages((prev) => [
            ...prev,
            {
              id: nextId(),
              role: "clarify",
              intro: data.intro,
              questions: data.questions,
              status: "pending",
            },
          ]);
        } else {
          const id = nextId();
          setMessages((prev) => [
            ...prev,
            {
              id,
              role: "assistant",
              full: data.answer,
              text: "",
              streaming: true,
              meta: data.meta,
            },
          ]);
          setStreamingId(id);
        }
      } catch (err) {
        setMessages((prev) => [
          ...prev,
          {
            id: nextId(),
            role: "assistant",
            full: "",
            text: `Something went wrong. ${
              err instanceof Error ? err.message : "Please try again."
            }`,
            streaming: false,
            error: true,
          },
        ]);
      } finally {
        setLoading(false);
      }
    },
    [],
  );

  const send = (value: string) => {
    const text = value.trim();
    if (!text || loading) return;
    setSuggestions([]);
    setMessages((prev) => [...prev, { id: nextId(), role: "user", text }]);
    setInput("");
    void runRequest(text, null);
  };

  const submitClarify = (msg: ClarifyMsg, answers: ApprovalCardAnswers) => {
    const resolved: Record<string, string> = {};
    const labels: string[] = [];
    for (const q of msg.questions) {
      const answer = answers[q.id];
      const value = answer?.selected?.[0] ?? answer?.custom?.trim() ?? "";
      resolved[q.id] = value;
      const label =
        q.options?.find((o) => o.value === value)?.label ?? value;
      if (label) labels.push(label);
    }
    setMessages((prev) =>
      prev.map((m) =>
        m.id === msg.id
          ? { ...m, status: "answered", summary: labels.join(" · ") }
          : m,
      ),
    );
    void runRequest("", resolved);
  };

  const stop = () => {
    abortRef.current?.abort();
    setLoading(false);
    setStreamingId(null);
    setMessages((prev) =>
      prev.map((m) =>
        m.role === "assistant" && m.streaming
          ? { ...m, streaming: false }
          : m,
      ),
    );
  };

  const newChat = () => {
    stop();
    setMessages([]);
    setInput("");
    setSuggestions([]);
    sessionId.current = newSessionId();
  };

  const startVoice = () => {
    const w = window as unknown as {
      SpeechRecognition?: new () => SpeechRecognitionLike;
      webkitSpeechRecognition?: new () => SpeechRecognitionLike;
    };
    const Ctor = w.SpeechRecognition || w.webkitSpeechRecognition;
    if (!Ctor || isListening) return;
    const rec = new Ctor();
    rec.lang = "en-US";
    rec.interimResults = true;
    rec.continuous = false;
    rec.onresult = (event) => {
      const transcript = Array.from(event.results)
        .map((r) => r[0].transcript)
        .join("");
      setInput(transcript);
    };
    rec.onerror = () => setIsListening(false);
    rec.onend = () => setIsListening(false);
    rec.start();
    recognitionRef.current = rec;
    setIsListening(true);
  };

  const stopVoice = () => {
    recognitionRef.current?.stop();
    setIsListening(false);
  };

  const empty = messages.length === 0;

  return (
    <div className="flex h-dvh flex-col bg-background">
      <header className="flex h-14 shrink-0 items-center justify-between border-b border-border/70 px-4 sm:px-6">
        <div className="flex min-w-0 items-center gap-3">
          <div className="grid size-8 shrink-0 place-items-center rounded-xl bg-foreground text-background">
            <Sparkles className="size-4" />
          </div>
          <div className="min-w-0">
            <p className="truncate text-sm font-medium">
              Prompt Engineering Chatbot
            </p>
            <p className="truncate text-xs text-muted-foreground">
              Groq + OpenRouter, auto-ranked in real time
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={newChat}
            className="inline-flex h-9 items-center gap-1.5 rounded-full border border-border/70 px-3 text-sm text-muted-foreground outline-none transition-colors hover:bg-muted hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring active:scale-[0.98] [transition:transform_140ms_ease-out,background-color_140ms_ease,color_140ms_ease]"
          >
            <Plus className="size-4" />
            <span className="hidden sm:inline">New chat</span>
          </button>
          <ThemeToggle />
        </div>
      </header>

      <MessageScroller
        busy={loading}
        className="min-h-0 flex-1"
        viewportClassName="px-3 py-6 sm:px-6"
        contentClassName="mx-auto flex min-h-full w-full max-w-3xl flex-col"
      >
        {empty ? (
          <EmptyState onPick={send} />
        ) : (
          <MessageGroup spacing="default">
            {messages.map((msg) =>
              msg.role === "user" ? (
                <Message key={msg.id} from="user" animateIn>
                  <MessageContent>
                    <MessageBubble variant="solid">
                      <MessageBubbleContent>{msg.text}</MessageBubbleContent>
                    </MessageBubble>
                  </MessageContent>
                </Message>
              ) : msg.role === "clarify" ? (
                <Message key={msg.id} from="assistant" animateIn>
                  <MessageAvatar>
                    <Sparkles className="size-4" />
                  </MessageAvatar>
                  <MessageContent>
                    <ApprovalCard
                      title="Let's tailor this"
                      status={msg.status}
                      questions={msg.questions.map(
                        (q): ApprovalCardQuestion => ({
                          id: q.id,
                          title: q.text,
                          description:
                            q.id === msg.questions[0].id ? msg.intro : undefined,
                          options: q.options?.map((o) => ({
                            value: o.value,
                            label: o.label,
                          })),
                          allowCustom: true,
                          customPlaceholder: "Or type your own…",
                        }),
                      )}
                      onSubmit={(answers) => submitClarify(msg, answers)}
                      submitLabel="Generate answer"
                      result={
                        msg.summary
                          ? `Tailored for: ${msg.summary}`
                          : "Answer on the way."
                      }
                    />
                  </MessageContent>
                </Message>
              ) : (
                <Message key={msg.id} from="assistant" animateIn>
                  <MessageAvatar>
                    <Sparkles className="size-4" />
                  </MessageAvatar>
                  <MessageContent className="gap-1.5">
                    {msg.meta ? (
                      <MessageHeader>
                        <span>{MODEL_LABELS[msg.meta.winner] ?? msg.meta.model}</span>
                        <span className="text-muted-foreground/70">
                          auto-ranked
                        </span>
                      </MessageHeader>
                    ) : null}
                    {msg.error ? (
                      <MessageBubble variant="danger">
                        <MessageBubbleContent>{msg.text}</MessageBubbleContent>
                      </MessageBubble>
                    ) : (
                      <MessageBubble variant="ghost" className="w-full">
                        <MessageBubbleContent>
                          <StreamingResponse
                            status={msg.streaming ? "streaming" : "complete"}
                            copyText={msg.full}
                            showActions={!msg.streaming}
                          >
                            {msg.streaming ? (
                              <div className="whitespace-pre-wrap">{msg.text}</div>
                            ) : (
                              <Markdown>{msg.full}</Markdown>
                            )}
                          </StreamingResponse>
                        </MessageBubbleContent>
                      </MessageBubble>
                    )}
                    {!msg.streaming && !msg.error && msg.meta ? (
                      <CompareAnswers meta={msg.meta} />
                    ) : null}
                  </MessageContent>
                </Message>
              ),
            )}
            {loading && !streamingId ? (
              <Message from="assistant" animateIn>
                <MessageAvatar>
                  <Sparkles className="size-4" />
                </MessageAvatar>
                <MessageContent>
                  <ReasoningText variant="scramble" phrases={THINKING_PHRASES} />
                </MessageContent>
              </Message>
            ) : null}
          </MessageGroup>
        )}
      </MessageScroller>

      <div className="shrink-0 border-t border-border/70 bg-background px-3 pb-4 pt-3 sm:px-6">
        <div className="mx-auto w-full max-w-3xl">
          {suggestions.length > 0 ? (
            <div className="mb-2 flex flex-wrap items-center gap-2 text-xs">
              <span className="text-muted-foreground">Did you mean:</span>
              {suggestions.map((s) => (
                <button
                  key={s}
                  type="button"
                  onClick={() => {
                    setInput(s);
                    setSuggestions([]);
                  }}
                  className="rounded-full border border-border/70 px-2.5 py-1 text-foreground outline-none transition-colors hover:bg-muted focus-visible:ring-2 focus-visible:ring-ring"
                >
                  {s}
                </button>
              ))}
            </div>
          ) : null}
          <PromptInput
            value={input}
            onValueChange={onInputChange}
            loading={loading}
            onStop={stop}
            onSubmit={(value) => send(value)}
            minRows={1}
            maxRows={6}
            placeholder="Type your message…"
            leadingAction={
              micSupported ? (
                <button
                  type="button"
                  aria-label={isListening ? "Stop voice input" : "Start voice input"}
                  aria-pressed={isListening}
                  onClick={isListening ? stopVoice : startVoice}
                  className={cn(
                    "grid size-8 place-items-center rounded-full text-muted-foreground outline-none transition-colors hover:bg-muted hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring active:scale-95 [transition:transform_140ms_ease-out,background-color_140ms_ease,color_140ms_ease]",
                    isListening &&
                      "bg-red-500/15 text-red-500 hover:bg-red-500/20 hover:text-red-500",
                  )}
                >
                  <Mic className={cn("size-4", isListening && "animate-pulse")} />
                </button>
              ) : null
            }
          />
          <p className="mt-2 text-center text-[11px] text-muted-foreground/70">
            Technical questions get tailored answers — pick your use case and skill
            level when asked.
          </p>
        </div>
      </div>
    </div>
  );
}

function EmptyState({ onPick }: { onPick: (value: string) => void }) {
  return (
    <div className="flex flex-1 flex-col items-center justify-center px-4 text-center">
      <div className="mb-5 grid size-14 place-items-center rounded-2xl bg-muted">
        <Sparkles className="size-6 text-foreground" />
      </div>
      <h2 className="text-xl font-medium tracking-tight">
        Ask anything, get the better answer
      </h2>
      <p className="mt-2 max-w-md text-pretty text-sm text-muted-foreground">
        Two models answer in parallel and the stronger reply is picked for you.
        Ask a technical question and it&apos;ll tailor the depth to you.
      </p>
      <div className="mt-6 grid w-full max-w-lg grid-cols-1 gap-2 sm:grid-cols-2">
        {STARTERS.map((s) => (
          <button
            key={s}
            type="button"
            onClick={() => onPick(s)}
            className="rounded-xl border border-border/70 px-3.5 py-3 text-left text-sm text-foreground outline-none transition-colors hover:bg-muted focus-visible:ring-2 focus-visible:ring-ring active:scale-[0.99] [transition:transform_140ms_ease-out,background-color_140ms_ease]"
          >
            {s}
          </button>
        ))}
      </div>
    </div>
  );
}

function CompareAnswers({ meta }: { meta: ChatMeta }) {
  const [open, setOpen] = useState(false);
  const other = meta.winner === "groq" ? "openrouter" : "groq";
  const otherText = meta.alternates?.[other];
  if (!otherText) return null;
  const otherOk = meta.ok ? meta.ok[other] : true;

  return (
    <div className="mt-0.5">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="inline-flex items-center gap-1 rounded-md px-1.5 py-1 text-xs text-muted-foreground outline-none transition-colors hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring"
      >
        {open ? "Hide" : "Compare"} {MODEL_LABELS[other]}&apos;s answer
      </button>
      {open ? (
        <div className="mt-2 rounded-xl border border-border/70 bg-muted/40 p-3">
          <div className="mb-2 flex items-center gap-2">
            <span className="rounded-full bg-muted px-2 py-0.5 text-[11px] font-medium text-muted-foreground">
              {MODEL_LABELS[other]}
            </span>
            {!otherOk ? (
              <span className="text-[11px] text-amber-600 dark:text-amber-400">
                did not return a usable answer
              </span>
            ) : null}
          </div>
          <div className="text-sm leading-6 text-foreground/90 [&_code]:rounded [&_code]:bg-muted [&_code]:px-1 [&_code]:py-0.5 [&_pre]:my-2 [&_ul]:my-2 [&_ul]:list-disc [&_ul]:pl-5 [&_ol]:my-2 [&_ol]:list-decimal [&_ol]:pl-5 [&_p+p]:mt-2">
            <Markdown>{otherText}</Markdown>
          </div>
        </div>
      ) : null}
    </div>
  );
}

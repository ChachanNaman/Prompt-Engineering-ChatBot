import type { AgentCodeLanguage } from "@/components/agents/agent-code";

export const API_BASE = (
  process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000"
).replace(/\/$/, "");

export interface ClarifyOption {
  value: string;
  label: string;
}

export interface ClarifyQuestion {
  id: string;
  text: string;
  options?: ClarifyOption[];
}

export interface ChatMeta {
  winner: "groq" | "openrouter";
  model: string;
  scores: { groq: number; openrouter: number };
  ok?: { groq: boolean; openrouter: boolean };
  alternates?: { groq: string; openrouter: string };
}

export type ChatResponse =
  | { type: "clarify"; intro?: string; questions: ClarifyQuestion[] }
  | { type: "answer"; answer: string; meta: ChatMeta | null };

export interface ChatPayload {
  session_id: string;
  text: string;
  answers?: Record<string, string> | null;
}

export async function postChat(payload: ChatPayload): Promise<ChatResponse> {
  const res = await fetch(`${API_BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await res.json().catch(() => null);
  if (!res.ok) {
    throw new Error((data && data.detail) || `Request failed (${res.status})`);
  }
  return data as ChatResponse;
}

export function newSessionId(): string {
  return `session_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
}

export const MODEL_LABELS: Record<string, string> = {
  groq: "Groq · GPT-OSS 20B",
  openrouter: "OpenRouter · GPT-3.5",
};

// The bundled highlighter only knows a fixed set of grammars; map common
// Markdown fence languages onto them and fall back to plain "text".
const LANGUAGE_MAP: Record<string, AgentCodeLanguage> = {
  ts: "typescript",
  typescript: "typescript",
  tsx: "tsx",
  js: "javascript",
  javascript: "javascript",
  jsx: "jsx",
  py: "python",
  python: "python",
  java: "java",
  "c++": "cpp",
  cpp: "cpp",
  cc: "cpp",
  c: "c",
  "c#": "csharp",
  cs: "csharp",
  csharp: "csharp",
  go: "go",
  golang: "go",
  rust: "rust",
  rs: "rust",
  rb: "ruby",
  ruby: "ruby",
  php: "php",
  sql: "sql",
  css: "css",
  html: "html",
  xml: "html",
  yaml: "yaml",
  yml: "yaml",
  json: "json",
  md: "markdown",
  markdown: "markdown",
  sh: "bash",
  bash: "bash",
  shell: "bash",
  zsh: "bash",
  diff: "diff",
};

export function mapLanguage(lang?: string): AgentCodeLanguage {
  if (!lang) return "text";
  return LANGUAGE_MAP[lang.toLowerCase()] ?? "text";
}

import os
import re
import asyncio
import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()
app = FastAPI(title="Prompt Engineering ChatBot API")

# --- CORS ---
# ALLOWED_ORIGINS is a comma-separated list, e.g. "http://localhost:3000,https://myapp.vercel.app"
_default_origins = "http://localhost:3000,http://127.0.0.1:3000"
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", _default_origins).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API Keys and Session Storage ---
# .strip() guards against trailing newlines/whitespace from how some hosting
# dashboards store pasted env var values — a stray "\n" in an Authorization
# header makes httpx raise "Illegal header value" and silently breaks the call.
GROQ_API_KEY = (os.getenv("GROQ_API_KEY") or "").strip() or None
OPENROUTER_API_KEY = (os.getenv("OPENROUTER_API_KEY") or "").strip() or None

SESSIONS: dict = {}  # In-memory session storage: what the user gave and got


# --- Pydantic Models ---
class ChatRequest(BaseModel):
    session_id: str
    text: str
    answers: dict | None = None


# --- Technical question detection ---
# Single-word signals matched against the tokenized message (whole words only,
# so "art" never matches inside "start").
_TECH_WORDS = {
    # languages / runtimes / tools
    "python", "javascript", "typescript", "java", "kotlin", "swift", "rust",
    "golang", "php", "ruby", "scala", "react", "reactjs", "nextjs", "vue",
    "angular", "svelte", "node", "nodejs", "fastapi", "django", "flask",
    "spring", "express", "sql", "nosql", "mongodb", "postgres", "postgresql",
    "mysql", "redis", "docker", "kubernetes", "k8s", "git", "regex", "css",
    "html", "api", "rest", "graphql", "http", "json", "linux", "bash",
    "numpy", "pandas", "pytorch", "tensorflow", "sklearn",
    # general programming vocabulary
    "code", "coding", "program", "programming", "function", "algorithm",
    "algorithms", "debug", "debugging", "compile", "runtime", "syntax",
    "variable", "loop", "recursion", "recursive", "pointer", "memory",
    "async", "await", "thread", "threading", "concurrency", "class",
    "interface", "inheritance", "polymorphism", "closure", "callback",
    "middleware", "endpoint", "deploy", "deployment", "database", "query",
    "schema", "cache", "caching", "compiler", "bug", "exception",
    # data structures & algorithms (DSA)
    "dsa", "array", "arrays", "string", "strings", "stack", "queue", "deque",
    "linkedlist", "list", "tree", "trees", "bst", "heap", "graph", "graphs",
    "hashmap", "hashtable", "hashing", "trie", "matrix", "sorting", "sort",
    "search", "binary", "traversal", "bfs", "dfs", "dijkstra", "greedy",
    "backtracking", "recursion", "memoization", "leetcode", "complexity",
    "bigo", "optimize", "optimization", "iterative", "pointers",
}

# Multi-word / phrase signals matched as substrings of the lowercased message.
_TECH_PHRASES = (
    "data structure", "linked list", "binary tree", "binary search",
    "dynamic programming", "time complexity", "space complexity", "big o",
    "big-o", "hash map", "hash table", "priority queue", "two pointer",
    "sliding window", "depth first", "breadth first", "design pattern",
    "machine learning", "neural network", "unit test", "write a function",
    "reverse a", "sort a", "search a",
)


def is_technical_question(text: str) -> bool:
    lowered = text.lower()
    if any(phrase in lowered for phrase in _TECH_PHRASES):
        return True
    words = set(re.findall(r"[a-z0-9+#']+", lowered))
    return bool(words & _TECH_WORDS)


# --- Lightweight, dependency-free answer scorer ---
# Scores each candidate answer against the original query using cheap
# heuristics: how much of the query's meaningful vocabulary the answer
# covers, whether the answer looks like a real (non-error, non-empty)
# response, and whether its length is in a sensible range. This replaces
# a torch/sentence-transformers CrossEncoder, which added ~800MB of
# dependencies and multi-second cold starts for a marginal ranking gain.
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "to", "of", "in", "on", "for", "with", "and", "or", "but", "if",
    "how", "what", "why", "when", "where", "who", "which", "do", "does",
    "did", "i", "you", "it", "this", "that", "my", "me", "can", "could",
    "would", "should", "please", "explain", "tell", "about",
}

_ERROR_MARKERS = (
    "error occurred", "api key not configured", "returned an empty response",
    "openrouter api error", "groq api error",
)


def _tokenize(text: str) -> set:
    return {w for w in re.findall(r"[a-zA-Z0-9']+", text.lower()) if w not in _STOPWORDS and len(w) > 2}


def score_answer(query: str, answer: str) -> float:
    if not answer or not answer.strip():
        return 0.0

    lowered = answer.lower()
    if any(marker in lowered for marker in _ERROR_MARKERS):
        return 0.0

    query_terms = _tokenize(query)
    answer_terms = _tokenize(answer)
    overlap = len(query_terms & answer_terms) / len(query_terms) if query_terms else 0.5

    word_count = len(answer.split())
    if word_count < 8:
        length_score = word_count / 8
    elif word_count > 400:
        length_score = max(0.4, 1 - (word_count - 400) / 800)
    else:
        length_score = 1.0

    structure_bonus = 0.1 if ("```" in answer or re.search(r"\n\s*[-*\d]", answer)) else 0.0

    return round(0.6 * overlap + 0.3 * length_score + structure_bonus, 4)


# --- Groq API Connector ---
async def ask_groq(messages: list) -> str:
    if not GROQ_API_KEY:
        return "Groq API key not configured."

    api_url = "https://api.groq.com/openai/v1/chat/completions"
    # NOTE: Groq decommissioned the Llama 3.1 models on this account (the old
    # "llama-3.1-8b-instant" now 404s), which silently knocked this provider out
    # of every comparison. gpt-oss-20b is a currently-served Groq chat model.
    payload = {"model": "openai/gpt-oss-20b", "messages": messages}
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}

    last_error = "unknown error"
    async with httpx.AsyncClient(timeout=30) as client:
        # One retry: transient 429/5xx and cold-start timeouts are common on the
        # free tier and otherwise silently knock out this model in the comparison.
        for attempt in range(2):
            try:
                r = await client.post(api_url, json=payload, headers=headers)
                r.raise_for_status()
                content = r.json().get("choices", [{}])[0].get("message", {}).get("content", "")
                return content.strip() or "Groq returned an empty response."
            except Exception as e:
                last_error = str(e)
                if attempt == 0:
                    await asyncio.sleep(0.8)
    return f"An error occurred with the Groq API: {last_error}"


# --- OpenRouter API Connector ---
async def ask_openrouter(messages: list) -> str:
    if not OPENROUTER_API_KEY:
        return "OpenRouter API key not configured."

    api_url = "https://openrouter.ai/api/v1/chat/completions"

    payload = {
        "model": "openai/gpt-3.5-turbo",
        "messages": messages,
        "max_tokens": 1024,
        "temperature": 0.7,
        "stream": False,
    }
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": ALLOWED_ORIGINS[0] if ALLOWED_ORIGINS else "http://localhost:3000",
        "X-Title": "Prompt-Engineering-ChatBot",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    async with httpx.AsyncClient(timeout=60) as client:
        try:
            r = await client.post(api_url, json=payload, headers=headers)
            if r.status_code != 200:
                return f"OpenRouter API Error: Status {r.status_code} - {r.text}"
            content = r.json().get("choices", [{}])[0].get("message", {}).get("content", "")
            return content.strip() or "OpenRouter returned an empty response. Please try again."
        except Exception as e:
            return f"OpenRouter API Error: {str(e)}"


# --- Health check (used by Railway/uptime pings) ---
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "groq_configured": bool(GROQ_API_KEY),
        "openrouter_configured": bool(OPENROUTER_API_KEY),
    }


# --- Main Chat Endpoint ---
# Sessions persist across a whole conversation (not just one turn) so
# follow-up questions ("explain that further") have context. A session
# only resets when the frontend starts a new chat (new session_id) or
# the process restarts (in-memory store).
MAX_HISTORY_TURNS = 8  # user+assistant pairs kept per session, to bound prompt size


@app.post("/chat")
async def chat(req: ChatRequest):
    session = SESSIONS.setdefault(
        req.session_id, {"history": [], "system_prompt": "You are a helpful assistant."}
    )

    if req.answers:
        # The user just answered the clarifying questions for a pending technical question.
        use_case = req.answers.get("use_case", "") or "learning"
        skill_level = req.answers.get("skill_level", "") or "intermediate"
        original_question = session.pop("pending_question", req.text)

        depth = {
            "beginner": "Explain in plain language, avoid jargon, and define any term you introduce.",
            "intermediate": "Assume working familiarity; be concise but complete.",
            "advanced": "Be technical and precise; skip the basics and focus on nuances, edge cases and trade-offs.",
        }.get(skill_level.lower(), "Assume working familiarity; be concise but complete.")

        session["system_prompt"] = (
            "You are an expert software engineering and prompt engineering tutor. "
            f"{depth} "
            "Always put any code inside fenced Markdown code blocks with the correct language tag "
            "(e.g. ```python). Keep prose and explanations outside the code blocks. "
            "For DSA questions, briefly note the time and space complexity."
        )
        user_content = (
            f'Question: "{original_question}"\n'
            f"Use case: {use_case}. Skill level: {skill_level}.\n"
            "Give a tailored answer following the guidance above."
        )
        score_query = original_question
    else:
        text = req.text.strip()
        if not text:
            return {"type": "answer", "answer": "Please type a message.", "meta": None}

        # Ask for use-case + skill-level on EVERY technical/DSA question (unless
        # we're already waiting on the answers to a previous one). Non-technical
        # questions skip straight to a plain answer.
        if "pending_question" not in session and is_technical_question(text):
            session["pending_question"] = text
            return {
                "type": "clarify",
                "intro": "This looks like a technical question. Pick your use case and skill level and I'll tailor the answer.",
                "questions": [
                    {
                        "id": "use_case",
                        "text": "What are you using this for?",
                        "options": [
                            {"value": "learning", "label": "Learning the concept"},
                            {"value": "interview", "label": "Interview / exam prep"},
                            {"value": "production", "label": "Production / real project"},
                            {"value": "research", "label": "Research / deep dive"},
                        ],
                    },
                    {
                        "id": "skill_level",
                        "text": "What's your skill level?",
                        "options": [
                            {"value": "beginner", "label": "Beginner"},
                            {"value": "intermediate", "label": "Intermediate"},
                            {"value": "advanced", "label": "Advanced"},
                        ],
                    },
                ],
            }
        user_content = text
        score_query = text

    messages = [{"role": "system", "content": session["system_prompt"]}]
    messages.extend(session["history"])
    messages.append({"role": "user", "content": user_content})

    groq_response, openrouter_response = await asyncio.gather(
        ask_groq(messages),
        ask_openrouter(messages),
    )

    groq_score = score_answer(score_query, groq_response)
    openrouter_score = score_answer(score_query, openrouter_response)

    def _ok(resp: str) -> bool:
        lowered = (resp or "").lower()
        return bool(resp and resp.strip()) and not any(m in lowered for m in _ERROR_MARKERS) \
            and "api error" not in lowered and "not configured" not in lowered

    groq_ok = _ok(groq_response)
    openrouter_ok = _ok(openrouter_response)

    # Prefer a model that actually returned a usable answer over one that errored,
    # regardless of the heuristic score, so a single provider failure never
    # leaves the user staring at an error as the "winning" answer.
    if groq_ok and not openrouter_ok:
        winner = "groq"
    elif openrouter_ok and not groq_ok:
        winner = "openrouter"
    elif groq_score >= openrouter_score:
        winner = "groq"
    else:
        winner = "openrouter"

    if winner == "groq":
        model_name, raw_answer = "Groq GPT-OSS 20B", groq_response
    else:
        model_name, raw_answer = "GPT-3.5 Turbo", openrouter_response

    session["history"].append({"role": "user", "content": user_content})
    session["history"].append({"role": "assistant", "content": raw_answer})
    session["history"] = session["history"][-(MAX_HISTORY_TURNS * 2):]

    return {
        "type": "answer",
        "answer": f"{raw_answer}\n\n---\n*Answer from **{model_name}**, selected by the ranking model.*",
        "meta": {
            "winner": winner,
            "model": model_name,
            "scores": {"groq": groq_score, "openrouter": openrouter_score},
            "ok": {"groq": groq_ok, "openrouter": openrouter_ok},
            "alternates": {"groq": groq_response, "openrouter": openrouter_response},
        },
    }

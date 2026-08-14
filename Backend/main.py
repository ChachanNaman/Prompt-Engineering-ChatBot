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
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

SESSIONS: dict = {}  # In-memory session storage: what the user gave and got


# --- Pydantic Models ---
class ChatRequest(BaseModel):
    session_id: str
    text: str
    answers: dict | None = None


# --- Technical question detection ---
def is_technical_question(text: str) -> bool:
    technical_keywords = [
        "python", "code", "javascript", "react", "fastapi", "typescript",
        "sql", "api", "algorithm", "function", "debug", "error", "css",
        "html", "java", "docker", "git", "regex",
    ]
    words = set(re.findall(r"[a-z0-9']+", text.lower()))
    return any(keyword in words for keyword in technical_keywords)


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
    payload = {"model": "llama-3.1-8b-instant", "messages": messages}
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}

    async with httpx.AsyncClient(timeout=30) as client:
        try:
            r = await client.post(api_url, json=payload, headers=headers)
            r.raise_for_status()
            content = r.json().get("choices", [{}])[0].get("message", {}).get("content", "")
            return content.strip() or "Groq returned an empty response."
        except Exception as e:
            return f"An error occurred with the Groq API: {str(e)}"


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
        use_case = req.answers.get("use_case", "")
        skill_level = req.answers.get("skill_level", "")
        original_question = session.pop("pending_question", req.text)
        session["system_prompt"] = "You are an expert prompt engineering tutor and Python developer."
        user_content = (
            f'User\'s original question: "{original_question}"\n'
            f'Their use-case is: "{use_case}"\n'
            f'Their skill-level is: "{skill_level}"\n'
            f"Please provide a tailored answer."
        )
        score_query = original_question
    else:
        text = req.text.strip()
        if not text:
            return {"type": "answer", "answer": "Please type a message.", "meta": None}

        # Only trigger the clarification flow for the first technical question
        # of a brand-new conversation — not on every technical-sounding follow-up.
        if not session["history"] and "pending_question" not in session and is_technical_question(text):
            session["pending_question"] = text
            return {
                "type": "clarify",
                "questions": [
                    {"id": "use_case", "text": "This seems like a technical question. Which use case do you want (learning, research, production)?"},
                    {"id": "skill_level", "text": "What's your technical knowledge level? (beginner, intermediate, advanced)"},
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

    if groq_score >= openrouter_score:
        winner, model_name, raw_answer = "groq", "Groq Llama 3.1", groq_response
    else:
        winner, model_name, raw_answer = "openrouter", "GPT-3.5 Turbo", openrouter_response

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
            "alternates": {"groq": groq_response, "openrouter": openrouter_response},
        },
    }

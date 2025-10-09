from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import httpx
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

# --- Middleware for CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API Key and Session Storage ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SESSIONS = {}

# --- Pydantic Models ---
class ChatRequest(BaseModel):
    session_id: str
    text: str
    answers: dict | None = None

# --- NEW: Function to detect technical questions ---
def is_technical_question(text: str):
    """A simple check for keywords to decide if a question is technical."""
    technical_keywords = ["python", "code", "llm", "api", "prompt", "programming", "function", "script", "app", "error"]
    return any(keyword in text.lower() for keyword in technical_keywords)

# --- Groq API Connector (No changes here) ---
async def ask_groq(prompt: str):
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="Groq API key not configured on the server.")
    
    api_url = "https://api.groq.com/openai/v1/chat/completions"
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [{"role": "user", "content": prompt}]
    }
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            r = await client.post(api_url, json=payload, headers=headers)
            r.raise_for_status()
            response_json = r.json()
            return response_json["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as e:
            error_details = e.response.json().get('error', {}).get('message', 'Unknown API error')
            raise HTTPException(status_code=e.response.status_code, detail=f"Groq API Error: {error_details}")
        except (KeyError, IndexError):
            raise HTTPException(status_code=500, detail="Could not parse the response from the AI.")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

# --- Main Chat Endpoint (Completely updated logic) ---
@app.post("/chat")
async def chat(req: ChatRequest):
    sid = req.session_id
    session = SESSIONS.setdefault(sid, {})

    # --- THIS IS THE NEW "SMART" LOGIC ---
    
    # If this is the first message of a conversation, decide if it's technical
    if "is_technical" not in session:
        session["is_technical"] = is_technical_question(req.text)
        session["original_question"] = req.text

    # --- PATH 1: General Conversation ---
    if not session["is_technical"]:
        prompt = f"You are a helpful assistant. Answer the user's question directly and concisely. User question: {session['original_question']}"
        answer = await ask_groq(prompt)
        SESSIONS.pop(sid, None)  # End session
        return {"type": "answer", "answer": answer}

    # --- PATH 2: Technical Conversation ---
    else:
        if not session.get("clarified"):
            # If we have the answers, update the session
            if req.answers:
                session.update(req.answers)
                session["clarified"] = True
            # Otherwise, ask the clarifying questions
            else:
                return {
                    "type": "clarify",
                    "questions": [
                        {"id": "use_case", "text": "This seems like a technical question. Which use case do you want (learning, research, production)?"},
                        {"id": "skill_level", "text": "What's your technical knowledge level? (beginner, intermediate, advanced)"}
                    ]
                }
        
        # Once we have the context, build the detailed prompt
        original_question = session.get("original_question", "No question found.")
        use_case = session.get("use_case", "general")
        skill_level = session.get("skill_level", "beginner")

        prompt = f"""
User question: {original_question}
Use-case: {use_case}
Skill-level: {skill_level}

You are a prompt engineering tutor. Answer in three parts:
1. A clear explanation of the concept, tailored to the user's skill level.
2. A minimal, correct Python code example.
3. Tips on how to write better prompts related to this topic.
"""
        answer = await ask_groq(prompt)
        
        SESSIONS.pop(sid, None) # End session
        return {"type": "answer", "answer": answer}


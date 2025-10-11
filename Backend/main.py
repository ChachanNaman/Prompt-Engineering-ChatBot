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

# --- Function to detect technical questions ---
def is_technical_question(text: str):
    """A simple check for keywords to decide if a question is technical."""
    technical_keywords = [
        "python", "code", "javascript", "java", "c++", "html", "css", 
        "llm", "api", "prompt", "programming", "function", "script", "app", 
        "error", "bug", "debug", "install", "library", "framework", "algorithm",
        "data structure", "binary search", "3sum", "react", "fastapi"
    ]
    return any(keyword in text.lower() for keyword in technical_keywords)

# --- Groq API Connector ---
async def ask_groq(prompt: str, system_prompt: str = None):
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="Groq API key not configured on the server.")
    
    api_url = "https://api.groq.com/openai/v1/chat/completions"
    
    if system_prompt is None:
        system_prompt = "You are a helpful assistant. Answer questions concisely."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    
    payload = {"model": "llama-3.1-8b-instant", "messages": messages}
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

# --- Main Chat Endpoint (Final Hybrid Logic) ---
@app.post("/chat")
async def chat(req: ChatRequest):
    sid = req.session_id
    session = SESSIONS.setdefault(sid, {})

    # If this is the first message of a conversation, decide its type
    if "is_technical" not in session:
        session["is_technical"] = is_technical_question(req.text)
        session["original_question"] = req.text

    # --- PATH 1: General Conversation ---
    if not session["is_technical"]:
        answer = await ask_groq(session['original_question'])
        SESSIONS.pop(sid, None)  # End session after one answer
        return {"type": "answer", "answer": answer}

    # --- PATH 2: Technical Conversation ---
    else:
        if not session.get("clarified"):
            if req.answers:
                session.update(req.answers)
                session["clarified"] = True
            else:
                return {
                    "type": "clarify",
                    "questions": [
                        {"id": "use_case", "text": "This seems like a technical question. Which use case do you want (learning, research, production)?"},
                        {"id": "skill_level", "text": "What's your technical knowledge level? (beginner, intermediate, advanced)"}
                    ]
                }
        
        original_question = session.get("original_question", "No question found.")
        use_case = session.get("use_case", "general")
        skill_level = session.get("skill_level", "beginner")

        system_prompt = "You are an expert prompt engineering tutor and Python developer."
        detailed_prompt = f"""
        User's original question: "{original_question}"
        Their use-case is: "{use_case}"
        Their skill-level is: "{skill_level}"

        Please provide an answer tailored to these needs. Structure your response in three distinct parts:
        1. **Clear Explanation:** Explain the core concept in a way that is appropriate for their skill level.
        2. **Minimal Python Example:** Provide a clear, correct, and minimal Python code snippet that demonstrates the concept.
        3. **Prompt Improvement Tips:** Give 2-3 specific tips on how the user could write a better prompt to get this kind of answer in the future.
        """
        answer = await ask_groq(detailed_prompt, system_prompt)
        
        SESSIONS.pop(sid, None)
        return {"type": "answer", "answer": answer}


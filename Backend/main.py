from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import httpx
from dotenv import load_dotenv
import asyncio
from sentence_transformers import CrossEncoder

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

# --- API Keys and Session Storage ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")
SESSIONS = {}

# --- Load the Machine Learning Ranking Model ---
print("Loading ranking model...")
ranking_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
print("Ranking model loaded.")


# --- Pydantic Models ---
class ChatRequest(BaseModel):
    session_id: str
    text: str
    answers: dict | None = None

# --- Function to detect technical questions ---
def is_technical_question(text: str):
    technical_keywords = ["python", "code", "javascript", "react", "fastapi"]
    return any(keyword in text.lower() for keyword in technical_keywords)

# --- Groq API Connector ---
async def ask_groq(prompt: str, system_prompt: str = None):
    if not GROQ_API_KEY:
        return "Groq API key not configured."
    
    print("... Calling Groq API")
    api_url = "https://api.groq.com/openai/v1/chat/completions"
    if system_prompt is None:
        system_prompt = "You are a helpful assistant."

    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
    payload = {"model": "llama-3.1-8b-instant", "messages": messages}
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}

    async with httpx.AsyncClient(timeout=30) as client:
        try:
            r = await client.post(api_url, json=payload, headers=headers)
            r.raise_for_status()
            content = r.json().get("choices", [{}])[0].get("message", {}).get("content", "")
            if not content.strip(): return "Groq returned an empty response."
            print("✅ Groq API call successful")
            return content
        except Exception as e:
            print(f"🔴 Groq API Error: {e}")
            return f"An error occurred with the Groq API: {str(e)}"

# --- FINAL: Fireworks.ai API Connector (Corrected) ---
async def ask_fireworks(prompt: str, system_prompt: str = None):
    if not FIREWORKS_API_KEY:
        return "Fireworks API key not configured."

    print("... Calling Fireworks.ai API")
    api_url = "https://api.fireworks.ai/inference/openai/v1/chat/completions"
    if system_prompt is None:
        system_prompt = "You are a helpful assistant."
    
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
    
    # THIS IS THE FIX: Using the simple model name for the OpenAI endpoint
    payload = {"model": "llama-v3-8b-instruct", "messages": messages}
    
    headers = {"Authorization": f"Bearer {FIREWORKS_API_KEY}", "Content-Type": "application/json"}

    async with httpx.AsyncClient(timeout=60) as client:
        try:
            r = await client.post(api_url, json=payload, headers=headers)
            r.raise_for_status()
            content = r.json().get("choices", [{}])[0].get("message", {}).get("content", "")
            if not content.strip(): return "Fireworks AI returned an empty response."
            print("✅ Fireworks API call successful")
            return content
        except Exception as e:
            print(f"🔴 Fireworks API Error: {e}")
            return f"An error occurred with the Fireworks API: {str(e)}"

# --- Main Chat Endpoint ---
@app.post("/chat")
async def chat(req: ChatRequest):
    print("\n--- New Request Received ---")
    session = SESSIONS.setdefault(req.session_id, {})

    # --- Session logic ---
    if session.get("is_technical") and not session.get("clarified") and req.text.strip():
        print("User asked a new question while clarifying. Resetting session.")
        session = {}
        SESSIONS[req.session_id] = session

    if req.answers:
        session.update(req.answers)
        session["clarified"] = True

    if "is_technical" not in session:
        session["is_technical"] = is_technical_question(req.text)
        session["original_question"] = req.text
    
    system_prompt = "You are a helpful assistant."
    prompt_to_send = session["original_question"]

    # --- Technical question logic ---
    if session.get("is_technical") and not session.get("clarified"):
        print("Asking clarification questions...")
        return { "type": "clarify", "questions": [ {"id": "use_case", "text": "This seems like a technical question..."}, {"id": "skill_level", "text": "What's your technical knowledge level..."} ]}
    elif session.get("is_technical"):
        print("Handling clarified technical question...")
        system_prompt = "You are an expert prompt engineering tutor and Python developer."
        original_question = session.get("original_question", "")
        use_case = session.get("use_case", "")
        skill_level = session.get("skill_level", "")
        prompt_to_send = f"""
        User's original question: "{original_question}"
        Their use-case is: "{use_case}"
        Their skill-level is: "{skill_level}"
        Please provide a tailored answer...
        """

    print("Starting concurrent API calls to Groq and Fireworks...")
    groq_response, fireworks_response = await asyncio.gather(
        ask_groq(prompt_to_send, system_prompt),
        ask_fireworks(prompt_to_send, system_prompt)
    )
    print("... Both API calls finished.")

    print("Ranking responses with ML model...")
    query = session["original_question"]
    scores = ranking_model.predict([(query, groq_response), (query, fireworks_response)])
    print(f"Scores - Groq: {scores[0]:.4f}, Fireworks: {scores[1]:.4f}")

    if scores[0] >= scores[1]:
        print("🏆 Groq response selected.")
        best_answer = groq_response + "\n\n---\n*Answer from **Groq Llama 3.1**, selected by the ranking model.*"
    else:
        print("🏆 Fireworks response selected.")
        # Corrected the model name in the footer text
        best_answer = fireworks_response + "\n\n---\n*Answer from **Fireworks AI (Llama 3 8B)**, selected by the ranking model.*"

    SESSIONS.pop(req.session_id, None)
    print("--- Request Complete ---\n")
    return {"type": "answer", "answer": best_answer}
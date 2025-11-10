from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import httpx
from dotenv import load_dotenv
import asyncio
from sentence_transformers import CrossEncoder

load_dotenv()  # Load environment variables from .env file
app = FastAPI()     # Create FastAPI app

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
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

SESSIONS = {} # In-memory session storage what user gave and get

# --- Load the Machine Learning Ranking Model ---
print("Loading ranking model...")
ranking_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
print("Ranking model loaded.")
#When your server first starts, 
#it loads the ms-marco-MiniLM-L-6-v2 model into memory.

# --- Pydantic Models ---
#Defines the JSON structure FastAPI expects for a /chat request.
class ChatRequest(BaseModel):
    session_id: str
    text: str
    answers: dict | None = None

# --- Function to detect technical questions ---
#A simple helper function. It just checks if the user's text contains any keywords from your list.
# This is the trigger for your "clarification" logic.
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

    #They use httpx.AsyncClient to make the actual HTTP POST request, 
    # sending the user's prompt and your API key.

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

# --- OpenRouter API Connector ---
async def ask_openrouter(prompt: str, system_prompt: str = None):
    if not OPENROUTER_API_KEY:
        return "OpenRouter API key not configured."

    print("... Calling OpenRouter API")
    api_url = "https://openrouter.ai/api/v1/chat/completions"
    
    if system_prompt is None:
        system_prompt = "You are a helpful assistant."
    
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
    
    payload = {
        "model": "openai/gpt-3.5-turbo",  # Changed to a more reliable model
        "messages": messages,
        "max_tokens": 1024,
        "temperature": 0.7,
        "stream": False
    }
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "http://localhost:3000",
        "X-Title": "Prompt-Engineering-ChatBot",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    async with httpx.AsyncClient(timeout=60) as client:
        try:
            print(f"Sending request to OpenRouter with model: {payload['model']}")
            r = await client.post(api_url, json=payload, headers=headers)
            
            if r.status_code != 200:
                error_msg = f"OpenRouter API Error: Status {r.status_code} - {r.text}"
                print(f"🔴 {error_msg}")
                return error_msg
                
            response_json = r.json()
            content = response_json.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            if not content or not content.strip():
                print("🔴 OpenRouter returned empty content")
                return "OpenRouter returned an empty response. Please try again."
                
            print("✅ OpenRouter API call successful")
            return content
            
        except Exception as e:
            error_msg = f"OpenRouter API Error: {str(e)}"
            print(f"🔴 {error_msg}")
            return error_msg


# --- Main Chat Endpoint ---
#This decorator tells FastAPI to create a new endpoint that listens for HTTP POST requests at the /chat URL.
@app.post("/chat")
async def chat(req: ChatRequest):
    print("\n--- New Request Received ---")
    session = SESSIONS.setdefault(req.session_id, {}) #It tries to get the session for the given session_id. 
    #If it doesn't exist, it creates a new empty dictionary {} for that user and returns it.

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
    prompt_to_send = req.text  # Set the prompt from the request text

    # --- Technical question logic ---
    if session.get("is_technical") and not session.get("clarified"):
        ##Is this a new technical question that has not been clarified yet?
        #It stops immediately and returns the clarify message with the two questions. 
        #It does not call any LLMs.
        print("Asking clarification questions...")
        return {
            "type": "clarify",
            "questions": [
                {"id": "use_case", "text": "This seems like a technical question. Which use case do you want (learning, research, production)?"},
                {"id": "skill_level", "text": "What's your technical knowledge level? (beginner, intermediate, advanced)"}
            ]
        }
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
        Please provide a tailored answer...f
        """
    #If the question was technical and clarified, it engineers a new prompt. 
    #Instead of just sending "how to use react",fzseit sends a much more detailed prompt including 
    #the user's original question, their use case (e.g., "production"), and their skill level (e.g., "beginner"). 
    #This will result in a much better, more tailored answer from the LLMs.

    print("Starting concurrent API calls to Groq and OpenRouter...")
    groq_response, openrouter_response = await asyncio.gather(
        ask_groq(prompt_to_send, system_prompt),
        ask_openrouter(prompt_to_send, system_prompt)
    )
    print("... Both API calls finished.")

    print("Ranking responses with ML model...")   
    query = session.get("original_question", req.text)  #his is where your loaded ML model is used.
    scores = ranking_model.predict([(query, groq_response), (query, openrouter_response)])
    print(f"Scores - Groq: {scores[0]:.4f}, OpenRouter: {scores[1]:.4f}")

    if scores[0] >= scores[1]:
        print("🏆 Groq response selected.")
        best_answer = groq_response + "\n\n---\n*Answer from **Groq Llama 3.1**, selected by the ranking model.*"
    else:
        print("🏆 OpenRouter response selected.")
        best_answer = openrouter_response + "\n\n---\n*Answer from **GPT-3.5-turbo**, selected by the ranking model.*"
    
    SESSIONS.pop(req.session_id, None)
    print("--- Request Complete ---\n")
    return {"type": "answer", "answer": best_answer}



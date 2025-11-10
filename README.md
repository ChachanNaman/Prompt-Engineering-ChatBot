# Prompt-Engineering-ChatBot

A simple chatbot framework built using prompt-engineering techniques.  
This repository contains a backend (Python) and a frontend (JavaScript/React) to run a conversational UI leveraging prompts and an LLM (or API) to generate responses.

---

## 🔍 Structure

- **Backend/** — Python backend code (API, model-interface, prompt wrappers)  
- **frontend/** — React (or web) frontend code (UI for chat)  
- **requirements.txt** — Python dependencies  
- **package-lock.json** — Node package lock for frontend  
- **.gitignore** — standard ignored files  

---

## ✅ Prerequisites

Before you begin, ensure you have:

- Python 3.8+ installed  
- Node.js 14+ / npm or yarn installed  
- An API key (OpenAI, Groq, OpenRouter, etc.)  
- Git installed  

---

## 🛠 Installation

### Clone the repository

```bash
git clone https://github.com/ChachanNaman/Prompt-Engineering-ChatBot.git
cd Prompt-Engineering-ChatBot
```

### Install backend dependencies

```bash
cd Backend
pip install -r requirements.txt
```

### Install frontend dependencies

```bash
cd ../frontend
npm install   # or yarn install
```

---

## 🔧 Configuration

Create a `.env` file in the backend and add:

```
GROQ_API_KEY=your_api_key_here
OPENROUTER_API_KEY=your_api_key_here
```

---

## 🚀 Running the Project

### Backend

```bash
cd Backend
python app.py
```

### Frontend

```bash
cd frontend
npm run start   # or yarn start
```

Now open:

```
http://localhost:3000
```

---

## 🧠 Usage

- Open the frontend UI  
- Type a message  
- The frontend sends your message to the backend  
- Backend builds the prompt → calls the LLM → returns response  
- You can customize prompt logic in `Backend/`

---

## 📁 Customisation & Extension

- Modify system prompts  
- Add memory / chat history  
- Change LLM provider (Groq, OpenRouter, OpenAI, Gemini, etc.)  
- Enhance UI (themes, voice, file upload)

---

## 📚 Resources & References

- OpenAI Prompt Engineering Guide  
- Prompt Engineering research papers (arXiv)  
- GeeksforGeeks articles on LLMs  

---

## 🧑‍💻 Contributing

1. Fork the repo  
2. Create a branch (`feature-new-prompt`)  
3. Make changes  
4. Submit a PR  
5. Ensure style/formatting is consistent  

---

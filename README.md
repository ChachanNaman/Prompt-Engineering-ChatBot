<div align="center">

# ✦ Prompt Engineering Chatbot

**A dual-model AI chatbot that races Groq (Llama 3.1) against OpenRouter (GPT‑3.5) on every message, auto-scores both answers, and serves you the better one — wrapped in a glassmorphic React UI.**

[![Backend](https://img.shields.io/badge/backend-FastAPI-009688?logo=fastapi&logoColor=white)](Backend)
[![Frontend](https://img.shields.io/badge/frontend-React_19-61DAFB?logo=react&logoColor=black)](frontend)
[![Deploy](https://img.shields.io/badge/deploy-Render_%2B_Vercel-6d5bf7)](#-deployment)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](#)

[Live Demo](#) &nbsp;·&nbsp; [Features](#-features) &nbsp;·&nbsp; [Architecture](#-architecture) &nbsp;·&nbsp; [Local Setup](#-local-setup) &nbsp;·&nbsp; [Deployment](#-deployment)

</div>

---

## 🔍 Overview

Most "AI chatbot" demos wrap a single API call in a UI. This one is a small pipeline:

1. It classifies whether your question is **technical** and, if so, asks two quick clarifying questions (use case + skill level) to tailor the prompt.
2. It fires the *same* engineered prompt at **two different LLM providers concurrently** — Groq's `llama-3.1-8b-instant` and OpenRouter's `gpt-3.5-turbo`.
3. A lightweight, dependency-free scoring function ranks both responses on vocabulary overlap with your question, answer length/quality heuristics, and structure — and the better one wins.
4. The UI shows you which model won, lets you copy the answer, retry on failure, and includes live grammar suggestions and voice input.

## ✨ Features

- **Concurrent dual-model querying** — `asyncio.gather` calls Groq and OpenRouter in parallel, not sequentially.
- **Auto-ranking** — a fast heuristic scorer (term overlap, length sanity, structure bonus) picks the best answer with no ML runtime required.
- **Technical-question clarification flow** — detects technical intent and asks for use-case + skill level before answering, so code explanations are actually tailored.
- **Live grammar suggestions** — debounced calls to LanguageTool suggest corrected phrasing as you type.
- **Voice input** — dictate your message via the Web Speech API (Chrome/Edge).
- **Markdown-rendered responses** — code blocks, lists, and formatting render properly instead of as raw text.
- **Model badge + confidence scores** — see which model answered and how it scored, on hover.
- **Glassmorphic UI with light/dark mode** — theme persists across sessions, animated ambient background, fully responsive.
- **Copy-to-clipboard, retry-on-error, new-chat reset** — the small conveniences a real chat UI needs.

## 🏗 Architecture

```
┌─────────────────┐        POST /chat         ┌──────────────────────┐
│  React frontend │ ────────────────────────▶ │   FastAPI backend     │
│  (Vercel)       │                            │   (Render)             │
│                  │ ◀──────────────────────── │                        │
└─────────────────┘      { answer, meta }      │  ┌──────────────────┐  │
                                                │  │ asyncio.gather() │  │
                                                │  └───┬──────────┬───┘  │
                                                │      ▼          ▼      │
                                                │   Groq API   OpenRouter│
                                                │  (Llama 3.1)   (GPT-3.5)│
                                                │      │          │      │
                                                │      ▼          ▼      │
                                                │   score_answer() heuristic │
                                                │      picks the winner  │
                                                └──────────────────────┘
```

**Backend** — Python 3.12, FastAPI, `httpx` for async HTTP calls, in-memory session store for the clarification flow.
**Frontend** — React 19 (Create React App), `react-markdown` for rendering, CSS custom properties for theming, no CSS framework — hand-rolled glassmorphism.

## 📁 Project Structure

```
Backend/
  main.py              FastAPI app: chat endpoint, clarification flow, scoring, CORS
  requirements.txt
  Procfile             Railway/Heroku-style start command
  .env.example
frontend/
  src/App.js           Chat UI, state, API calls, voice + grammar features
  src/App.css          Theming, glassmorphism, layout, animations
  .env.example
```

## ✅ Prerequisites

- Python 3.10+
- Node.js 18+ / npm
- A [Groq API key](https://console.groq.com/keys) and an [OpenRouter API key](https://openrouter.ai/keys) (both have free tiers)

## 🚀 Local Setup

### Backend

```bash
cd Backend
python3 -m venv ../venv && source ../venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in your API keys
uvicorn main:app --reload
```

Backend runs at `http://127.0.0.1:8000`. Check `http://127.0.0.1:8000/health` to confirm your keys are picked up.

### Frontend

```bash
cd frontend
npm install
cp .env.example .env.local   # optional locally — defaults to http://127.0.0.1:8000
npm start
```

Frontend runs at `http://localhost:3000`.

## 🌐 Deployment

This app is split-deployed: **Render** for the backend (long-running Python process, free tier), **Vercel** for the frontend (static React build).

### Backend → Render

1. Push this repo to GitHub.
2. On [Render](https://render.com), **New** → **Blueprint** → select this repo. It picks up the included `render.yaml` (root directory `Backend`, build/start commands, health check at `/health`).
   - Alternatively, create a **Web Service** manually: root directory `Backend`, build command `pip install -r requirements.txt`, start command `uvicorn main:app --host 0.0.0.0 --port $PORT`.
3. Add environment variables: `GROQ_API_KEY`, `OPENROUTER_API_KEY`, and `ALLOWED_ORIGINS` (set this to your Vercel URL once you have it, comma-separated with `http://localhost:3000` for local dev).
4. Note the generated public URL (e.g. `https://your-app.onrender.com`).

> **Free tier note:** Render's free web services spin down after 15 minutes of inactivity. The first request after idling takes ~30-50s to wake up. If this link is on your resume, consider pinging `/health` periodically (e.g. via [cron-job.org](https://cron-job.org) or UptimeRobot) to keep it warm during hours you expect it to be viewed.

### Frontend → Vercel

1. On [Vercel](https://vercel.com), import this repo → set the root directory to `frontend`.
2. Add environment variable `REACT_APP_API_URL` = your Render backend URL (no trailing slash).
3. Deploy. Vercel auto-detects Create React App (`npm run build`, output `build/`).
4. Once deployed, go back to Render and update `ALLOWED_ORIGINS` to include your new Vercel domain, then redeploy the backend.

## 🧠 How the ranking works

Instead of a heavyweight ML cross-encoder, `score_answer()` in `Backend/main.py` combines three cheap signals:

- **Term overlap** (60%) — how much of the question's meaningful vocabulary appears in the answer.
- **Length sanity** (30%) — penalizes answers that are too short to be useful or unreasonably long.
- **Structure bonus** (10%) — rewards answers that use code blocks or lists, since technical answers benefit from structure.

This keeps cold starts near-instant and removes an ~800MB PyTorch dependency, at the cost of being a heuristic rather than a learned ranker — a deliberate tradeoff for a fast, cheaply-hostable demo.

## 🗺 Roadmap Ideas

- Persist chat history (currently in-memory, cleared per session)
- Streaming responses token-by-token instead of waiting for both models
- Add a third provider and make the panel configurable
- Swap the heuristic scorer for a small hosted re-ranking API if latency budget allows

## 🧑‍💻 Contributing

1. Fork the repo
2. Create a branch (`feature/my-change`)
3. Make your changes, run `npm test` and `npm run build` in `frontend/`
4. Open a PR

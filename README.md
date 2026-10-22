# ShopEase Support

React/Vite chat UI backed by a FastAPI service. The API persists conversation history, escalation tickets, feedback and image-review records in SQLite. With `GEMINI_API_KEY` configured, chat and image triage use Gemini (`gemini-3.5-flash-lite` by default). Without a key, chat uses the local FAQ fallback and photos are routed for human review.

## Run locally

1. Create a Gemini API key in [Google AI Studio](https://aistudio.google.com/app/apikey). In `backend`, create a virtual environment, install dependencies, and put the key in `.env`:

   ```powershell
   cd backend
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   Copy-Item .env.example .env
   # Open .env and set GEMINI_API_KEY=your_key_here
   uvicorn main:app --reload --host 127.0.0.1 --port 8000
   ```

2. In another terminal, run the UI:

   ```powershell
   cd chatbot-ui
   npm install
   npm run dev
   ```

The API health endpoint is `http://localhost:8000/health`; interactive API docs are at `http://localhost:8000/docs`.

## API

- `GET /health` — service and database status.
- `GET /orders/{order_id}` — demo order lookup.
- `POST /chat` — FAQ/AI answer, saved conversation and escalation ticket.
- `POST /image` — accepts JPEG, PNG, WEBP or GIF up to 5 MB; uncertain assessments always go to human review.
- `POST /feedback` — records a vote for an assistant message.
- `POST /resume` — closes an open escalation for the session.

The sample order records in `main.py` are in-memory demo data. Connect `get_order` and `order_context` to your order system before using real customer orders. For production, also put the API behind HTTPS and add authentication, rate limiting, retention rules and a managed database.

"""ShopEase support API: persistent chat, feedback, escalation and image triage."""
import json
import os
import random
import re
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
FAQ = json.loads((BASE_DIR / "faq.json").read_text(encoding="utf-8"))
DB_PATH = Path(os.getenv("DATABASE_PATH", str(BASE_DIR / "shopease.db")))
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
MAX_IMAGE_BYTES = 5 * 1024 * 1024
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
client = None
if os.getenv("GEMINI_API_KEY"):
    try:
        from google import genai
        from google.genai import types
        client = genai.Client(
            api_key=os.environ["GEMINI_API_KEY"],
            http_options=types.HttpOptions(timeout=20000),
        )
    except ImportError:
        print("GEMINI_API_KEY is set, but google-genai is not installed; using FAQ fallback.")

app = FastAPI(title="ShopEase Support API", version="1.0.0")
origins = [origin.strip() for origin in os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if origin.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"], allow_headers=["Content-Type"])

# Seed orders are demo data. Replace them with your real order source before deployment.
ORDERS = {
    "SE1001": {"status": "Shipped", "carrier": "BlueDart", "tracking": "BD88213", "expected_delivery": "3 Oct 2026"},
    "SE1002": {"status": "Delivered", "carrier": "Delhivery", "tracking": "DL4410", "delivered_on": "28 Sep 2026"},
    "SE1003": {"status": "Processing", "expected_delivery": "6 Oct 2026"},
}
LANG_NAMES = {"en-IN": "English", "hi-IN": "Hindi", "kn-IN": "Kannada", "ta-IN": "Tamil"}
MODEL_COOLDOWN: dict[str, float] = {}


@contextmanager
def db():
    connection = sqlite3.connect(DB_PATH, timeout=10)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


def initialize_db():
    with db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('user','assistant')),
                content TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);
            CREATE TABLE IF NOT EXISTS escalations (
                session_id TEXT PRIMARY KEY, ticket_id TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL,
                message_id INTEGER NOT NULL, value TEXT NOT NULL CHECK(value IN ('up','down')),
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(session_id, message_id)
            );
            CREATE TABLE IF NOT EXISTS image_reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL,
                order_id TEXT, findings TEXT NOT NULL, confidence REAL NOT NULL,
                recommended_action TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
        """)


initialize_db()


class ChatIn(BaseModel):
    session_id: str = Field(min_length=8, max_length=100)
    text: str = Field(min_length=1, max_length=4000)
    language: str = Field(default="en-IN", max_length=10)


class FeedbackIn(BaseModel):
    session_id: str = Field(min_length=8, max_length=100)
    message_id: int = Field(gt=0)
    value: str = Field(pattern="^(up|down)$")


class ResumeIn(BaseModel):
    session_id: str = Field(min_length=8, max_length=100)


def order_context(session_id: str, text: str) -> str:
    with db() as conn:
        previous = [r[0] for r in conn.execute("SELECT content FROM messages WHERE session_id=? AND role='user' ORDER BY id DESC LIMIT 10", (session_id,))]
    for message in [text, *previous]:
        match = re.search(r"\bSE\d{4}\b", message.upper())
        if match:
            order_id = match.group(0)
            return f"{order_id}: {json.dumps(ORDERS[order_id])}" if order_id in ORDERS else f"{order_id}: NOT FOUND"
    return "No order id provided."


def system_prompt(language: str, order_info: str) -> str:
    faq = "\n".join(f"[{item['category']}] Q: {item['q']}\nA: {item['a']}" for item in FAQ["faqs"])
    language_name = LANG_NAMES.get(language, "English")
    return f"""You are the customer support assistant for {FAQ['company']}.
Use only the FAQ and order data below. Never invent policies, prices, or order details.
Always reply in the selected interface language: {language_name}, even when the customer's message is written in another language. Use the normal script for that language. Keep replies short and polite.
Escalate only for a human request, anger/frustration, fraud, legal action, safety issue, or repeated inability to help.
When uncertain, say so and offer a human agent. Return JSON only with keys reply, intent, category, sentiment, escalate.
FAQ:\n{faq}\nORDER DATA:\n{order_info}"""


def parse_model_json(raw: str) -> dict:
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Model did not return a JSON object")
    item = json.loads(raw[start:end + 1])
    if not isinstance(item.get("reply"), str) or not item["reply"].strip():
        raise ValueError("Model response has no reply")
    allowed_categories = {"Orders", "Payments", "Returns & refunds", "Account", "Product info", "Technical issues", "General", "Escalation", "Unknown"}
    return {
        "reply": item["reply"][:2000], "intent": str(item.get("intent", "support"))[:80],
        "category": item.get("category") if item.get("category") in allowed_categories else "Unknown",
        "sentiment": item.get("sentiment") if item.get("sentiment") in {"positive", "neutral", "negative"} else "neutral",
        "escalate": bool(item.get("escalate", False)),
    }


def fallback_answer(text: str, order_info: str) -> dict:
    lowered = text.lower()
    if re.search(r"\b(human|agent|representative|complaint|fraud|legal|unsafe)\b", lowered):
        return {"reply": "I’m connecting you with a human agent.", "intent": "human_request", "category": "Escalation", "sentiment": "negative", "escalate": True}
    order_match = re.search(r"\bSE\d{4}\b", text.upper())
    if order_match:
        if "NOT FOUND" in order_info:
            reply = f"I couldn’t find order {order_match.group(0)}. Please check the order ID or ask me to connect you with an agent."
        else:
            reply = f"Order details: {order_info.replace(order_match.group(0) + ': ', '')}"
        return {"reply": reply, "intent": "track_order", "category": "Orders", "sentiment": "neutral", "escalate": False}
    for faq in FAQ["faqs"]:
        terms = set(re.findall(r"[a-z]{4,}", faq["q"].lower()))
        matches = [term for term in terms if term in lowered]
        if len(matches) >= 2 or (matches and any(word in lowered for word in ("refund", "return", "payment", "password", "address", "warranty", "order"))):
            return {"reply": faq["a"], "intent": "faq_answer", "category": faq["category"], "sentiment": "neutral", "escalate": False}
    return {"reply": "I’m not sure about that. I can help with orders, payments, returns, and account questions, or connect you with an agent.", "intent": "unknown", "category": "Unknown", "sentiment": "neutral", "escalate": False}


def generate_chat(text: str, history: list[dict], language: str, order_info: str) -> dict:
    if client is None:
        return fallback_answer(text, order_info)
    contents = [
        types.Content(
            role="model" if turn["role"] == "assistant" else "user",
            parts=[types.Part(text=turn["content"])],
        )
        for turn in history
    ]
    contents.append(types.Content(role="user", parts=[types.Part(text=text)]))
    now = time.time()
    try_models = [MODEL] if MODEL_COOLDOWN.get(MODEL, 0) <= now else []
    try_models += [name for name in ("gemini-3.5-flash-lite",) if name != MODEL and MODEL_COOLDOWN.get(name, 0) <= now]
    for model in try_models:
        try:
            response = client.models.generate_content(model=model, contents=contents,
                config=types.GenerateContentConfig(system_instruction=system_prompt(language, order_info), response_mime_type="application/json", max_output_tokens=400))
            return parse_model_json(response.text or "")
        except Exception as exc:
            message = str(exc)
            MODEL_COOLDOWN[model] = time.time() + (300 if "429" in message else 60)
            print(f"Gemini request failed ({model}): {message[:160]}")
    return fallback_answer(text, order_info)


def save_message(session_id: str, role: str, content: str) -> int:
    with db() as conn:
        cursor = conn.execute("INSERT INTO messages(session_id, role, content) VALUES(?,?,?)", (session_id, role, content))
        return int(cursor.lastrowid)


def escalate(session_id: str) -> str:
    with db() as conn:
        row = conn.execute("SELECT ticket_id FROM escalations WHERE session_id=?", (session_id,)).fetchone()
        if row:
            return row[0]
        ticket = f"SE-{random.randint(10000, 99999)}"
        conn.execute("INSERT INTO escalations(session_id, ticket_id) VALUES(?,?)", (session_id, ticket))
        return ticket


@app.get("/health")
def health():
    with db() as conn:
        conn.execute("SELECT 1")
    return {"status": "ok", "database": "sqlite", "ai_enabled": client is not None, "model": MODEL if client else None}


@app.get("/orders/{order_id}")
def get_order(order_id: str):
    order = ORDERS.get(order_id.upper())
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    return {"order_id": order_id.upper(), **order}


@app.post("/chat")
def chat(body: ChatIn):
    sid = body.session_id
    with db() as conn:
        ticket = conn.execute("SELECT ticket_id FROM escalations WHERE session_id=?", (sid,)).fetchone()
        history = [dict(row) for row in conn.execute("SELECT role, content FROM messages WHERE session_id=? ORDER BY id DESC LIMIT 10", (sid,))]
    if ticket:
        return {"reply": f"A support ticket ({ticket[0]}) is open. An agent will reply here shortly.", "intent": "waiting", "category": "Escalation", "sentiment": "neutral", "escalate": True, "ticket_id": ticket[0]}
    history.reverse()
    order_info = order_context(sid, body.text)
    answer = generate_chat(body.text, history, body.language, order_info)
    save_message(sid, "user", body.text)
    message_id = save_message(sid, "assistant", answer["reply"])
    if answer["escalate"]:
        ticket_id = escalate(sid)
        answer["ticket_id"] = ticket_id
        answer["reply"] += f"\n(Ticket #{ticket_id})"
    answer["message_id"] = message_id
    return answer


@app.post("/image")
async def analyze_image(image: UploadFile = File(...), order_id: str = Form(""), session_id: str = Form(...)):
    content_type = image.content_type or ""
    if not content_type.startswith("image/") or content_type not in {"image/jpeg", "image/png", "image/webp", "image/gif"}:
        raise HTTPException(status_code=415, detail="Upload a JPEG, PNG, WEBP, or GIF image")
    data = await image.read(MAX_IMAGE_BYTES + 1)
    if not data:
        raise HTTPException(status_code=400, detail="The uploaded image is empty")
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image must be 5 MB or smaller")
    result = {"findings": "I can’t reliably assess this photo automatically. A support agent should review it.", "confidence": 0.3, "recommended_action": "human_review"}
    if client is not None:
        try:
            prompt = "Inspect this customer support product photo. Describe only visible damage, do not infer cause or eligibility. Return JSON only with findings (one concise sentence), confidence (0 to 1), and recommended_action (human_review or request_clearer_photo). Do not approve a refund or replacement."
            response = client.models.generate_content(model=MODEL, contents=[types.Part.from_bytes(data=data, mime_type=content_type), prompt], config=types.GenerateContentConfig(response_mime_type="application/json", max_output_tokens=300))
            candidate = json.loads(response.text or "{}")
            confidence = float(candidate.get("confidence", 0.3))
            if not 0 <= confidence <= 1:
                raise ValueError("Invalid confidence")
            result = {"findings": str(candidate.get("findings", result["findings"]))[:500], "confidence": confidence, "recommended_action": "human_review"}
        except Exception as exc:
            print(f"Image analysis failed; sending for human review: {str(exc)[:160]}")
    with db() as conn:
        conn.execute("INSERT INTO image_reviews(session_id, order_id, findings, confidence, recommended_action) VALUES(?,?,?,?,?)", (session_id, order_id.upper() or None, result["findings"], result["confidence"], result["recommended_action"]))
    return result


@app.post("/feedback")
def feedback(body: FeedbackIn):
    with db() as conn:
        exists = conn.execute("SELECT 1 FROM messages WHERE id=? AND session_id=? AND role='assistant'", (body.message_id, body.session_id)).fetchone()
        if not exists:
            raise HTTPException(status_code=404, detail="Assistant message not found")
        conn.execute("INSERT INTO feedback(session_id, message_id, value) VALUES(?,?,?) ON CONFLICT(session_id, message_id) DO UPDATE SET value=excluded.value", (body.session_id, body.message_id, body.value))
    return {"ok": True}


@app.post("/resume")
def resume(body: ResumeIn):
    with db() as conn:
        conn.execute("DELETE FROM escalations WHERE session_id=?", (body.session_id,))
    return {"ok": True}

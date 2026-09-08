from fastapi import FastAPI, Header, HTTPException, Depends
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
import re
import json
import os
import sqlite3
import time
from typing import Optional, List, Dict, Any
import google.generativeai as genai

app = FastAPI(title="AI Anywhere")

# ============================================================
# CONFIG
# ============================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set")
genai.configure(api_key=GEMINI_API_KEY)

# Model selection via environment variables
LIGHT_MODEL = os.getenv("AI_LIGHT_MODEL", "gemini-1.0-pro")
HEAVY_MODEL = os.getenv("AI_HEAVY_MODEL", "gemini-1.0-pro")

APP_SECRET_KEY = os.getenv("APP_SECRET_KEY", "").strip()
DB_FILE = os.getenv("AI_ANYWHERE_DB", "ai_memory.db")
HISTORY_LIMIT = 5
DAILY_FREE_LIMIT = int(os.getenv("DAILY_FREE_LIMIT", "5"))

# ============================================================
# AUTH
# ============================================================

def verify_api_key(x_api_key: str = Header(default="")):
    if not APP_SECRET_KEY:
        raise HTTPException(status_code=500, detail="Server auth not configured.")
    if x_api_key != APP_SECRET_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")

# ============================================================
# DATABASE (same as before – no changes)
# ============================================================

def db():
    conn = sqlite3.connect(DB_FILE, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=5000;")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            contact_name TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp REAL NOT NULL
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_chat_lookup
        ON chat_history(user_id, contact_name, timestamp)
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS user_profiles (
            user_id TEXT PRIMARY KEY,
            writing_style TEXT NOT NULL DEFAULT 'Natural, simple and respectful',
            emoji_preference TEXT NOT NULL DEFAULT 'rare'
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS usage_daily (
            user_id TEXT NOT NULL,
            day TEXT NOT NULL,
            count INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (user_id, day)
        )
    """)
    conn.commit()
    return conn

# ... (get_chat_history, save_chat_message, load_user_profile, save_user_profile, get_today_usage, increment_today_usage) 
# These functions remain exactly as in the original code – no changes needed.
# I will copy them below for completeness.

def get_chat_history(user_id: str, contact_name: str) -> List[Dict[str, str]]:
    conn = db()
    try:
        rows = conn.execute("""
            SELECT role, content
            FROM chat_history
            WHERE user_id = ? AND contact_name = ?
            ORDER BY timestamp DESC, id DESC
            LIMIT ?
        """, (user_id, contact_name, HISTORY_LIMIT)).fetchall()
        rows.reverse()
        return [{"role": r, "content": c} for r, c in rows]
    finally:
        conn.close()

def save_chat_message(user_id: str, contact_name: str, role: str, content: str):
    content = (content or "").strip()
    if not content:
        return
    conn = db()
    try:
        conn.execute("""
            INSERT INTO chat_history
            (user_id, contact_name, role, content, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """, (user_id, contact_name, role, content, time.time()))
        conn.execute("""
            DELETE FROM chat_history
            WHERE user_id = ?
              AND contact_name = ?
              AND id NOT IN (
                  SELECT id
                  FROM chat_history
                  WHERE user_id = ?
                    AND contact_name = ?
                  ORDER BY timestamp DESC, id DESC
                  LIMIT ?
              )
        """, (user_id, contact_name, user_id, contact_name, HISTORY_LIMIT))
        conn.commit()
    finally:
        conn.close()

# ============================================================
# USER PROFILE functions (unchanged)
# ============================================================

DEFAULT_PROFILE = {
    "writing_style": "Natural, simple and respectful",
    "emoji_preference": "rare",
}

def load_user_profile(user_id: str) -> Dict[str, Any]:
    conn = db()
    try:
        row = conn.execute(
            "SELECT writing_style, emoji_preference FROM user_profiles WHERE user_id = ?",
            (user_id,)
        ).fetchone()
        if row:
            return {"writing_style": row[0], "emoji_preference": row[1]}
        return dict(DEFAULT_PROFILE)
    finally:
        conn.close()

def save_user_profile(user_id: str, writing_style: str, emoji_preference: str):
    conn = db()
    try:
        conn.execute("""
            INSERT INTO user_profiles (user_id, writing_style, emoji_preference)
            VALUES (?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                writing_style = excluded.writing_style,
                emoji_preference = excluded.emoji_preference
        """, (user_id, writing_style, emoji_preference))
        conn.commit()
    finally:
        conn.close()

# ============================================================
# DAILY USAGE functions (unchanged)
# ============================================================

def _today_str() -> str:
    return time.strftime("%Y-%m-%d", time.gmtime())

def get_today_usage(user_id: str) -> int:
    conn = db()
    try:
        row = conn.execute(
            "SELECT count FROM usage_daily WHERE user_id = ? AND day = ?",
            (user_id, _today_str())
        ).fetchone()
        return row[0] if row else 0
    finally:
        conn.close()

def increment_today_usage(user_id: str):
    conn = db()
    try:
        conn.execute("""
            INSERT INTO usage_daily (user_id, day, count) VALUES (?, ?, 1)
            ON CONFLICT(user_id, day) DO UPDATE SET count = count + 1
        """, (user_id, _today_str()))
        conn.commit()
    finally:
        conn.close()

# ============================================================
# REQUEST MODELS (unchanged)
# ============================================================

class TextRequest(BaseModel):
    text: str
    command: str = "reply"
    user_id: str = "default_user_1"
    contact_name: str = "current_chat"
    custom_prompt: str = ""
    language: Optional[str] = None
    tone: Optional[str] = None
    recent_messages: Optional[List[Dict[str, str]]] = None
    is_premium: bool = False

class ClearMemoryRequest(BaseModel):
    user_id: str = "default_user_1"
    contact_name: str = "current_chat"

class UpdateProfileRequest(BaseModel):
    user_id: str = "default_user_1"
    writing_style: Optional[str] = None
    emoji_preference: Optional[str] = None

# ============================================================
# COMMAND ALIASES (unchanged)
# ============================================================

ALIASES = {
    "/reply": "reply", "reply": "reply",
    "/fix": "fix", "fix": "fix", "/grammar": "fix",
    "/english": "translate", "english": "translate",
    "/eng": "translate", "eng": "translate",
    "/translate": "translate", "translate": "translate",
    "/hindi": "hindi", "hindi": "hindi",
    "/hinglish": "hinglish", "hinglish": "hinglish",
    "/formal": "formal", "formal": "formal",
    "/professional": "formal", "professional": "formal",
    "/polite": "polite", "polite": "polite",
    "/casual": "casual", "casual": "casual",
    "/ask": "ask", "ask": "ask", "/ans": "ask", "ans": "ask",
    "/improve": "improve", "improve": "improve",
    "/better": "improve", "better": "improve",
    "/short": "short", "short": "short",
    "/long": "expand", "long": "expand", "/expand": "expand",
    "expand": "expand",
    "/bullet": "bullet", "bullet": "bullet",
    "/summ": "summarize", "summ": "summarize",
    "/summary": "summarize", "summary": "summarize",
    "/summarize": "summarize", "summarize": "summarize",
    "/emoji": "emoji", "emoji": "emoji",
    "/simple": "simple", "simple": "simple",
    "/rewrite": "rewrite", "rewrite": "rewrite",
}

def normalize_command(command: str) -> str:
    raw = (command or "reply").strip().lower()
    return ALIASES.get(raw, raw.lstrip("/"))

# ============================================================
# TEXT CLEANUP (unchanged)
# ============================================================

def clean_output(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    text = re.sub(r"^(text|output|result|response|answer)\s*:\s*", "", text, flags=re.IGNORECASE).strip()
    if len(text) >= 2 and ((text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'"))):
        text = text[1:-1].strip()
    return text

def sanitize_history(items):
    clean = []
    if not isinstance(items, list):
        return clean
    for item in items:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).lower().strip()
        content = str(item.get("content", "")).strip()
        if role in ("user", "assistant") and content:
            clean.append({"role": role, "content": content})
    return clean[-HISTORY_LIMIT:]

# ============================================================
# NEW: SHORT, FOCUSED SYSTEM PROMPTS
# ============================================================

# Common instruction for all tasks
BASE_INSTRUCTION = """
You are AI Anywhere, a text transformation assistant.
- Strictly follow the user's command.
- Preserve the original meaning, intent, language, and script unless explicitly asked to translate.
- Correct obvious typos and misinterpreted words using the context (e.g., "defred duty" → "deferred duty" in a customs context).
- Never add explanations, notes, or conversational filler. Output only the final transformed text.
"""

# For light commands (fix, translate, short, etc.)
LIGHT_SYSTEM = BASE_INSTRUCTION + """
The task is straightforward. Apply the transformation exactly as asked.
- If translating, output only the translation.
- If fixing, correct grammar/spelling while keeping the original language.
"""

# For heavy commands (reply, ask, improve, expand)
HEAVY_SYSTEM = BASE_INSTRUCTION + """
For @reply: Write a natural, human-like reply that fits the context and the user's communication style. Match the tone and language of the original message. Do not sound like an AI.
For @ask: Answer the question directly and factually. If you don't know, say "I don't know." Do not repeat or rephrase the question.
For @improve / @expand: Enhance clarity and naturalness without inventing facts.
"""

# ============================================================
# BUILD TASK (simplified, but still includes command-specific hints)
# ============================================================

def build_task(command, text, custom_prompt="", language=None, tone=None):
    if custom_prompt.strip():
        return f"Instruction: {custom_prompt.strip()}\n\nText to process:\n{text}"
    
    # For simple commands, just say the command and the text
    # Gemini can infer what to do from the command name
    command_instructions = {
        "reply": "Write a natural reply to this message.",
        "fix": "Fix grammar, spelling, and punctuation.",
        "translate": "Translate this text into the target language.",
        "hindi": "Translate to Hindi (Devanagari).",
        "hinglish": "Translate to Hinglish (Romanized Hindi).",
        "formal": "Rewrite in a formal tone.",
        "polite": "Rewrite politely.",
        "casual": "Rewrite in a casual tone.",
        "improve": "Improve clarity and naturalness.",
        "short": "Make it shorter.",
        "expand": "Expand naturally.",
        "bullet": "Convert to bullet points.",
        "summarize": "Summarize concisely.",
        "simple": "Rewrite in simpler language.",
        "ask": "Answer the question directly.",
        "emoji": "Add appropriate emojis.",
        "rewrite": "Rephrase naturally."
    }
    instruction = command_instructions.get(command, f"Apply the '{command}' operation.")
    if language:
        instruction += f" Use language: {language}."
    if tone:
        instruction += f" Tone: {tone}."
    return f"{instruction}\n\nText:\n{text}"

# ============================================================
# GEMINI GENERATION (sync wrapper)
# ============================================================

def generate_gemini_response(model_name: str, system_prompt: str, contents: List[Dict], temp: float = 0.25) -> str:
    """
    contents is a list of {'role': 'user'/'model', 'parts': [text]}.
    System prompt is passed separately.
    """
    model = genai.GenerativeModel(
        model_name=model_name,
        system_instruction=system_prompt
    )
    # Convert contents to the format expected by generate_content
    # The contents list can be passed as is if roles are 'user' and 'model'
    response = model.generate_content(
        contents=contents,
        generation_config={"temperature": temp}
    )
    return response.text

# ============================================================
# MAIN API ENDPOINT
# ============================================================

@app.post("/process_text", dependencies=[Depends(verify_api_key)])
async def process_text(request: TextRequest):
    original_text = (request.text or "").strip()
    command = normalize_command(request.command)
    user_id = (request.user_id or "default_user_1").strip()
    contact_name = (request.contact_name or "current_chat").strip()

    if not original_text:
        return {"result": "", "error": "Text is empty."}

    # Server-side free-tier check
    if not request.is_premium:
        used_today = await run_in_threadpool(get_today_usage, user_id)
        if used_today >= DAILY_FREE_LIMIT:
            return {
                "result": "",
                "error": f"Daily free limit reached ({DAILY_FREE_LIMIT}/day). Upgrade to Premium.",
                "limit_reached": True,
            }

    # Load profile and history
    profile = await run_in_threadpool(load_user_profile, user_id)
    history = await run_in_threadpool(get_chat_history, user_id, contact_name)

    if request.recent_messages:
        supplied = sanitize_history(request.recent_messages)
        if supplied:
            history = supplied

    # Build system prompt and select model based on command
    if command in ("reply", "ask", "improve", "expand"):
        system_prompt = HEAVY_SYSTEM
        model_name = HEAVY_MODEL
    else:
        system_prompt = LIGHT_SYSTEM
        model_name = LIGHT_MODEL

    # Optionally add user profile context (only if relevant)
    style = str(profile.get("writing_style", "Natural, simple and respectful"))
    emoji_pref = str(profile.get("emoji_preference", "rare"))
    if command == "reply":
        system_prompt += f"\nUser's writing style: {style}. Emoji preference: {emoji_pref}. Use this as a guide, but prioritize the actual conversation."

    # Build the task prompt
    task = build_task(command, original_text, request.custom_prompt, request.language, request.tone)

    # Prepare conversation contents for Gemini
    contents = []
    # Include history (up to HISTORY_LIMIT)
    for msg in history:
        role = "user" if msg["role"] == "user" else "model"
        contents.append({"role": role, "parts": [msg["content"]]})
    # Add current user message
    contents.append({"role": "user", "parts": [task]})

    print(f"REQUEST | user={user_id} | contact={contact_name} | command={command} | model={model_name}")

    try:
        # Call Gemini via threadpool
        result = await run_in_threadpool(
            generate_gemini_response,
            model_name,
            system_prompt,
            contents,
            0.25
        )

        result = clean_output(result)
        if not result:
            return {"result": "", "error": "AI returned empty result.", "model_used": model_name}

        # Save to history
        await run_in_threadpool(save_chat_message, user_id, contact_name, "user", original_text)
        await run_in_threadpool(save_chat_message, user_id, contact_name, "assistant", result)

        if not request.is_premium:
            await run_in_threadpool(increment_today_usage, user_id)

        return {"result": result, "model_used": model_name, "command": command}

    except Exception as e:
        import traceback
        print("=" * 50)
        print("GEMINI EXCEPTION TYPE:", type(e))
        print("GEMINI ERROR MESSAGE:", str(e))
        traceback.print_exc()
        print("=" * 50)
        
        # Ab error ko response mein bhi bhejo (taake app par hi dikhe)
        error_msg = str(e)
        if "404" in error_msg or "no longer available" in error_msg:
            return {"result": "", "error": f"Model not found: {error_msg}"}
        if "429" in error_msg or "quota" in error_msg.lower():
            return {"result": "", "error": "Quota exceeded. Please try later."}
        if "permission" in error_msg.lower() or "auth" in error_msg.lower():
            return {"result": "", "error": "API Key invalid or missing permissions."}
        return {"result": "", "error": f"AI Error: {error_msg[:100]}"}  # Response mein bhejo

# ============================================================
# OTHER ENDPOINTS (unchanged)
# ============================================================

@app.get("/keep_awake")
def keep_awake():
    return {"status": "AI Anywhere Server is Awake!"}

@app.post("/update_profile", dependencies=[Depends(verify_api_key)])
async def update_profile(request: UpdateProfileRequest):
    current = await run_in_threadpool(load_user_profile, request.user_id)
    writing_style = request.writing_style or current["writing_style"]
    emoji_preference = request.emoji_preference or current["emoji_preference"]
    await run_in_threadpool(save_user_profile, request.user_id, writing_style, emoji_preference)
    return {"status": "ok", "profile": {"writing_style": writing_style, "emoji_preference": emoji_preference}}

@app.post("/clear_memory", dependencies=[Depends(verify_api_key)])
def clear_memory(request: ClearMemoryRequest):
    conn = db()
    try:
        conn.execute("DELETE FROM chat_history WHERE user_id = ? AND contact_name = ?", (request.user_id, request.contact_name))
        conn.commit()
        return {"status": "ok", "message": "Conversation memory cleared."}
    finally:
        conn.close()

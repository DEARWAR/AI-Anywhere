from fastapi import FastAPI
from pydantic import BaseModel
from groq import Groq
import re
import json
import os
import sqlite3
import time

app = FastAPI()

# 🟢 CRON-JOB PING ENDPOINT (Server ko jagane ke liye)
@app.get("/keep_awake")
def keep_awake():
    return {"status": "AI Anywhere Server is Awake!"}

# ... (Baaki aapka puraana code iske neeche rahega)

@app.get("/keep_awake")
async def keep_awake():
    return {"status": "AI Anywhere Server is Awake!"}

# ⚠️ Apni API key zaroor daalein ⚠️
API_KEY = "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" 
client = Groq(api_key=API_KEY)

# ==========================================
# 🟢 1. DATABASE SETUP & FUNCTIONS
# ==========================================
DB_FILE = "ai_memory.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            contact_name TEXT,
            role TEXT,
            content TEXT,
            timestamp REAL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

def get_chat_history(user_id: str, contact_name: str):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT role, content FROM chat_history 
        WHERE user_id=? AND contact_name=? 
        ORDER BY timestamp DESC LIMIT 10
    ''', (user_id, contact_name))
    rows = cursor.fetchall()
    conn.close()
    return [{"role": row[0], "content": row[1]} for row in reversed(rows)]

def save_chat_message(user_id: str, contact_name: str, role: str, content: str):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO chat_history (user_id, contact_name, role, content, timestamp)
        VALUES (?, ?, ?, ?, ?)
    ''', (user_id, contact_name, role, content, time.time()))
    cursor.execute('''
        DELETE FROM chat_history 
        WHERE id NOT IN (
            SELECT id FROM chat_history 
            WHERE user_id=? AND contact_name=? 
            ORDER BY timestamp DESC LIMIT 10
        ) AND user_id=? AND contact_name=?
    ''', (user_id, contact_name, user_id, contact_name))
    conn.commit()
    conn.close()

def load_user_profile():
    try:
        if os.path.exists("user_profile.json"):
            with open("user_profile.json", "r", encoding="utf-8") as file:
                return json.load(file)
    except Exception as e:
        print(f"Profile load error: {e}")
    return {"user_name": "User", "writing_style": "Normal"}

# 🟢 UPDATE: Custom Prompt ka field add kiya[cite: 2]
class TextRequest(BaseModel):
    text: str
    command: str
    user_id: str = "default_user_1"
    contact_name: str = "current_chat"
    custom_prompt: str = "" 

@app.post("/process_text")
def process_text(request: TextRequest):
    original_text = request.text
    command = request.command.lower()
    u_id = request.user_id
    c_name = request.contact_name
    custom_instruction = request.custom_prompt # Naya variable
    
    print(f"\n📲 NEW REQUEST -> User: {u_id} | Contact: '{c_name}' | Command: {command}")
    
    system_context = """You are 'AI Anywhere', a personal communication intelligence.
    CRITICAL RULES:
    1. THE GOLDEN RULE: Understand context but NEVER invent it. If time, dates, or details are not provided, DO NOT guess them. Do not hallucinate.
    2. NO PREFIXES: NEVER add "Text: ", "Output: ", or quotes. ONLY give the final text.
    3. STRICT LANGUAGE: "Hindi" = Devanagari ONLY. "Hinglish" = Roman English alphabet ONLY. No Arabic/Urdu.
    4. PRESERVE MEANING: Do not change the original intent of the user.
    """
    
    profile = load_user_profile()
    personal_context = f"""
    USER BASELINE PERSONA:
    - Name: {profile.get('user_name', 'User')}
    - Persona & Style: {profile.get('writing_style', 'Normal')}
    """
    system_prompt = system_context + "\n" + personal_context
    
    try:
        heavy_model = "qwen/qwen3.6-27b"        
        fast_model = "groq/compound-mini" #[cite: 2]    
        
        # 🟢 SMART ROUTER (Custom Commands Handle Karega)
        if command == "reply":
            selected_model = heavy_model
        else:
            selected_model = fast_model

        # 🟢 DYNAMIC PROMPT ENGINE
        if custom_instruction.strip() != "":
            # Agar Android se khud ka banaya hua rule aaya hai
            prompt = f"Task: {custom_instruction}\nText: '{original_text}'"
        else:
            # Puraane Hardcoded Rules
            if command == "fix":
                prompt = f"Task: Fix grammar and spelling.\nNow fix this: '{original_text}'"
            elif command in ["eng", "translate"]:
                prompt = f"Task: Translate to natural English.\nNow translate this: '{original_text}'"
            elif command == "hindi":
                prompt = f"Task: Translate to everyday spoken Hindi in Devanagari script (हिंदी) ONLY.\nNow translate this: '{original_text}'"
            elif command == "hinglish":
                prompt = f"Task: Translate to 'Hinglish'.\nNow translate this: '{original_text}'"
            elif command in ["formal", "polite"]:
                prompt = f"Task: Make this sound professional but keep it a SHORT 1-2 sentence message.\nNow do this: '{original_text}'"
            elif command == "casual":
                prompt = f"Task: Rewrite to sound like a friendly text message to a friend.\nNow do this: '{original_text}'"
            elif command == "reply":
                prompt = f"Task: Write a natural 1-sentence reply to this message. Do not invent times/dates.\nNow reply to this: '{original_text}'"
            elif command in ["ask", "ans"]:
                prompt = f"Task: Provide a direct, factual answer to this question. No conversational filler.\nQuestion: '{original_text}'"
            elif command == "expand":
                prompt = f"Task: Expand this into EXACTLY 2-3 natural sentences without inventing fake facts.\nText: '{original_text}'"
            elif command == "bullet":
                prompt = f"Task: Convert to a clean bulleted list.\nText: '{original_text}'"
            elif command == "summ":
                prompt = f"Task: Summarize this in 1 short sentence. Keep it in the exact same language.\nText: '{original_text}'"
            elif command == "emoji":
                prompt = f"Task: Add relevant emojis without changing the words.\nText: '{original_text}'"
            else:
                prompt = f"Task: Apply the command '{command}' to this text. Output only the final text.\nText: '{original_text}'"

        if command == "reply":
            strict_reminder = "\n\n[CRITICAL: Reply naturally based on chat history. NO EMOJIS. No AI vibes.]"
        elif command == "emoji":
            strict_reminder = "\n\n[CRITICAL: Add emojis naturally. ONLY the final text.]"
        else:
            strict_reminder = "\n\n[CRITICAL: Execute task perfectly. ZERO EMOJIS. Give ONLY final text.]"
        
        final_user_message = prompt + strict_reminder

        messages_to_send = [{"role": "system", "content": system_prompt}]
        past_chats = get_chat_history(u_id, c_name)
        messages_to_send.extend(past_chats) 
        messages_to_send.append({"role": "user", "content": final_user_message})

        chat_completion = client.chat.completions.create(
            messages=messages_to_send, 
            model=selected_model, 
            temperature=0.3,
        )
        
        ai_result = chat_completion.choices[0].message.content.strip()
        
        if "</think>" in ai_result:
            ai_result = re.sub(r'<think>.*?</think>', '', ai_result, flags=re.DOTALL).strip()
        elif "<think>" in ai_result:
            ai_result = ai_result.split("<think>")[-1].strip()
            
        if "Here's a thinking process:" in ai_result or "*Analyze User Input:*" in ai_result:
            parts = ai_result.split("\n\n")
            ai_result = parts[-1].strip()

        if ai_result.lower().startswith("text:"):
            ai_result = ai_result[5:].strip(" '\"")
            
        if not ai_result:
            ai_result = "Bhai, AI ne blank answer diya."

        save_chat_message(u_id, c_name, "user", prompt) 
        save_chat_message(u_id, c_name, "assistant", ai_result)
        
        return {"result": ai_result, "model_used": selected_model}
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return {"result": f"Error from AI: {str(e)}"}

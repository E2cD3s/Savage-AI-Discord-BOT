from flask import Flask, render_template, jsonify, request
import threading
import torch
import time
import asyncio
import logging
import psutil
import datetime
import os
from collections import deque

import shutil
import subprocess
import requests

app = Flask(__name__)

# Global references (assigned when server starts)
BOT = None
CONFIG = None
CONNECTIONS = None
TTS_SERVICE = None
SPEAK_FUNC = None
UPDATE_PROMPT_FUNC = None
AI_CLIENT = None  # For model switching

LOGS = deque(maxlen=50)

def get_gpu_stats():
    """
    Returns (vram_info, gpu_load_str) using nvidia-smi.
    """
    vram = "N/A"
    gpu_load = "N/A"
    try:
        if shutil.which("nvidia-smi"):
            output = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.total,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                encoding="utf-8"
            )
            lines = output.strip().split('\n')
            if lines:
                parts = lines[0].split(',')
                total_mb = float(parts[0])
                used_mb = float(parts[1])
                gpu_util = parts[2].strip()
                
                total_gb = total_mb / 1024
                used_gb = used_mb / 1024
                
                vram = f"{used_gb:.2f}GB / {total_gb:.2f}GB"
                gpu_load = f"{gpu_util}%"
    except:
        vram = "Err"
        gpu_load = "Err"
    return vram, gpu_load

def log_message(author, content, msg_type="info"):
    """
    Adds a message to the live feed.
    msg_type: 'user', 'bot', 'info', 'error'
    """
    entry = {
        "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
        "author": author,
        "content": content,
        "type": msg_type
    }
    LOGS.appendleft(entry)


def run_server(bot, config, connections, tts_service, speak_func=None, update_prompt_func=None, ai_client=None):
    global BOT, CONFIG, CONNECTIONS, TTS_SERVICE, SPEAK_FUNC, UPDATE_PROMPT_FUNC, AI_CLIENT
    BOT = bot
    CONFIG = config
    CONNECTIONS = connections
    TTS_SERVICE = tts_service
    SPEAK_FUNC = speak_func
    UPDATE_PROMPT_FUNC = update_prompt_func
    AI_CLIENT = ai_client
    
    # Store this log function in the bot for easy access? 
    # Or just import web_server in discord_bot.py and call web_server.log_message
    
    # Startup Log
    log_message("System", "Dashboard connected. Live feed active.", "info")
    
    # Disable Flask banner
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/status')
def get_status():
    # VRAM & GPU Load
    vram, gpu_load = get_gpu_stats()
        
    # Ping
    ping = round(BOT.latency * 1000) if BOT else 0
    
    connected_info = "Idle"
    if CONNECTIONS:
        # Use simple iteration over keys copy to avoid "dictionary changed size" error
        for guild_id in list(CONNECTIONS.keys()):
            vc = CONNECTIONS.get(guild_id)
            if vc and vc.is_connected():
                connected_info = f"{vc.guild.name} / {vc.channel.name}"
                break
    
    # Guild List
    guilds = []
    if BOT:
        for g in BOT.guilds:
            guilds.append({
                "id": str(g.id),
                "name": g.name,
                "member_count": g.member_count,
                "icon": str(g.icon.url) if g.icon else None
            })

    # System Analytics
    cpu_usage = psutil.cpu_percent()
    ram_usage = psutil.virtual_memory().percent
    
    uptime_str = "Unknown"
    if BOT and hasattr(BOT, "launch_time"):
        uptime_seconds = int(time.time() - BOT.launch_time)
        uptime_str = str(datetime.timedelta(seconds=uptime_seconds))

    # Filter Features
    raw_features = CONFIG.get("features", {})
    features = raw_features # Send all features

    descriptions = {
        "roast": "🔥 Enables the /roast command.\nUsage: /roast @user",
        "rude_welcomes": "😈 Insults users when they join VC.\nAutomatic (Passive)",
        "savage_reactions": "🤡 Randomly reacts to messages.\nAutomatic (Passive)",
        "auto_clown": "🎪 Chance to reply with a Clown License.\nConfigurable Chance",
        "activity_shaming": "🕵️ Mocks users for Music, Games, & Streams.\nOnly triggers when you interact."
    }
    # AI Model Info
    stt_model = CONFIG.get("stt_model_size", "small.en")
    model_info = {
        "llm": {
            "name": CONFIG.get("local_ai_model", os.getenv("LOCAL_AI_MODEL", "Unknown")),
            "endpoint": CONFIG.get("local_ai_base_url", os.getenv("LOCAL_AI_BASE_URL", "Unknown"))
        },
        "stt": f"Faster-Whisper ({stt_model}) CUDA",
        "tts": "Kokoro-82M (Hexgrad) CUDA",
        "vad": "Silero VAD v4 (CUDA)",
        "image_gen": "Stable Diffusion 1.5 (CPU Offload)"
    }
                
    return jsonify({
        "vram": vram,
        "clown_chance": CONFIG.get("auto_clown_chance", 0.0) * 100,
        "rudeness_level": CONFIG.get("features", {}).get("rudeness_level", 100),
        "context_depth": CONFIG.get("max_memory_depth", 5),
        "gpu_load": gpu_load,
        "ping": ping,
        "cpu": cpu_usage,
        "ram": ram_usage,
        "uptime": uptime_str,
        "vc_status": connected_info,
        "features": features,
        "descriptions": descriptions,
        "guilds": guilds,
        "models": model_info
    })

@app.route('/api/leave_guild', methods=['POST'])
def leave_guild():
    guild_id = request.json.get('guild_id')
    if not guild_id: return jsonify({"error": "No guild ID"}), 400
    
    guild = BOT.get_guild(int(guild_id))
    if guild:
        asyncio.run_coroutine_threadsafe(guild.leave(), BOT.loop)
        return jsonify({"status": "ok", "message": f"Left {guild.name}"})
        
    return jsonify({"error": "Guild not found"}), 404

@app.route('/api/logs')
def get_logs():
    return jsonify(list(LOGS))

@app.route('/api/voice', methods=['GET', 'POST'])
def manage_voice():
    if request.method == 'GET':
        # Return current voice and available options (hardcoded for now)
        start_voice = TTS_SERVICE.default_voice if TTS_SERVICE else "af_bella"
        # Common Kokoro variants
        options = ["af_bella", "af_nicole", "af_sarah", "am_adam", "am_michael", "bf_emma", "bf_isabella", "bm_george", "bm_lewis"]
        return jsonify({"current": start_voice, "options": options})
        
    if request.method == 'POST':
        voice = request.json.get('voice')
        if voice and TTS_SERVICE:
            TTS_SERVICE.default_voice = voice
            # Also update config so it persists (runtime only unless we save)
            if CONFIG:
                CONFIG["kokoro_voice"] = voice
            return jsonify({"status": "ok", "voice": voice})
        return jsonify({"error": "Invalid request"}), 400

@app.route('/api/disconnect', methods=['POST'])
def disconnect():
    if CONNECTIONS:
        # We must run coroutines on the bot's loop
        for guild_id in list(CONNECTIONS.keys()):
            vc = CONNECTIONS[guild_id]
            async def disconnect_safe():
                 await vc.disconnect()
            
            asyncio.run_coroutine_threadsafe(disconnect_safe(), BOT.loop)
            del CONNECTIONS[guild_id]
            
    return jsonify({"status": "ok"})

@app.route('/api/rudeness', methods=['POST'])
def set_rudeness():
    level = request.json.get('level')
    if level is None: return jsonify({"error": "No level"}), 400
    
    level = int(level)
    if CONFIG:
        CONFIG["rudeness_level"] = level
    
    if UPDATE_PROMPT_FUNC:
        UPDATE_PROMPT_FUNC(level)
        
    return jsonify({"status": "ok", "level": level})

@app.route('/api/clown_chance', methods=['POST'])
def set_clown_chance():
    data = request.json
    chance = float(data.get('chance', 0)) / 100.0 # Convert 10% to 0.1
    if CONFIG:
        CONFIG["auto_clown_chance"] = chance
    return jsonify({"status": "ok", "chance": chance})

@app.route('/api/context_depth', methods=['POST'])
def set_context_depth():
    data = request.json
    depth = int(data.get('depth', 5))
    if CONFIG:
        CONFIG["max_memory_depth"] = depth
        
    # Update Memory Manager if accessible
    if hasattr(app, "memory_manager"):
        app.memory_manager.set_limit(depth)
        
    return jsonify({"status": "ok", "depth": depth})

@app.route('/api/reset_memory', methods=['POST'])
def reset_memory():
    if hasattr(app, "memory_manager"):
        app.memory_manager.clear_all()
        return jsonify({"status": "ok", "message": "Memory wiped."})
    return jsonify({"error": "Memory manager not initialized"}), 500

@app.route('/api/toggle', methods=['POST'])
def toggle():
    data = request.json
    feature = data.get('feature')
    state = data.get('state')
    
    if feature and CONFIG:
        if "features" not in CONFIG: CONFIG["features"] = {}
        CONFIG["features"][feature] = state
        return jsonify({"status": "ok", "feature": feature, "state": state})
        
    return jsonify({"error": "Invalid request"}), 400

@app.route('/api/restart', methods=['POST'])
def restart_bot():
    def suicide():
        time.sleep(1)
        os._exit(0) # Force exit, letting batch script loop
        
    # Schedule death
    threading.Thread(target=suicide).start()
    return jsonify({"status": "ok", "message": "Restarting system..."})

@app.route('/api/say', methods=['POST'])
def say():
    text = request.json.get('text')
    if not text: return jsonify({"error": "No text"}), 400
    
    # Send to active VCs
    if CONNECTIONS and SPEAK_FUNC:
        for guild_id, vc in CONNECTIONS.items():
            if vc.is_connected():
                # Use the thread-safe callback
                SPEAK_FUNC(guild_id, text)

    return jsonify({"status": "ok"})

@app.route('/api/models', methods=['GET'])
def list_models():
    """Fetch available models from LM Studio."""
    try:
        base_url = CONFIG.get("local_ai_base_url", os.getenv("LOCAL_AI_BASE_URL", "http://localhost:1234"))
        # LM Studio uses OpenAI-compatible /v1/models endpoint
        response = requests.get(f"{base_url}/models", timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = [m.get("id") for m in data.get("data", [])]
            return jsonify({"models": models, "current": AI_CLIENT.model if AI_CLIENT else "Unknown"})
        else:
            return jsonify({"error": f"LM Studio returned {response.status_code}"}), 500
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Could not connect to LM Studio: {str(e)}"}), 500

@app.route('/api/set_model', methods=['POST'])
def set_model():
    """Change the active LLM model."""
    model_id = request.json.get('model')
    if not model_id:
        return jsonify({"error": "No model specified"}), 400
    
    if AI_CLIENT:
        AI_CLIENT.model = model_id
        logging.info(f"Switched LLM model to: {model_id}")
        return jsonify({"status": "ok", "model": model_id})
    else:
        return jsonify({"error": "AI Client not initialized"}), 500

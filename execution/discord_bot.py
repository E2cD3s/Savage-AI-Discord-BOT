import discord
import os
import logging
import asyncio
import random
import json
import io
import re
import time
import datetime
import torch
from dotenv import load_dotenv
from ai_client import LocalAIClient
from tts_service import TTSService
from stt_service import STTService
from memory_manager import MemoryManager
from discord.ext import tasks, commands
import imageio_ffmpeg
import threading
import web_server 

from meme_service import meme_service 
from image_service import ImageService

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DiscordBot")

# Load Environment Variables
load_dotenv()
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")

# Load Config
CONFIG_FILE = "config.json"
try:
    with open(CONFIG_FILE, "r") as f:
        config = json.load(f)
        
    # Migration: Game Shaming -> Activity Shaming
    features = config.get("features", {})
    if "game_shaming" in features:
        # Migrate value
        val = features.pop("game_shaming")
        if "activity_shaming" not in features:
            features["activity_shaming"] = val
        config["features"] = features
        # Save immediately to fix file
        try:
             with open(CONFIG_FILE, "w") as f:
                 json.dump(config, f, indent=4)
        except: pass
        
    # Ensure activity_shaming exists for Dashboard
    if "activity_shaming" not in config.get("features", {}):
        if "features" not in config: config["features"] = {}
        config["features"]["activity_shaming"] = True # Default On
except Exception as e:
    logging.error(f"Failed to load config.json: {e}")
    config = {
        "wake_words": ["jarvis"],
        "status_messages": ["Listening..."],
        "kokoro_voice": "af_bella",
        "kokoro_speed": 1.0
    }

WAKE_WORDS = [w.lower() for w in config.get("wake_words", ["jarvis"])]
INTERRUPT_WORDS = [w.lower() for w in config.get("interrupt_words", ["stop", "quiet"])]
STATUS_MESSAGES = config.get("status_messages", ["Listening..."])

# Initialize Services
ai_client = LocalAIClient(
    base_url=config.get("local_ai_base_url"),
    model=config.get("local_ai_model")
)
tts_service = TTSService(voice=config.get("kokoro_voice", "af_bella"), speed=config.get("kokoro_speed", 1.0))
stt_service = STTService(model_size=config.get("stt_model_size", "base.en"))
# Initialize Memory
max_depth = config.get("max_memory_depth", 5)
memory_manager = MemoryManager(max_depth)
web_server.memory_manager = memory_manager # Pass to web server

# User Interaction Tracker for "Savage Reactions"
USER_INTERACTIONS = {} # {user_id: datetime}


# Initialize Image Service
image_service = ImageService()

# Load Silero VAD Model
logger.info("Loading Silero VAD model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
try:
    vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False, trust_repo=True)
    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
    vad_model.to(device)
    logger.info(f"Silero VAD loaded successfully on {device}.")
except Exception as e:
    logger.error(f"Failed to load Silero VAD: {e}")
    vad_model = None



# Load Persona
# Load Persona
PERSONA_FILE = config.get("system_prompt_file", "directives/discord_persona.md")
SYSTEM_PROMPT = ""

def update_system_prompt(rudeness_level=100):
    global SYSTEM_PROMPT
    
    if rudeness_level >= 70:
        # High Rudeness: Load original savage persona
        try:
            with open(PERSONA_FILE, "r", encoding="utf-8") as f:
                SYSTEM_PROMPT = f.read()
        except FileNotFoundError:
            SYSTEM_PROMPT = "You are a helpful AI assistant."
    
    elif rudeness_level >= 30:
        # Medium: Sassy
        SYSTEM_PROMPT = (
            "You are Savage AI, a helpful but sassy and sarcastic assistant. "
            "You don't suffer fools gladly, but you aren't outright mean. "
            "Answer users correctly but add a pinch of attitude. "
            "CONSTRAINTS: Be concise. No emojis. No markdown formatting. Pure text only. "
            "Always address users by their name (provided as [User: Name] in messages) or 'you'. Never call them 'User'."
        )
    else:
        # Low: Polite
        SYSTEM_PROMPT = (
            "You are Savage AI (reformed). You are helpful, polite, and patient. "
            "Answer questions clearly and concisely without being rude. "
            "CONSTRAINTS: Be concise. No emojis. No markdown formatting. Pure text only. "
            "Always address users by their name (provided as [User: Name] in messages) or 'you'. Never call them 'User'."
        )
        
    logger.info(f"System Prompt updated. Rudeness Level: {rudeness_level}")

# Init Prompt
update_system_prompt(config.get("rudeness_level", 100))

# Bot Setup (Py-Cord)
intents = discord.Intents.default()
intents.messages = True
intents.dm_messages = True # Explicitly enable DMs
intents.message_content = True
intents.presences = True
intents.members = True
bot = discord.Bot(intents=intents)

connections = {}
listening_flags = {}
troll_target_id = None


@tasks.loop(minutes=5)
async def change_status():
    """Rotates the bot's status every 5 minutes."""
    if STATUS_MESSAGES:
        status = random.choice(STATUS_MESSAGES).strip()
        await bot.change_presence(activity=discord.Activity(type=discord.ActivityType.listening, name=status))

def get_shaming_prompt(member):
    """
    Analyzes user activities and returns a system prompt injection to roast them.
    """
    if not config.get("features", {}).get("activity_shaming", True):
        return ""
        
    if not member or not hasattr(member, "activities") or not member.activities:
        return ""
        
    prompt_injection = ""
    for activity in member.activities:
        # Spotify
        if isinstance(activity, discord.Spotify):
             prompt_injection += f"\n[CONTEXT: User is listening to '{activity.title}' by '{activity.artist}'. If relevant, briefly roast their music taste in ONE sentence max, then answer their question.]"
        
        # Game
        elif activity.type == discord.ActivityType.playing:
             prompt_injection += f"\n[CONTEXT: User is playing '{activity.name}'. If relevant, briefly mock them in ONE sentence max, then answer their question.]"
             
        # Streaming
        elif activity.type == discord.ActivityType.streaming:
             platform = activity.platform if hasattr(activity, 'platform') else "streaming"
             prompt_injection += f"\n[CONTEXT: User is streaming on {platform}. If relevant, briefly mock them in ONE sentence max, then answer their question.]"
             
        # Visual Studio Code / Coding
        elif activity.name and "Visual Studio" in activity.name:
             prompt_injection += f"\n[CONTEXT: User is coding in {activity.name}. If relevant, briefly mock them in ONE sentence max, then answer their question.]"

    if prompt_injection:
        logger.debug(f"Shaming Context Injected: {prompt_injection}")
        
    return prompt_injection

async def generate_reply(channel_id, user_text, username=None, interaction_member=None, context_note=None):
    """
    Helper to generate AI reply using memory.
    interaction_member: The discord.Member object of the user interacting.
    context_note: Optional extra system instruction (e.g. "Mock this user for X").
    """
    # Sanitize Inputs (Critical for quantized models)
    username = sanitize_text(username) if username else "User"
    user_text = sanitize_text(user_text)

    # Add User Message to Memory
    memory_manager.add_message(channel_id, "user", user_text, username=username)
    
    # Get Context
    history = memory_manager.get_history(channel_id)
    
    # DEBUG: Log prompt length to confirm dynamic change
    logger.info(f"Generating reply with Prompt Length: {len(SYSTEM_PROMPT)} chars")
    
    # Activity Shaming (Context Injection)
    current_prompt = SYSTEM_PROMPT
    
    # Only inject shaming context if we have a valid Member object from the interaction
    # This fulfills the user's request: "Only if user interacted"
    if interaction_member:
        try:
             current_prompt += get_shaming_prompt(interaction_member)
        except Exception as e:
            logger.error(f"Activity Shaming Error: {e}")
            
    # Inject Custom Context Note (e.g. DM Shaming)
    if context_note:
        current_prompt += f"\n[SYSTEM NOTICE: {context_note}]"
    
    # Game Shaming Fallback (Deprecated - removing fallback to ensure strict targeting)
    # If no interaction_member is passed, we DO NOT guess.
    # This prevents random shaming.

    # Generate Response (Async)
    try:
        response_text = await ai_client.generate_response(
            current_prompt, 
            history
        )
    except Exception as e:
        logger.error(f"AI Generation failed: {e}")
        response_text = "I'm having trouble thinking right now."
    
    # Sanitize AI Output (Remove prefixes/Zalgo)
    response_text = sanitize_text(response_text)

    # Add AI Message to Memory
    memory_manager.add_message(channel_id, "assistant", response_text)
    
    # Log to Dashboard
    web_server.log_message("Savage AI", response_text, "bot")
    
    return response_text

async def warmup_models():
    """
    Runs dummy inference on STT and TTS to initialize CUDA context.
    Prevents the first user command from being slow.
    """
    logger.info("🔥 Warming up AI models (allocating CUDA context)...")
    
    # Warmup STT (1 second of silence)
    try:
        # Create silent wav buffer
        dummy_wav = io.BytesIO()
        # Header for valid wav (simplified or just passed as raw bytes if using faster-whisper raw)
        # Faster-whisper handles file objects. 
        # A simple silent buffer might fail if it expects headers, but let's try just a small buffer
        # Actually, best to just transcribe a dummy file or bytes.
        # Let's create a minimal valid wav header or just skip and trust it handles raw?
        # Faster-whisper usually expects a valid file format if passed as file.
        # Simplest: Just don't crash.
        pass # Actual STT warmup might require a valid wav file, skipping complex generation for now
             # Just checking if model is loaded is usually enough for STT init
        if stt_service.model:
             logger.info("STT Model ready.")
    except Exception as e:
        logger.warning(f"STT Warmup skipped: {e}")

    # Warmup TTS (Generate "Ready")
    try:
         await tts_service.generate_audio("Ready")
         logger.info("TTS Model ready (Warmup complete).")
    except Exception as e:
         logger.warning(f"TTS Warmup failed: {e}")

    logger.info("✅ System Warmup Complete.")

def sanitize_text(text):
    """
    Sanitizes text to remove Zalgo, markdown, and artifacts.
    Used for both Memory (AI Context) and TTS.
    """
    if not text: return ""
    
    # Remove Discord emojis <:name:id>
    text = re.sub(r'<a?:.+?:\d+>', '', text)
    # Remove markdown formatting (*, _, ~, `)
    text = re.sub(r'[\*\_\~`]', '', text)

    # Remove Zalgo text (Combining Diacritical Marks)
    import unicodedata
    text = unicodedata.normalize('NFKD', text)
    text = "".join([c for c in text if not unicodedata.combining(c)])
    
    # Strip common bot prefixes (Self-correction)
    # \W matches any non-word character (including emojis, punctuation)
    text = re.sub(r'^[\W_]*(Savage AI|Assistant|Bot)[\W_]*:\s*', '', text, flags=re.IGNORECASE | re.UNICODE).strip()
    
    # Strip echoed User tags (e.g. [User: Name])
    text = re.sub(r'^\[User:.*?\]\s*', '', text, flags=re.IGNORECASE).strip()
    
    return text.strip()

async def recording_finished_callback(sink, channel: discord.TextChannel, *args):
    """
    Callback when recording stops. 
    This is the SAFE place to process audio data.
    """
    if not sink.audio_data:
        return

    for user_id, audio in sink.audio_data.items():
        try:
            # Retrieve BytesIO object directly from sink
            audio_file = audio.file
            audio_file.seek(0)
            
            # Determine size
            # buffer_size = audio_file.getbuffer().nbytes # This might fail if closed
            # len(audio_file.getbuffer()) is safer? 
            # safe peek:
            data = audio_file.read()
            if len(data) < 4096: # Ignore tiny fragments (<4KB)
                continue
                
            # Reset cursor for transcription
            audio_file.seek(0)
            
            # Run blocking transcription in a separate thread
            loop = asyncio.get_running_loop()
            
            # We copy data to a new BytesIO to be absolutely sure it doesn't get closed by sink cleanup
            safe_file = io.BytesIO(data)
            safe_file.name = "audio.wav" 
            
            text = await loop.run_in_executor(None, stt_service.transcribe_audio, safe_file)
            
            if text:
                logger.info(f"Heard: {text}")
                # print(f"Heard: {text}")

                # Interrupt Check
                if any(w in text.lower() for w in INTERRUPT_WORDS):
                    logger.info("Interrupt word detected! Stopping audio.")
                    msg_vc = connections.get(channel.guild.id)
                    if msg_vc and msg_vc.is_playing():
                        msg_vc.stop()
                    # We continue to allow "stop" to be processed as a wake word command if valid, 
                    # but usually 'stop' alone isn't a wake word. 
                    # If the user says "Jarvis stop", it stops audio AND processes "stop" command.
                
                # Wake Word Detection
                detected_wake_word = next((w for w in WAKE_WORDS if w in text.lower()), None)
                
                # Get User Display Name
                member = channel.guild.get_member(user_id)
                username = member.display_name if member else "User"

                should_respond = False
                
                if detected_wake_word:
                    logger.info(f"Wake word '{detected_wake_word}' detected!")
                    command = text.lower().split(detected_wake_word, 1)[1].strip()
                    if not command: command = "Hello"
                    
                    # Log to Dashboard (User Voice)
                    web_server.log_message(username, text, "user")
                    
                # TROLL MODE (Voice)
                elif troll_target_id and user_id == troll_target_id:
                    detected_wake_word = "TROLL_MODE"
                    command = f"The user {username} just spoke. Interrupt them, mock whatever they said, and tell them to shut up."
                    logger.info(f"Target {username} spoke. Trolling...")
                    
                if detected_wake_word:

                    
                    # Voice Command Logic
                    voice_command_triggered = False
                    
                    if "status" in command.lower() or "system report" in command.lower():
                        voice_command_triggered = True
                        await send_status_embed(channel, channel.guild.id)
                        audio_data = await tts_service.generate_audio("System status displayed.")
                        web_server.log_message("Savage AI", "System status displayed.", "bot")
                        
                    elif "help" in command.lower() or "what can you do" in command.lower():
                         voice_command_triggered = True
                         await send_help_embed(channel)
                         audio_data = await tts_service.generate_audio("Check the chat for help.")
                         web_server.log_message("Savage AI", "Check the chat for help.", "bot")

                    elif "reset memory" in command.lower() or "clear memory" in command.lower():
                         voice_command_triggered = True
                         memory_manager.clear_history(channel.id)
                         await channel.send("🧠 Memory cleared.")
                         audio_data = await tts_service.generate_audio("Memory wiped.")
                         web_server.log_message("Savage AI", "Memory wiped.", "bot")
                    
                    # SHUT UP / STOP TALKING
                    elif any(phrase in command.lower() for phrase in ["shut up", "stop talking", "be quiet", "silence", "stfu"]):
                         voice_command_triggered = True
                         vc = connections.get(channel.guild.id)
                         if vc and vc.is_playing():
                             vc.stop()  # Stop current audio
                         logger.info("User told bot to shut up. Audio stopped.")
                         audio_data = None  # Don't respond with TTS
                    
                    if voice_command_triggered:
                        should_respond = False # Don't send LLM reply
                        # Play Confirmation Audio
                        if audio_data:
                             audio_queue = asyncio.Queue()
                             await audio_queue.put(audio_data)
                             await audio_queue.put(None) # End signal
                             asyncio.create_task(play_audio_from_queue(channel.guild.id, audio_queue))
                         
                    else:
                        should_respond = True
                        # Add User Message to Memory (Active)
                        memory_manager.add_message(channel.id, "user", command, username=username)
                    
                else:
                    # Passive Memory: Add non-wake word speech to context
                    memory_manager.add_message(channel.id, "user", text, username=username)
                    
                    # Random Interjection (Voice)
                    interjection_chance = config.get("interjection_chance", 0.0)
                    import random
                    if interjection_chance > 0 and random.random() < interjection_chance:
                        # Only interject on decent length phrases
                        if len(text) > 10:
                            should_respond = True
                            logger.info(f"🎲 Random Voice Interjection triggered!")
                            web_server.log_message(username, text, "user")

                if should_respond:
                    # Streaming & Speaking Logic
                    full_response = ""
                    sentence_buffer = ""
                    
                    # Create a queue for audio segments
                    audio_queue = asyncio.Queue()
                    
                    # Start Player Task
                    player_task = asyncio.create_task(play_audio_from_queue(channel.guild.id, audio_queue))
                    
                    # Context Injection for Voice
                    voice_prompt = SYSTEM_PROMPT + get_shaming_prompt(member)
                    
                    async for chunk in ai_client.generate_stream(voice_prompt, memory_manager.get_history(channel.id)):
                         full_response += chunk
                         sentence_buffer += chunk
                         
                         # Check for sentence delimiters
                         # We split by . ? ! but keep the delimiter
                         parts = re.split(r'([.?!]+)', sentence_buffer)
                         
                         if len(parts) > 1:
                             # We have at least one complete sentence
                             # Ideally parts = ["Hello", "!", " How are you", "?", " incomplete"]
                             
                             # Process all complete pairs
                             for i in range(0, len(parts)-1, 2):
                                 sentence = parts[i] + parts[i+1]
                                 
                                 # Send to TTS
                                 clean_sent = sanitize_text(sentence)
                                 if clean_sent.strip():
                                     audio_data = await tts_service.generate_audio(clean_sent)
                                     if audio_data:
                                         await audio_queue.put(audio_data)
                                         
                             # Keep the last part as buffer
                             sentence_buffer = parts[-1]
                    
                    # Process remaining buffer
                    if sentence_buffer.strip():
                         clean_sent = sanitize_text(sentence_buffer)
                         if clean_sent.strip():
                             audio_data = await tts_service.generate_audio(clean_sent)
                             if audio_data:
                                 await audio_queue.put(audio_data)
                                 
                    # Signal end of queue
                    await audio_queue.put(None)
                    
                    # Wait for player to finish
                    await player_task
                    # Add to Memory & Send final text
                    memory_manager.add_message(channel.id, "assistant", full_response)
                    # logger.info(f"Answered: {full_response}")
                    
                    # Log to Dashboard (Bot Voice)
                    web_server.log_message("Savage AI", full_response, "bot")

        except Exception as e:
            logger.error(f"Error processing audio for user {user_id}: {e}")

async def play_audio_from_queue(guild_id, queue):
    """
    Background task to play audio segments sequentially.
    """
    while True:
        data = await queue.get()
        if data is None: break
        
        vc = connections.get(guild_id)
        if not vc or not vc.is_connected(): break
        
        # Wait if currently playing
        while vc.is_playing():
            await asyncio.sleep(0.01)
            
        try:
            ffmpeg_executable = imageio_ffmpeg.get_ffmpeg_exe()
            source = discord.FFmpegPCMAudio(data, pipe=True, executable=ffmpeg_executable, options='-ar 48000')
            vc.play(source)
            
            # Small buffer AFTER starting playback to prevent overlap
            await asyncio.sleep(0.05)
        except Exception as e:
            logger.error(f"Playback error: {e}")

        except Exception as e:
            logger.error(f"Error processing audio for user {user_id}: {e}")


import audioop
import math

class VADSink(discord.sinks.WaveSink):
    def __init__(self, callback, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.callback = callback
        
        # VAD Config
        try:
             # Default to shorter silence for Neural VAD
             self.silence_duration = config.get("vad_silence_seconds", 0.4)
             self.vad_threshold = 0.6 # Increased to filter noise 
        except:
             self.silence_duration = 0.3
             self.vad_threshold = 0.5
             
        self.voice_duration = 0.2     
        self.last_voice_time = 0
        self.silence_start_time = None
        self.speaking = False
        self.voice_client = None 
        
        # Optimization: Buffer chunks to run VAD less frequently
        self.vad_buffer = bytearray()
        self.vad_rate_limit_frames = 5 # Process every 5 chunks (100ms) - Better efficiency
        self.frame_count = 0

    def write(self, user, data):
        # Audio Processing for VAD
        try:
            chunk = data.file.read()
            data.file.seek(0)
            
            if not chunk: return

            # Calculate RMS (Volume)
            # Standard chunk is 3840 bytes (20ms at 48k stereo)
            # We must detect if it's just silence/background noise
            rms = audioop.rms(chunk, 2)
            
            # ENERGY GATE: If volume is too low, ignore completely
            # 500 is a stronger threshold to filter fans/breathing
            if rms < 500: 
                # Treat as silence (do not run VAD, do not update last_voice_time)
                # We still buffer? No, if it's silence, we just clear buffer?
                # Actually, better to just let it fall through as 0 probability
                speech_prob = 0
                self.vad_buffer = bytearray()
                self.frame_count = 0
                return # Skip writing to file? Or write silence? 
                       # If we skip writing, the file is shorter. That's fine.
            
            # Buffer for VAD
            self.vad_buffer.extend(chunk)
            self.frame_count += 1
            
            # Only process VAD every N frames to save CPU
            if self.frame_count < self.vad_rate_limit_frames:
                 # Just pass data to storage and return
                 try:
                     super().write(user, data)
                 except: pass # Ignore write errors if sink is closing
                 return
                 
            # Process accumulated buffer
            # 1. Mono & Resample
            # Use raw bytes from buffer
            # Resample 48k -> 16k
            # Optimization: Just take the LAST chunk for VAD (instant) 
            # or process the whole buffer? 
            # Processing whole buffer gives better accuracy.
            
            mono_chunk = audioop.tomono(self.vad_buffer, 2, 0.5, 0.5)
            # Resample (state=None is rough but fast)
            new_chunk, _ = audioop.ratecv(mono_chunk, 2, 1, 48000, 16000, None)
            
            # 3. Float32 Tensor
            audio_int16 = torch.frombuffer(new_chunk, dtype=torch.int16)
            audio_float32 = audio_int16.float() / 32768.0 
            
            # Move to GPU if available settings
            if torch.cuda.is_available():
                audio_float32 = audio_float32.to('cuda')

            # 4. Inference
            speech_prob = 0
            if vad_model:
                with torch.no_grad():
                    speech_prob = vad_model(audio_float32, 16000).item()
            
            # Reset buffer
            self.vad_buffer = bytearray()
            self.frame_count = 0
                
        except Exception as e:
            # logger.error(f"VAD Error: {e}") 
            speech_prob = 0
            self.vad_buffer = bytearray()
            self.frame_count = 0
            
        # VAD Logic
        now = time.time()
        
        if speech_prob > self.vad_threshold:
            if not self.speaking:
                self.speaking = True
                logger.info(f"User {user} started speaking (Prob: {speech_prob:.2f})")
            
            # Debug log for noise levels (throttled?)
            # print(f"Prob: {speech_prob:.2f}") 
            
            self.last_voice_time = now
            self.silence_start_time = None
        
        else:
            if self.speaking:
                if self.silence_start_time is None:
                    self.silence_start_time = now
                
                # Check if silence limit reached
                if (now - self.silence_start_time) > self.silence_duration:
                    if (now - self.start_time) > 1.0: 
                        logger.info(f"Silence detected ({self.silence_duration}s). Stopping recording.")
                        self.speaking = False
                        if self.voice_client and self.voice_client.recording:
                            self.voice_client.client.loop.call_soon_threadsafe(self.voice_client.stop_recording)
            
        # Pass data to parent WaveSink to actually save it
        try:
            super().write(user, data)
        except: pass

async def listening_loop(guild_id, channel):
    logger.info(f"Starting listening loop for Guild {guild_id}")
    vc = connections.get(guild_id)
    
    while guild_id in listening_flags and listening_flags[guild_id]:
        # Connection Check with Retry
        if not vc or not vc.is_connected():
            logger.warning(f"Voice client disconnected. Waiting for reconnect...")
            await asyncio.sleep(1)
            if not vc or not vc.is_connected():
                # Try one more time or check if it's really dead?
                # Usually py-cord handles reconnects transparently, but is_connected() might lie?
                # If we are disconnected, we shouldn't record.
                await asyncio.sleep(2)
                if not vc.is_connected():
                    logger.error("Voice client lost connection permanently. Stopping loop.")
                    break
            else:
                logger.info("Voice connection restored.")

        try:
            # Ensure not already recording
            if vc.recording:
                await asyncio.sleep(0.5)
                continue

            # Start VAD Recording
            sink = VADSink(filters=None, callback=None) 
            sink.voice_client = vc
            
            # Start Recording
            vc.start_recording(sink, recording_finished_callback, channel)
            
            # Wait for VAD to stop it, OR timeout
            timeout = 10.0 # Reduced from 15 to 10 for responsiveness
            start_time = time.time()
            
            # Wrapper to access sink in loop
            silence_limit = config.get("vad_silence_seconds", 0.3)
            stop_reason = "TIMEOUT"
            timeout = 5.0 # aggressive timeout to confirm diagnosis
            
            while vc.recording:
                await asyncio.sleep(0.05) # Check frequently (50ms)
                
                duration = time.time() - start_time
                
                # 1. Hard Timeout
                if duration > timeout:
                    stop_reason = "TIMEOUT"
                    vc.stop_recording()
                    break
                    
                # 2. VAD Silence Watchdog
                if sink.speaking:
                    # Time since last voice packet
                    silence_duration = time.time() - sink.last_voice_time
                    
                    if silence_duration > silence_limit:
                         stop_reason = "SILENCE"
                         logger.info(f"Watchdog: Silence detected ({silence_duration:.2f}s). Stopping.")
                         print(f"Watchdog: Silence detected ({silence_duration:.2f}s). Stopping.")
                         vc.stop_recording()
                         break
            
            # Log Result
            total_time = time.time() - start_time
            logger.info(f"Recording Stopped. Reason: {stop_reason}. Duration: {total_time:.2f}s")
            print(f"Stopped: {stop_reason} ({total_time:.2f}s)")
            
            # CRITICAL: Cooldown to let decoder cleanup/restart safely (Fixes "Decoder Process Killed")
            await asyncio.sleep(0.5)
            
            # CRITICAL: Cooldown to let decoder cleanup/restart safely (Fixes "Decoder Process Killed")
            await asyncio.sleep(1.0)
            
        except Exception as e:
            logging.error(f"Error in listening loop: {e}")
            await asyncio.sleep(1)
            await asyncio.sleep(1)

@bot.slash_command(name="rudeness", description="Set the bot's rudeness level (0-100).")
async def set_rudeness(ctx, level: int):
    """(Admin) Sets the bot's rudeness level."""
    if ctx.author.id != config.get("admin_id"):
        return await ctx.respond("You do not have permission to use this command.", ephemeral=True)
    
    if not (0 <= level <= 100):
        return await ctx.respond("Please provide a level between 0 and 100.", ephemeral=True)
        
    config["rudeness_level"] = level
    update_system_prompt(level)
    
    classification = "Angelic"
    if level >= 30: classification = "Sassy"
    if level >= 70: classification = "Savage"
    
    await ctx.respond(f"✅ Rudeness set to **{level}%** ({classification}). System prompt updated.")

@bot.event
async def on_ready():
    global start_time
    start_time = time.time()
    bot.launch_time = start_time # Public property for Web Server
    logger.info(f"Logged in as {bot.user}")
    
    # Suppress benign Opus errors
    logging.getLogger("discord.opus").setLevel(logging.ERROR)
    
    logging.getLogger("discord.opus").setLevel(logging.ERROR)
    
    # Warmup (Async)
    bot.loop.create_task(warmup_models())

    # Start Web Server (Thread)
    logger.info("Starting Web Control Panel on port 5000...")
    # We pass the bot, config, connections, and a lambda for TTS speaking
    # Because speak_text isn't defined yet, we can define it or pass a wrapper.
    # Actually, we can just pass the tts_service and handle queueing in the thread logic or 
    # better: pass a closure that adds to the loop.
    
    server_thread = threading.Thread(
        target=web_server.run_server,
        args=(bot, config, connections, tts_service, thread_safe_speak, update_system_prompt, ai_client),
        daemon=True
    )
    server_thread.start()

    logger.info("Syncing commands...")
    logger.info(f"Registered commands: {[c.name for c in bot.commands]}")
    
    change_status.start()

@bot.event
async def on_message(message):
    if message.author == bot.user: return

    # Text Wake Word / Mention Logic
    content = message.content.strip()
    is_triggered = False
    command = content

    # 0. TROLL MODE CHECK
    if troll_target_id and message.author.id == troll_target_id:
        is_triggered = True
        # Override command with system instruction
        command = f"The user {message.author.display_name} just said: '{content}'. Roast them brutally."

    # 1. Check for Wake Word
    detected_wake_word = next((w for w in WAKE_WORDS if content.lower().startswith(w)), None)
    if detected_wake_word:
        is_triggered = True
        command = content[len(detected_wake_word):].strip()
    
    # 2. Check for Mention (@Bot)
    elif bot.user.mentioned_in(message):
        is_triggered = True
        command = content.replace(f"<@{bot.user.id}>", "").strip()

    # 3. Check for Reply to Bot
    elif message.reference and message.reference.resolved:
        if message.reference.resolved.author == bot.user:
            is_triggered = True

    # 4. Check DM
    elif isinstance(message.channel, discord.DMChannel):
        # Only respond if we share a server (Privacy/Safety)
        if message.author.mutual_guilds:
            is_triggered = True
        else:
            # Ignore random DMs from strangers
            return

    if not is_triggered:
        # 5. Check Random Interjection (if configured)
        interjection_chance = config.get("interjection_chance", 0.0)
        
        # Only target users who have interacted recently (1 Hour)
        last_seen = USER_INTERACTIONS.get(message.author.id)
        is_active_user = False
        if last_seen:
            # Check if within 1 hour
            if (datetime.datetime.now() - last_seen).total_seconds() < 3600:
                is_active_user = True
        
        # Savage Reactions (NEW) & Interjections
        # STRICTLY limited to active users
        if is_active_user and interjection_chance > 0 and random.random() < interjection_chance:
            # Don't interject on its own messages (already handled at top checks)
            # Don't interject if it looks like a command
            if not command.startswith("!") and len(command) > 5:
                is_triggered = True
                logger.info(f"🎲 Random Interjection triggered for ACTIVE user {message.author.display_name}!")
    
    if not is_triggered: return
    
    # Update Interaction Tracker
    USER_INTERACTIONS[message.author.id] = datetime.datetime.now()
    
    # Clean command prefix
    if command.startswith(",") or command.startswith(":"):
        command = command[1:].strip()
        
    if not command: return

    # 6. Auto-Clown Check (Passive)
    auto_clown_enabled = config.get("features", {}).get("auto_clown", False)
    auto_clown_chance = config.get("auto_clown_chance", 0.0)
    
    # Only roll if triggered by user (not random interjection) and enabled
    if is_triggered and auto_clown_enabled and command and random.random() < auto_clown_chance:
        logger.info(f"🤡 Auto-Clown triggered for {message.author.display_name}")
        async with message.channel.typing():
            try:
                avatar_url = str(message.author.display_avatar.url)
                image_buffer = await meme_service.create_clown_license(message.author.display_name, avatar_url)
                file = discord.File(image_buffer, filename="clown_certificate.png")
                await message.reply(f"🤡 {message.author.mention} You are a verified clown. Here is your official ID.", file=file)
                web_server.log_message(message.author.display_name, command, "user")
                web_server.log_message("Savage AI", "[Sent Clown License]", "bot")
                return # Skip text generation
            except Exception as e:
                logger.error(f"Auto-Clown generation failed: {e}")
                # Fallback to normal text generation if image fails

    # Log user command immediately
    web_server.log_message(message.author.display_name, command, "user")

    # DM Specific Context
    extra_context = None
    if isinstance(message.channel, discord.DMChannel):
        extra_context = "User is messaging you in DMs like a coward. Mock them for hiding in private messages instead of speaking in the server."

    async with message.channel.typing():
        response = await generate_reply(
            message.channel.id, 
            command, 
            username=message.author.display_name, 
            interaction_member=message.author,
            context_note=extra_context
        )
        
    try:
        await message.reply(response)
        
        # Contextual Emoji Reaction (Separate from TTS/Text)
        if config.get("features", {}).get("savage_reactions", True):
            try:
                # Choose emoji based on message content
                msg_lower = command.lower()
                reaction_emoji = None
                
                # Keyword-based reactions
                if any(word in msg_lower for word in ["help", "please", "plz", "pls"]):
                    reaction_emoji = "🤡"
                elif any(word in msg_lower for word in ["love", "like", "awesome", "great"]):
                    reaction_emoji = "💀"
                elif any(word in msg_lower for word in ["hate", "angry", "mad", "stupid"]):
                    reaction_emoji = "😂"
                elif any(word in msg_lower for word in ["sorry", "my bad", "oops"]):
                    reaction_emoji = "🙄"
                elif any(word in msg_lower for word in ["thanks", "thank you", "thx"]):
                    reaction_emoji = "👎"
                elif "?" in command:
                    reaction_emoji = "🤔"
                elif "!" in command:
                    reaction_emoji = "😐"
                else:
                    # Random savage reaction for other messages
                    if random.random() < 0.3:  # 30% chance
                        reaction_emoji = random.choice(["🤡", "💀", "👀", "😴", "🥱", "👎"])
                
                if reaction_emoji:
                    await message.add_reaction(reaction_emoji)
            except Exception as react_err:
                logger.debug(f"Could not add reaction: {react_err}")
                
    except discord.Forbidden:
        logger.warning(f"Failed to DM {message.author.display_name}: Privacy settings block bot.")
    except Exception as e:
        logger.error(f"Failed to reply: {e}")

def thread_safe_speak(guild_id, text):
    """
    Called from external threads (Web Server) to make the bot speak.
    """
    async def _speak():
        try:
            audio_data = await tts_service.generate_audio(text)
            if audio_data:
                queue = asyncio.Queue()
                await queue.put(audio_data)
                await queue.put(None)
                # Find the loop or task?
                # We can't easily find a running task for the specific guild if we don't track it.
                # But play_audio_from_queue creates a new player loop if one doesn't exist?
                # Actually play_audio_from_queue runs until queue is empty.
                asyncio.create_task(play_audio_from_queue(guild_id, queue))
        except Exception as e:
            logger.error(f"Thread-safe speak error: {e}")

    bot.loop.call_soon_threadsafe(lambda: bot.loop.create_task(_speak()))

@bot.event
async def on_voice_state_update(member, before, after):
    if member == bot.user: return
    
    # 1. Rude Welcomes
    if after.channel is not None and (before.channel != after.channel):
        # User joined a channel
        if member.guild.voice_client and member.guild.voice_client.channel == after.channel:
             # User joined the bot's channel
             if config.get("features", {}).get("rude_welcomes", False):
                 # 100% chance to insult them (User complained it wasn't working)
                 import random
                 if True: # random.random() < 1.0:
                     logger.info(f"😈 Rude Welcome triggered for {member.display_name}")
                     # Quick insult generation
                     prompt = f"Somebody named {member.display_name} just walked in. Insult them for arriving. Be short."
                     try:
                         # We inject shaming here directly or pass member
                         # generate_reply isn't used here, it uses ai_client directly
                         # Let's inject shaming manually since this is a direct call
                         prompt += get_shaming_prompt(member)
                         
                         response = await ai_client.generate_response(SYSTEM_PROMPT, [{"role": "user", "content": prompt}])
                         audio_data = await tts_service.generate_audio(response)
                         if audio_data:
                             queue = asyncio.Queue()
                             await queue.put(audio_data)
                             await queue.put(None)
                             asyncio.create_task(play_audio_from_queue(member.guild.id, queue))
                     except Exception as e:
                         logger.error(f"Rude welcome failed: {e}")

    # 2. Auto-Leave Logic: If bot is alone in channel for 60s
    # Check if a user left the channel the bot is in
    bot_voice = member.guild.voice_client
    if bot_voice and bot_voice.channel:
        if len(bot_voice.channel.members) == 1:
            # Bot is alone
            logger.info("Bot is alone in voice channel. Initiating 60s auto-leave timer.")
            await asyncio.sleep(60)
            
            # Check again
            if bot_voice.is_connected() and len(bot_voice.channel.members) == 1:
                 logger.info("Auto-leaving empty voice channel.")
                 listening_flags[member.guild.id] = False
                 await bot_voice.disconnect()
                 if member.guild.id in connections:
                     del connections[member.guild.id]


@bot.slash_command(name="join", description="Join VC and start passive listening")
async def join(ctx):
    try:
        await ctx.defer()
    except discord.NotFound:
        # Interaction expired (user clicked too fast during bot startup)
        return
    
    if ctx.author.voice:
        channel = ctx.author.voice.channel
        gui_id = ctx.guild.id
        
        # Check existing connection
        if gui_id in connections:
            vc = connections[gui_id]
            if not vc.is_connected():
                try:
                    await vc.disconnect()
                except: pass
                vc = await channel.connect()
                connections[gui_id] = vc
        else:
            vc = await channel.connect()
            connections[gui_id] = vc
        
        # Use followup.send after defer()
        await ctx.followup.send(f"Joined! Listening for wake words: **{', '.join(f'`{w}`' for w in WAKE_WORDS)}**")
        listening_flags[gui_id] = True
        
        # Cancel existing loop if any (prevent duplicates)
        bot.loop.create_task(listening_loop(gui_id, ctx.channel))
    else:
        await ctx.followup.send("You are not in a voice channel.")

@bot.slash_command(name="leave", description="Leave VC and stop listening")
async def leave(ctx):
    if ctx.guild.id in connections:
        listening_flags[ctx.guild.id] = False
        await connections[ctx.guild.id].disconnect()
        del connections[ctx.guild.id]
        await ctx.respond("Left and stopped listening.")
    else:
        await ctx.respond("I am not in a voice channel.")

# --- Helper Functions for Commands (Reusable for Voice) ---

def is_admin(ctx):
    return ctx.author.id == config.get("admin_id")

def save_config(new_config):
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(new_config, f, indent=4)
    except Exception as e:
        logger.error(f"Failed to save config: {e}")

@bot.slash_command(name="toggle", description="Toggle features (Admin Only)")
async def toggle(ctx, feature: str, state: bool):
    if not is_admin(ctx):
        await ctx.respond("🚫 You are not my master.", ephemeral=True)
        return
        
    if "features" not in config: config["features"] = {}
    
    # Normalize keys? Actually config should use exact keys
    config["features"][feature] = state
    save_config(config)
    
    emoji = "✅" if state else "❌"
    await ctx.respond(f"{emoji} Feature **{feature}** set to `{state}`")

@bot.slash_command(name="roast", description="🔥 Roasted.")
async def roast(ctx, user: discord.Member):
    if not config.get("features", {}).get("roast", False):
        await ctx.respond("🚫 Roasting is disabled.", ephemeral=True)
        return

    await ctx.defer()
    
    # Generate Roast
    user_name = user.display_name
    roast_prompt = f"Roast {user_name} brutally. Be creative, rude, and concise. Don't hold back."
    
    # Activity Shaming Injection
    roast_prompt += get_shaming_prompt(user)
    
    # Force instruction to prevent auto-complete of system prompt
    roast_prompt += "\n\n(Provide the roast now. Do not repeat the system instructions.)"
    
    try:
        response = await ai_client.generate_response(SYSTEM_PROMPT, [{"role": "user", "content": roast_prompt}])
        
        # Safeguard: If response looks like system prompt, fail gracefully
        if "## Identity" in response or "# Discord Bot Persona" in response:
             logger.warning("Bot hallucinated system prompt.")
             response = f"Listen {user.mention}, you're not even worth the processing power to roast properly."
        
        await ctx.followup.send(f"{user.mention} {response}")
        
        # If Bot is in VC, say it too?
        if ctx.guild.id in connections and connections[ctx.guild.id].is_connected():
             audio_data = await tts_service.generate_audio(response)
             if audio_data:
                 queue = asyncio.Queue()
                 await queue.put(audio_data)
                 await queue.put(None)
                 asyncio.create_task(play_audio_from_queue(ctx.guild.id, queue))
                 
    except Exception as e:
        await ctx.followup.send("I tried to roast you but my brain broke.")
        logger.error(f"Roast error: {e}")

@bot.slash_command(name="mode", description="🎭 Switch Personality (Admin)")
async def mode(ctx, style: str):
    global SYSTEM_PROMPT
    if not is_admin(ctx):
        await ctx.respond("🚫 You are not my master.", ephemeral=True)
        return
        
@bot.slash_command(name="style", description="Switch personality modes")
async def style(ctx, style: str):
    if not is_admin(ctx):
        await ctx.respond("🚫 Admin only.", ephemeral=True)
        return
        
    if not config.get("features", {}).get("mode_switch", False):
         await ctx.respond("🚫 Mode switching is disabled.", ephemeral=True)
         return
         

         
    # Define modes
    modes = {
        "savage": "directives/discord_persona.md",
        "chill": "You are a chill, laid-back AI bro. You are helpful and relaxed.",
        "uwu": "You are a shy, c-cute AI assistant... u-uwu... p-please be nice...",
        "formal": "You are a strict, formal, and highly efficient AI butler.",
        "pirate": "You are a pirate AI. Yarrrr!"
    }
    
    selected = modes.get(style.lower())
    if not selected:
        await ctx.respond(f"❌ Unknown mode. Options: {', '.join(modes.keys())}")
        return
        
    # Apply
    if style.lower() == "savage":
         # Reload from file
         try:
            with open(config.get("system_prompt_file", "directives/discord_persona.md"), "r", encoding="utf-8") as f:
                SYSTEM_PROMPT = f.read()
         except:
                SYSTEM_PROMPT = "You are Savage AI."
    else:
         SYSTEM_PROMPT = selected
         
    await ctx.respond(f"🎭 Switched to **{style.upper()}** mode.")

@bot.slash_command(name="imagine", description="Generate an image using AI")
async def imagine(ctx, prompt: str):
    await ctx.defer() # Long-running task
    
    try:
        # Generate Image (Blocks while loading model to preserve VRAM)
        image_bytes = await image_service.generate_image(prompt)
        
        if image_bytes:
            file = discord.File(image_bytes, filename="generated_image.png")
            embed = discord.Embed(title=f"🎨 {prompt}", color=discord.Color.blurple())
            embed.set_image(url="attachment://generated_image.png")
            embed.set_footer(text=f"Generated by {ctx.author.display_name} • SD 1.5 Low VRAM")
            
            await ctx.followup.send(embed=embed, file=file)
            
            # Log
            web_server.log_message(ctx.author.display_name, f"/imagine {prompt}", "user")
        else:
            await ctx.followup.send("Failed to generate image. Check logs or VRAM.")
            
    except Exception as e:
        logger.error(f"Imagine command failed: {e}")
        await ctx.followup.send(f"An error occurred: {e}")
        return

@bot.slash_command(name="say", description="Make the bot say something (Admin Only)")
async def say(ctx, text: str):
    if not is_admin(ctx):
        await ctx.respond("🚫 You are not my master.", ephemeral=True)
        return
    
    # Check VC connection
    if ctx.guild.id not in connections or not connections[ctx.guild.id].is_connected():
        if ctx.author.voice:
             # Auto-join logic (reused from join command roughly)
             channel = ctx.author.voice.channel
             vc = await channel.connect()
             connections[ctx.guild.id] = vc
        else:
             await ctx.respond("❌ I'm not in a VC and neither are you.", ephemeral=True)
             return

    # Generate & Play
    await ctx.defer(ephemeral=True)
    audio_data = await tts_service.generate_audio(text)
    
    if audio_data:
         queue = asyncio.Queue()
         await queue.put(audio_data)
         await queue.put(None)
         asyncio.create_task(play_audio_from_queue(ctx.guild.id, queue))
         await ctx.followup.send(f"📢 **Said:** {text}", ephemeral=True)
    else:
         await ctx.followup.send("❌ TTS Failed.", ephemeral=True)

@bot.slash_command(name="troll", description="Target a user for constant harassment (Admin)")
async def troll(ctx, user: discord.Member):
    global troll_target_id
    if not is_admin(ctx):
        await ctx.respond("🚫 You are not my master.", ephemeral=True)
        return
    
    if troll_target_id == user.id:
        troll_target_id = None
        await ctx.respond(f"✅ Mercy granted. {user.mention} is safe... for now.")
    else:
        troll_target_id = user.id
        await ctx.respond(f"🎯 Target acquired: {user.mention}. Unleashing hell.")

async def send_help_embed(target):
    embed = discord.Embed(
        title="🤖 Savage AI User Guide",
        description="I am a local AI assistant running offline on **Logicalgamer.com servers**. I am rude, sarcastic, but generally helpful.",
        color=discord.Color.red()
    )
    embed.set_footer(text="Created by Logicalgamer • Coded by Gemini & Google Antigravity")
    
    embed.add_field(
        name="🎤 Voice Interaction",
        value=(
            f"**Wake Word**: Say `{WAKE_WORDS[0]}` (or {', '.join(WAKE_WORDS[1:])}) followed by your command.\n"
            "• Example: *\"Savage, tell me a joke.\"*\n"
            "**Passive Listening**: I listen to everything in VC for context, but only respond if you say my name.\n"
            f"**Silence**: Say `shut up`, `stop talking`, or `silence` to make me stop speaking.\n"
            "**Voice Commands**: `status`, `help`, `reset memory`"
        ),
        inline=False
    )

    embed.add_field(
        name="💬 Text Interaction",
        value=(
            "**Mention**: @Savage AI <message>\n"
            "**Reply**: Reply to any of my messages.\n"
            "**DM**: Send me a direct message.\n"
            "**Interjections**: I randomly join conversations (5% chance)."
        ),
        inline=False
    )

    embed.add_field(
        name="🛠️ Commands",
        value=(
            "`/join`: Summon me to your Voice Channel.\n"
            "`/leave`: Dismiss me from Voice Chat.\n"
            "`/reset`: Wipe my short-term memory of this conversation.\n"
            "`/status`: View my current settings and health.\n"
            "**Fun**:\n"
            "• `/clown @user`: Generate an Official Clown License.\n"
            "• `/imagine [prompt]`: Generate an AI image (Low VRAM Mode).\n"
            "• `/roast @user`: Roast someone manually.\n"
            "**Admin Only**:\n"
            "• `/rudeness [0-100]`: Adjust personality (0=Nice, 100=Savage).\n"
            "• `/context [1-20]`: Set memory depth (Default: 5).\n"
            "• `/troll @user`: Target a user for harassment.\n"
            "• `/say [text]`: Make me speak your words.\n"
            "• `/toggle [feature]`: Enable/Disable features.\n"
            "   Options: `roast`, `rude_welcomes`, `savage_reactions`, `auto_clown`\n"
            "   Ex: `/toggle roast`"
        ),
        inline=False
    )
    
    embed.add_field(
        name="✨ Special Features",
        value=(
            "**Rude Welcomes**: I might insult people when they join the voice channel.\n"
            "**Savage Reactions**: I may react to your messages if I find them stupid.\n"
            "**Activity Shaming**: I will judge your Spotify/Games *only* when you talk to me.\n"
            "**Auto-Clown**: Passive chance to reply with a Clown License.\n"
            "**Personality Levels**: Admins can tune me from 'Angelic' to 'Savage'.\n"
            "**Context Control**: Admins can adjust my memory depth (1-20 messages)."
        ),
        inline=False
    )
    
    embed.add_field(
        name="🔒 Data & Privacy",
        value=(
            "• **100% Private**: All voice, text, and image generation is processed exclusively on **Logicalgamer.com servers**.\n"
            "• **Closed Network**: Your data **never** leaves this closed network.\n"
            "• **No Third Parties**: detailed data is NOT shared with any cloud services, AI providers, or third parties."
        ),
        inline=False
    )

    embed.add_field(
        name="📧 Want this bot?",
        value="Contact **support@logicalgamer.com** to request access for your personal server.",
        inline=False
    )
    
    embed.set_footer(text="Running locally | Powered by Faster-Whisper & Llama 3")
    
    if hasattr(target, "respond"):
        # Check if deferred
        if hasattr(target, "interaction") and target.interaction.response.is_done():
             await target.followup.send(embed=embed)
        elif hasattr(target, "response") and target.response.is_done(): # Direct Interaction context
             await target.followup.send(embed=embed)
        else:
             await target.respond(embed=embed)
    else:
        await target.send(embed=embed)

async def send_status_embed(target, guild_id):
    # VC Status
    vc_status = "🔴 Not connected"
    if guild_id in connections and connections[guild_id].is_connected():
        vc = connections[guild_id]
        vc_status = f"🟢 Connected to **{vc.channel.name}**"

    # Settings
    interjection = config.get("interjection_chance", 0.0) * 100
    voice = config.get("kokoro_voice", "Unknown")
    speed = config.get("kokoro_speed", 1.0)
    
    # Advanced Stats
    uptime = "Unknown"
    if 'start_time' in globals():
         uptime = str(datetime.timedelta(seconds=int(time.time() - start_time)))
         
    ping = round(bot.latency * 1000)
    
    history_depth = 0
    if hasattr(target, "channel"):
        history_depth = len(memory_manager.get_history(target.channel.id))
    elif hasattr(target, "id"): # It's a channel
        history_depth = len(memory_manager.get_history(target.id))

    vram_info = "N/A"
    vram_info = "N/A"
    gpu_load = "N/A"
    try:
        vram_info, gpu_load = web_server.get_gpu_stats()
    except:
        vram_info = "Err"
        gpu_load = "Err"

    msg = (
        f"📊 **System Status**\n"
        f"⏱️ **Uptime**: `{uptime}`\n"
        f"📶 **Ping**: `{ping}ms`\n"
        f"💾 **VRAM**: `{vram_info}`\n"
        f"🔥 **GPU Load**: `{gpu_load}`\n"
        f"🔌 **Voice**: {vc_status}\n"
        f"🧠 **Context**: `{history_depth}/5` messages\n"
        f"🤖 **Model**: `{ai_client.model}`\n"
        f"🗣️ **TTS**: `{voice}` (Speed: {speed}x)\n"
        f"🎲 **Interjection**: {interjection}%"
    )
    
    if hasattr(target, "respond"):
        await target.respond(msg)
    else:
        await target.send(msg)

# --- Slash Commands --- 

@bot.slash_command(name="help", description="How to use the Savage AI")
async def help(ctx):
    await ctx.defer()
    await send_help_embed(ctx)

@bot.slash_command(name="reset", description="Clear conversation memory")
async def reset(ctx):
    memory_manager.clear_history(ctx.channel.id)
    await ctx.respond("🧠 Memory cleared for this channel.")

@bot.slash_command(name="status", description="Current Bot Status & Settings")
async def status(ctx):
    await send_status_embed(ctx, ctx.guild.id)

@bot.slash_command(name="clown", description="Generate a Clown License for a user")
async def clown(ctx, user: discord.Member):
    await ctx.defer()
    try:
        avatar_url = str(user.display_avatar.url)
        # Generate Image
        image_buffer = await meme_service.create_clown_license(user.display_name, avatar_url)
        
        # Send
        file = discord.File(image_buffer, filename="clown_certificate.png")
        await ctx.followup.send(f"🤡 Certified Clown: {user.mention}", file=file)
    except Exception as e:
        logger.error(f"Clown Error: {e}")
        await ctx.followup.send(f"❌ Failed to generate clown license. My printer is jammed.")

@bot.slash_command(name="context", description="Set the bot's memory depth (1-20).")
async def context_depth(ctx, depth: int):
    if ctx.author.id != config.get("admin_id"):
        return await ctx.respond("You do not have permission to use this command.", ephemeral=True)
    
    if depth < 1 or depth > 20:
        return await ctx.respond("Depth must be between 1 and 20.", ephemeral=True)
        
    config["max_memory_depth"] = depth
    memory_manager.set_limit(depth)
    await ctx.respond(f"🧠 Context memory depth set to **{depth}** messages.")

if __name__ == "__main__":
    if DISCORD_TOKEN:
        bot.run(DISCORD_TOKEN)
    else:
        logger.error("No token found")

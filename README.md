# 🤖 Savage AI (Discord Bot)

> **⚠️ VIBE CODED PROJECT**
> This project was built using "Vibe Coding" (Agentic AI). The code was generated iteratively to meet specific vibes and functionality. It may contain quirks, unconventional structures, or unoptimized patterns. Use at your own risk!

Savage AI is a locally-hosted, voice-enabled Discord bot with a distinct personality. It doesn't just help—it judges.

## ✨ Features

### 🎤 **Voice Interaction**
*   **Real-Time VAD**: Uses Silero VAD to detect when you are speaking.
*   **Wake Word Detection**: Responds to "Savage", "Assistant", "AI".
*   **TTS (Text-to-Speech)**: High-quality, low-latency audio using **Kokoro-82M**.
*   **STT (Speech-to-Text)**: Fast transcription using **Faster-Whisper**.
*   **Voice Commands**: Tell the bot to "shut up", "reset memory", or check "status".

### 😈 **Personality Engine**
*   **Dynamic Rudeness**: Admin-controlled rudeness level (0-100%).
*   **Active Shaming**: Mocks users for playing League of Legends, Roblox, etc.
*   **Rude Welcomes**: Insults users when they join the voice channel.
*   **Savage Reactions**: Reacts to messages with emojis based on sentiment (🤡, 💀, 🙄).
*   **Auto-Clown**: Randomly replies with an "Official Clown License".

### 🖼️ **Creative & Utility**
*   **Image Generation**: Generate AI art locally (`/imagine`) using Stable Diffusion (Low VRAM mode).
*   **Dashboard**: Full web-based control panel to toggle features, monitor stats, and switch models.
*   **Memory**: Context-aware conversations with configurable depth.

---

## 🚀 Installation

### 1. Prerequisites
*   Windows OS (Preferred)
*   Python 3.10+
*   NVIDIA GPU (Recommended for TTS/STT/Image Gen)
*   [FFmpeg](https://ffmpeg.org/download.html) installed and added to PATH.
*   [LM Studio](https://lmstudio.ai/) (or any OpenAI-compatible Local AI server) running.

### 2. Setup
1.  **Clone or Download** this repository.
2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: You may need to install PyTorch manually depending on your CUDA version)*
3.  **Configuration**:
    *   Rename `.env.example` to `.env` and add your Discord Token.
    *   Edit `config.json` to tweak settings (if desired).
4.  **Start Local AI**:
    *   Open LM Studio, load a model (e.g., Llama 3), and start the server on port `1234`.

### 3. Run
Double-click `start_bot.bat` or run:
```bash
python execution/discord_bot.py
```

---

## 🛠️ Configuration

**`.env`**:
```ini
DISCORD_TOKEN=your_token_here
LOCAL_AI_BASE_URL=http://localhost:1234/v1
```

**`config.json`**:
*   `rudeness_level`: 0 (Nice) to 100 (Savage)
*   `features`: Toggle specific modules like `roast`, `auto_clown`, etc.

---

## ⚠️ Disclaimer
This bot is designed to be **rude/funny**. Ensure your server members are okay with a bit of "savage" humor. Features like "Activity Shaming" and "Rude Welcomes" can be disabled in the Dashboard.

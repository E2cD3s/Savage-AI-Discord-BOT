# 🤖 Savage AI (Discord Bot)

> **⚠️ VIBE CODED PROJECT**
> This project was built using "Vibe Coding" (Agentic AI). The code was generated iteratively to meet specific vibes and functionality. It may contain quirks, unconventional structures, or unoptimized patterns. Use at your own risk!

Savage AI is a **fully local**, **voice-enabled** Discord bot that combines modern AI audio stacks with a dynamic, rude personality engine. It doesn't just assist you—it judges you.

---

## ✨ Features Overview

### 🎤 **Advanced Voice Interaction**
The bot acts as a user in Voice Chat, not just a music bot.
*   **Real-Time VAD (Voice Activity Detection)**: Uses **Silero VAD** to detect speech with a 0.5s safety buffer, allowing for natural pauses without cutting the user off.
*   **Passive Listening**: The bot listens to context but only responds when triggered (Wake Word or Interjection).
*   **Wake Word Detection**: Responds to **"Savage"**, **"Assistant"**, or **"AI"**.
*   **Interjections**: Configurable (5%) chance to randomly jump into conversations if it "hears" something stupid.
*   **Silence Commands**: Immediately stops talking when you say "Shut up", "Silence", or "Stop talking".
*   **Local STT**: Transcribes audio using **Faster-Whisper (`small.en`)** with Beam Size 5 for high accuracy (CUDA accelerated).
*   **Local TTS**: Generates speech using **Kokoro-82M** (v1.0), achieving ~40ms latency on GPU.

### 😈 **Personality Engine**
A dynamic system that adjusts behavior based on admin settings.
*   **Rudeness Slider (0-100%)**:
    *   **0-30% (Angelic)**: Helpful, polite, standard assistant.
    *   **31-70% (Sassy)**: Helpful but sarcastic, makes fun of simple questions.
    *   **71-100% (Savage)**: Toxic, aggressive, actively insults users while answering.
*   **Active Shaming**: Detects what game/music you are playing via Discord Activity status.
    *   *Playing League of Legends?* It will mock your life choices.
    *   *Listening to Taylor Swift?* It will judge your taste.
*   **Rude Welcomes**: Customized insults when specific users join the Voice Channel.
*   **Savage Reactions**: Analyzes text messages and reacts with emojis (🤡, 💀, 🙄, 👎) based on sentiment.
*   **Auto-Clown**: 1% chance to reply to any text message with an "Official Clown License" image.
*   **DM Shaming**: If a user DMs the bot, it mocks them for being too scared to speak in the public server.

### 🖼️ **Creative & Utility**
*   **Local Image Generation**:
    *   Command: `/imagine [prompt]`
    *   Model: **Stable Diffusion 1.5** (Optimized for Low VRAM).
    *   Performance: Offloads to CPU when not in use to save VRAM for LLM/TTS.
*   **Deep Memory System**:
    *   context-aware conversations (remembers the last 5-20 messages).
    *   Persistent per-channel, short-term memory.
    *   `/reset` command to verify or wipe memory instantly.

### 💻 **Web Dashboard**
A full local web interface (`http://localhost:5000`) to control the bot in real-time.
*   **System Monitor**: View CPU, RAM, GPU Load, VRAM Usage, and Ping.
*   **Model Switcher**: Dynamic dropdown to switch **LLM Models** via LM Studio API without restarting.
*   **Feature Toggles**: Instantly enable/disable specific features (Roast, Rude Welcomes, etc.).
*   **Live Feed**: See a real-time log of what the bot is hearing and thinking.
*   **Voice Selector**: Switch between 10+ different TTS voices (American, British, etc.).
*   **Control Center**: Buttons to Force Disconnect, Reset Memory, or Restart the System.

### 🔍 **Technical Stack**
*   **LLM**: Connects to **LM Studio** (OpenAI-compatible API) for intelligence.
*   **STT**: `faster-whisper` (impl. CTranslate2).
*   **TTS**: `kokoro` (ONNX/PyTorch).
*   **VAD**: `silero-vad` (TorchHub).
*   **Image**: `diffusers` (HuggingFace).
*   **Backend**: Python 3.10 + Flask (Dashboard).

---

## 🚀 Installation Guide

### 1. Prerequisites
*   **OS**: Windows 10/11 (Preferred for Audio Drivers).
*   **Python**: Version 3.10 or higher.
*   **GPU**: NVIDIA GPU with at least 6GB VRAM recommended (supports CPU-only but slow).
*   **FFmpeg**: Must be installed and added to System PATH. [Download Here](https://ffmpeg.org/download.html).
*   **LM Studio**: Download from [lmstudio.ai](https://lmstudio.ai/) to run the LLM.

### 2. Setup
1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/yourusername/SavageAI.git
    cd SavageAI
    ```

2.  **Install Python Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *(Tip: If you have issues with PyTorch, install the CUDA version specifically from [pytorch.org](https://pytorch.org/))*

3.  **Environment Config**:
    *   Rename `.env.example` to `.env`.
    *   Add your **Discord Bot Token**.
    *   Ensure `LOCAL_AI_BASE_URL` matches your LM Studio port (default `http://localhost:1234/v1`).

4.  **LM Studio Setup**:
    *   Load a model (Recommended: `Llama-3-8B-Instruct` or `Mistral-7B`).
    *   Start the **Local Server** on port `1234`.
    *   Ensure "Cross-Origin-Resource-Sharing (CORS)" is enabled if accessing from a browser app (optional).

### 3. Run the Bot
Double-click `start_bot.bat` OR run via terminal:
```bash
python execution/discord_bot.py
```
*The Dashboard will launch at `http://localhost:5000`.*

---

## 🛠️ Slash Command List

### **User Commands**
| Command | Description |
| :--- | :--- |
| `/join` | Summon the bot to your Voice Channel. |
| `/leave` | Dismiss the bot from Voice Chat. |
| `/reset` | Wipe short-term memory for the current channel. |
| `/status` | View bot health, latency, and loaded models. |
| `/clown @user` | Generate and send an "Official Clown License" to a user. |
| `/imagine [prompt]` | Generate an AI image based on your prompt. |
| `/roast @user` | Manually trigger a roast on a specific user. |

### **Admin Commands**
| Command | Description |
| :--- | :--- |
| `/rudeness [0-100]` | Set the global rudeness/personality level. |
| `/context [1-20]` | Set how many messages the bot remembers (Memory Depth). |
| `/say [text]` | Force the bot to say something in Voice Chat (TTS). |
| `/troll @user` | Target a specific user for constant harassment. |
| `/toggle [feature]` | Enable/disable features (`roast`, `savage_reactions`, `auto_clown`). |

---

## ⚠️ Disclaimer
This bot is designed to be **offensive**.
*   It **will** insult users.
*   It **will** judge their activities.
*   It **will** be unhelpful at high rudeness levels.

**Developer Note**: All processing (Voice/Image/Text) happens **locally**. No user data is sent to external clouds (OpenAI/Google/Anthropic) unless you specifically configure the bot to use a cloud API as the backend.

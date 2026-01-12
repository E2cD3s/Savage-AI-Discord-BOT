from kokoro import KPipeline
import soundfile as sf
import os
import logging
import torch
import io
import warnings

# Suppress specific warnings from Kokoro/Torch
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.rnn")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.nn.utils.weight_norm")

class TTSService:
    def __init__(self, voice="af_bella", speed=1.0):
        """
        Initializes Kokoro TTS with customizable voice/speed.
        """
        # self.output_dir = ".tmp"
        # os.makedirs(self.output_dir, exist_ok=True)
        
        # Config
        self.default_voice = voice
        self.speed = speed
        
        # Check for GPU
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logging.info(f"Loading Kokoro TTS on {self.device}...")
        
        try:
            # lang_code='a' is typically American English in Kokoro default
            # repo_id='hexgrad/Kokoro-82M' suppresses the default warning
            # KPipeline(..., device='cuda') is supported in newer versions
            self.pipeline = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M', device=self.device) 
            logging.info(f"Kokoro TTS pipeline loaded successfully on {self.device}.")
        except Exception as e:
            logging.error(f"Failed to load Kokoro TTS: {e}")
            self.pipeline = None
            
        # Default Voice
        # Default Voice set in init args
        # self.default_voice = 'af_bella'

    async def generate_audio(self, text: str) -> io.BytesIO | None:
        """
        Generates audio from text using Kokoro and saves to memory.
        Returns BytesIO object.
        """
        if not self.pipeline:
            logging.error("TTS Pipeline not loaded.")
            return None

        try:
             # Kokoro generation
            # pipeline(text, voice, speed, split_pattern)
            # It returns a generator, but usually we just want the audio.
            
            logging.info(f"Generating TTS for: {text[:30]}...")
            
            # Simple synthesis to one file
            generator = self.pipeline(
                text, 
                voice=self.default_voice, 
                speed=self.speed, 
                split_pattern=r'\n+'
            )
            
            # Concatenate all audio segments
            all_audio = []
            for i, (gs, ps, audio) in enumerate(generator):
                all_audio.extend(audio)
                
            if not all_audio:
                logging.warning("Kokoro produced no audio.")
                return None

            # Save to memory (BytesIO)
            audio_buffer = io.BytesIO()
            # Kokoro usually outputs 24000Hz
            sf.write(audio_buffer, all_audio, 24000, format='WAV')
            audio_buffer.seek(0)
            
            logging.info(f"Audio generated in memory ({len(audio_buffer.getbuffer())} bytes)")
            return audio_buffer

        except Exception as e:
            logging.error(f"Error generating TTS audio: {e}")
            return None

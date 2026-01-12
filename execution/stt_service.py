from faster_whisper import WhisperModel
import os
import logging

class STTService:
    def __init__(self, model_size="base.en", device="cuda", compute_type="float16"):
        """
        Initializes Faster-Whisper on GPU (as requested for speed).
        """
        logging.info(f"Loading Faster-Whisper model: {model_size} on {device}...")
        try:
            self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
            logging.info("Faster-Whisper loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load Faster-Whisper on {device}: {e}")
            logging.info("Attempting fallback to CPU...")
            try:
                self.model = WhisperModel("medium", device="cpu", compute_type="int8")
                logging.info("Faster-Whisper loaded successfully on CPU.")
            except Exception as e2:
                logging.error(f"Failed to load Faster-Whisper on CPU: {e2}")
                self.model = None

    def transcribe_audio(self, audio_source) -> str | None:
        """
        Transcribes audio using Faster-Whisper.
        audio_source: str (path) or BinaryIO (memory)
        """
        if not self.model:
            logging.error("STT Model not loaded.")
            return None

        try:
            # Faster-Whisper accepts file-like objects directly
            # beam_size=5 improves accuracy significantly over greedy (1)
            segments, info = self.model.transcribe(audio_source, beam_size=5)
            logging.info(f"Detected language '{info.language}' with probability {info.language_probability}")
            
            text = " ".join([segment.text for segment in segments])
            logging.info(f"Transcribed text: {text}")
            return text.strip()
        except Exception as e:
            logging.error(f"Error during transcription: {e}")
            return None

import io

import scipy.io.wavfile
import torch
from pocket_tts import TTSModel


class TextToSpeech:
    def __init__(self, voice_name: str = "alba"):
        print("Loading Pocket TTS model into memory...")
        # 1. Load the model ONCE during initialization
        self.model = TTSModel.load_model()
        self.sample_rate = self.model.sample_rate

        # 2. Pre-load the voice state to save time during requests
        # get_state_for_audio_prompt is a relatively slow operation
        self.voice_state = self.model.get_state_for_audio_prompt(voice_name)
        print(f"Model loaded with voice: {voice_name}")

    def generate_wav_bytes(self, text: str) -> io.BytesIO:
        """Generates audio and returns it as a byte stream (no disk writing)."""
        # Generate audio (returns a 1D torch tensor)
        audio_tensor = self.model.generate_audio(self.voice_state, text)

        # Convert to numpy for scipy
        audio_data = audio_tensor.numpy()

        # Create an in-memory file
        byte_io = io.BytesIO()
        scipy.io.wavfile.write(byte_io, self.sample_rate, audio_data)
        byte_io.seek(0)

        return byte_io

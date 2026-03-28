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

        # 3. OPT: Cache generated WAV bytes by text so repeated questions
        #    (e.g. a question asked twice across reloads) are served instantly
        #    without re-running the TTS model.  Keys are the raw text strings;
        #    values are the raw bytes of the WAV file.
        self._cache: dict[str, bytes] = {}

    def generate_wav_bytes(self, text: str) -> io.BytesIO:
        """Generates audio and returns it as a byte stream (no disk writing).

        Results are memoised: identical text strings are returned from an
        in-memory cache on subsequent calls, skipping model inference entirely.
        """
        # FIX OPT: serve from cache when available
        if text in self._cache:
            return io.BytesIO(self._cache[text])

        # Generate audio (returns a 1D torch tensor)
        audio_tensor = self.model.generate_audio(self.voice_state, text)

        # Convert to numpy for scipy
        audio_data = audio_tensor.numpy()

        # Create an in-memory file
        byte_io = io.BytesIO()
        scipy.io.wavfile.write(byte_io, self.sample_rate, audio_data)

        # Store the raw bytes in the cache before seeking so both paths
        # (cached and uncached) return a freshly-seeked BytesIO object.
        self._cache[text] = byte_io.getvalue()

        byte_io.seek(0)
        return byte_io

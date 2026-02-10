import io

import numpy as np
import torch
from qwen_asr import Qwen3ASRModel


class SpeechToText:
    def __init__(self):
        # Load once
        self.model = Qwen3ASRModel.from_pretrained(
            "Qwen3-ASR-0.6B", dtype=torch.float32, device_map="cpu"
        )
        self.audio_buffer = []  # To store recent chunks for context

    def transcribe_chunk(self, byte_chunk: bytes):
        """
        Processes a small slice of audio (e.g., 1-2 seconds).
        Returns the transcription for that specific slice.
        """
        # Convert raw bytes (from client) to the format the model expects
        # Assuming 16kHz Mono PCM audio
        audio_np = (
            np.frombuffer(byte_chunk, dtype=np.int16).astype(np.float32) / 32768.0
        )

        # In a real server, you'd prepend a bit of the 'previous' audio here
        # for better accuracy, then call the model:
        result = self.model.transcribe(audio=audio_np, language="English")

        return result[0].text if result else ""

"""
Speech-to-Text + Disfluency Detection
---------------------------------------
Uses Qwen3 ASR to transcribe audio and counts disfluency markers
to flag stuttering/hesitation as a stress signal.

Supports:
    - Raw PCM bytes (from client WebSocket stream)
    - .opus files (converted via ffmpeg before transcription)

Requirements:
    pip install qwen-asr torch numpy
    sudo apt install ffmpeg
"""

import io
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
from qwen_asr import Qwen3ASRModel

# ─── Disfluency config ────────────────────────────────────────────────────────

# Filled pauses and hesitation markers Qwen3 preserves by default
DISFLUENCY_MARKERS = {
    "um",
    "uh",
    "er",
    "eh",
    "ah",
    "hmm",
    "hm",
    "uhh",
    "umm",
    "ahh",
    "err",
    "uhhh",
    "like",  # informal filler — may produce false positives, remove if needed
    "you know",
    "i mean",
    "sort of",
    "kind of",
}

# Regex for prolongations e.g. "sooo", "aaand", "nooo"
PROLONGATION_RE = re.compile(r"\b\w*(.)\1{2,}\w*\b")

# Hyphenated false starts e.g. "s-so", "b-because"
FALSE_START_RE = re.compile(r"\b[a-z]-\w+\b", re.IGNORECASE)

# Adjacent repetitions (existing): "I I", "and and and"
ADJACENT_RE = re.compile(r"\b(\w+)\s+\1\b", re.IGNORECASE)

# Threshold: disfluency tokens / total words
# Above this → stutter flag raised
DISFLUENCY_THRESHOLD = 0.10  # lowered to 10% after real-world testing

# Minimum word length for windowed/phrase repetition checks
# Skips short words like "a", "I", "is", "the" which repeat naturally
MIN_REPEAT_WORD_LEN = 3

# How many words ahead to look for non-adjacent repetitions
REPETITION_WINDOW = 8


# ══════════════════════════════════════════════════════════════════════════════
#  REPETITION HELPERS
# ══════════════════════════════════════════════════════════════════════════════


def count_adjacent_repetitions(words: list) -> int:
    """
    Count runs of consecutive identical words.
    "I I I I" = 3 repetitions (not 1).
    "and and and" = 2 repetitions.
    """
    count = 0
    i = 0
    while i < len(words) - 1:
        if words[i].lower() == words[i + 1].lower():
            run = 1
            while i + run < len(words) and words[i + run].lower() == words[i].lower():
                run += 1
            count += run - 1
            i += run
        else:
            i += 1
    return count


def count_windowed_repetitions(words: list, window: int = REPETITION_WINDOW) -> int:
    """
    Count non-adjacent word repetitions within a sliding window.
    Catches: "whether ... whether ... whether" style repetition.
    Skips short words to avoid false positives on common function words.
    """
    count = 0
    for i, word in enumerate(words):
        if len(word) < MIN_REPEAT_WORD_LEN:
            continue
        ahead = words[i + 1 : i + window]
        if word.lower() in [w.lower() for w in ahead]:
            count += 1
    return count


def count_phrase_repetitions(words: list) -> int:
    """
    Count repeated bigrams and trigrams.
    Catches: "in a large in a large", "on the on the" etc.
    """
    count = 0
    text = " ".join(words)
    for n in [2, 3]:
        for i in range(len(words) - n):
            phrase = " ".join(words[i : i + n]).lower()
            # Skip phrases made entirely of short words
            if all(len(w) < MIN_REPEAT_WORD_LEN for w in phrase.split()):
                continue
            rest = " ".join(words[i + n :]).lower()
            if phrase in rest:
                count += 1
    return count


# ══════════════════════════════════════════════════════════════════════════════
#  DISFLUENCY ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════


def analyze_disfluencies(transcript: str) -> dict:
    """
    Count disfluency markers in a transcript and return a summary dict.

    Detects:
        - Filled pauses       : um, uh, er, ah, hmm, etc.
        - Prolongations       : sooo, aaand, etc.
        - False starts        : s-so, b-because (hyphenated)
        - Adjacent repetitions: I I I, and and and
        - Windowed repetitions: whether ... whether ... whether
        - Phrase repetitions  : in a large in a large

    Returns:
        {
            "total_words":          int,
            "filled_pauses":        int,
            "prolongations":        int,
            "false_starts":         int,
            "adjacent_repetitions": int,
            "windowed_repetitions": int,
            "phrase_repetitions":   int,
            "total_disfluencies":   int,
            "disfluency_rate":      float,
            "stutter_flag":         bool,
        }
    """
    if not transcript or not transcript.strip():
        return {
            "total_words": 0,
            "filled_pauses": 0,
            "prolongations": 0,
            "false_starts": 0,
            "adjacent_repetitions": 0,
            "windowed_repetitions": 0,
            "phrase_repetitions": 0,
            "total_disfluencies": 0,
            "disfluency_rate": 0.0,
            "stutter_flag": False,
        }

    text = transcript.lower().strip()
    # Strip punctuation for word-level analysis
    clean_text = re.sub(r"[^\w\s-]", " ", text)
    words = clean_text.split()
    total_words = max(len(words), 1)

    # Filled pauses (single-word markers)
    single_markers = {m for m in DISFLUENCY_MARKERS if " " not in m}
    multi_markers = {m for m in DISFLUENCY_MARKERS if " " in m}
    filled_pauses = sum(1 for w in words if w in single_markers)
    for marker in multi_markers:
        filled_pauses += text.count(marker)

    # Prolongations e.g. sooo, aaand
    prolongations = len(PROLONGATION_RE.findall(text))

    # Hyphenated false starts e.g. s-so, b-because
    false_starts = len(FALSE_START_RE.findall(text))

    # Adjacent repetitions: "I I I" = 2, "and and" = 1
    adjacent_reps = count_adjacent_repetitions(words)

    # Non-adjacent windowed repetitions: "whether...whether...whether"
    windowed_reps = count_windowed_repetitions(words)

    # Phrase-level repetitions: "in a large in a large"
    phrase_reps = count_phrase_repetitions(words)

    total_disfluencies = (
        filled_pauses
        + prolongations
        + false_starts
        + adjacent_reps
        + windowed_reps
        + phrase_reps
    )
    disfluency_rate = total_disfluencies / total_words

    return {
        "total_words": total_words,
        "filled_pauses": filled_pauses,
        "prolongations": prolongations,
        "false_starts": false_starts,
        "adjacent_repetitions": adjacent_reps,
        "windowed_repetitions": windowed_reps,
        "phrase_repetitions": phrase_reps,
        "total_disfluencies": total_disfluencies,
        "disfluency_rate": round(disfluency_rate, 4),
        "stutter_flag": disfluency_rate > DISFLUENCY_THRESHOLD,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  OPUS DECODING
#  Converts .opus bytes or file path -> float32 numpy array at 16kHz mono
#  Uses ffmpeg under the hood (same dependency as your audio pipeline)
# ══════════════════════════════════════════════════════════════════════════════


def decode_opus_bytes(opus_bytes: bytes, sample_rate: int = 16000) -> np.ndarray:
    """
    Decode raw .opus bytes to a float32 numpy array.

    Args:
        opus_bytes:  Raw bytes of an .opus file (e.g. from client upload)
        sample_rate: Target sample rate (default 16000)

    Returns:
        float32 numpy array, shape (n_samples,), range [-1.0, 1.0]
    """
    with tempfile.NamedTemporaryFile(suffix=".opus", delete=False) as tmp_in:
        tmp_in.write(opus_bytes)
        tmp_in_path = tmp_in.name

    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            tmp_in_path,
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(sample_rate),
            "-ac",
            "1",  # mono
            "-f",
            "s16le",  # raw PCM output (no WAV header)
            "pipe:1",  # write to stdout
        ]
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        if result.returncode != 0:
            raise RuntimeError("ffmpeg failed to decode .opus file")

        pcm = np.frombuffer(result.stdout, dtype=np.int16)
        return pcm.astype(np.float32) / 32768.0

    finally:
        Path(tmp_in_path).unlink(missing_ok=True)


def decode_opus_file(file_path: str, sample_rate: int = 16000) -> np.ndarray:
    """
    Decode a .opus file from disk to a float32 numpy array.

    Args:
        file_path:   Path to the .opus file
        sample_rate: Target sample rate (default 16000)

    Returns:
        float32 numpy array, shape (n_samples,), range [-1.0, 1.0]
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(file_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        "-f",
        "s16le",
        "pipe:1",
    ]
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed to decode: {file_path}")

    pcm = np.frombuffer(result.stdout, dtype=np.int16)
    return pcm.astype(np.float32) / 32768.0


# ══════════════════════════════════════════════════════════════════════════════
#  SPEECH TO TEXT CLASS
# ══════════════════════════════════════════════════════════════════════════════


class SpeechToText:
    def __init__(self):
        self.model = Qwen3ASRModel.from_pretrained(
            "Qwen3-ASR-0.6B", dtype=torch.float32, device_map="cpu"
        )
        self.audio_buffer = []

    def _transcribe_array(self, audio_np: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Internal helper: transcribe a float32 numpy array.
        Writes to a temporary WAV file first since qwen_asr expects a file path,
        not a raw numpy array.
        """
        import soundfile as sf

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            sf.write(tmp_path, audio_np, sample_rate, subtype="PCM_16")
            result = self.model.transcribe(audio=tmp_path, language="English")
            return result[0].text if result else ""
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def transcribe_chunk(self, byte_chunk: bytes) -> str:
        """
        Transcribe a raw PCM byte chunk (16kHz mono int16).
        Returns the transcript string.
        """
        audio_np = (
            np.frombuffer(byte_chunk, dtype=np.int16).astype(np.float32) / 32768.0
        )
        return self._transcribe_array(audio_np)

    def transcribe_opus_bytes(self, opus_bytes: bytes) -> str:
        """
        Transcribe a .opus audio clip received as raw bytes.
        Typical use: client sends .opus blob over HTTP/WebSocket.

        Args:
            opus_bytes: Raw bytes of the .opus file

        Returns:
            Transcript string
        """
        audio_np = decode_opus_bytes(opus_bytes)
        return self._transcribe_array(audio_np)

    def transcribe_opus_file(self, file_path: str) -> str:
        """
        Transcribe a .opus file from disk.

        Args:
            file_path: Path to .opus file

        Returns:
            Transcript string
        """
        audio_np = decode_opus_file(file_path)
        return self._transcribe_array(audio_np)

    def transcribe_and_analyze(self, opus_bytes: bytes) -> dict:
        """
        Full pipeline: decode opus -> transcribe -> analyze disfluencies.
        This is the main entry point for the server-side stress analysis.

        Args:
            opus_bytes: Raw .opus bytes from client

        Returns:
            {
                "transcript":     str,
                "stutter_flag":   bool,
                "disfluency_rate": float,
                "details":        dict   # full disfluency breakdown
            }
        """
        transcript = self.transcribe_opus_bytes(opus_bytes)
        analysis = analyze_disfluencies(transcript)

        return {
            "transcript": transcript,
            "stutter_flag": analysis["stutter_flag"],
            "disfluency_rate": analysis["disfluency_rate"],
            "details": analysis,
        }


# ══════════════════════════════════════════════════════════════════════════════
#  STRESS LEVEL COMBINER
#  Merges audio model output with disfluency flag into a final stress level
# ══════════════════════════════════════════════════════════════════════════════


def combine_stress_signals(model_stressed: bool, stutter_flag: bool) -> dict:
    """
    Combine the audio model's stress prediction with the disfluency flag.

    Args:
        model_stressed: True if XGBoost/SVM flagged audio as stressed
        stutter_flag:   True if disfluency rate exceeded threshold

    Returns:
        {
            "level":   str,   # "highly_stressed" | "stressed" | "mild" | "not_stressed"
            "message": str,   # human-readable label
        }
    """
    if model_stressed and stutter_flag:
        return {
            "level": "highly_stressed",
            "message": "Highly stressed — both vocal patterns and speech disruption detected",
        }
    elif model_stressed and not stutter_flag:
        return {
            "level": "stressed",
            "message": "Stressed — elevated vocal stress indicators detected",
        }
    elif not model_stressed and stutter_flag:
        return {
            "level": "mild",
            "message": "Mild stress indicators — speech disruption detected",
        }
    else:
        return {
            "level": "not_stressed",
            "message": "No significant stress indicators detected",
        }


# ══════════════════════════════════════════════════════════════════════════════
#  QUICK TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Test disfluency analyzer without needing the ASR model
    test_transcripts = [
        "Um I was just, uh, thinking that maybe we could, you know, try a different approach",
        "The report is ready and I have sent it to the team",
        "I I I don't know what to do sooo I just kept going",
    ]

    print("Disfluency Analysis Test")
    print("=" * 50)
    for t in test_transcripts:
        result = analyze_disfluencies(t)
        print(f"\nTranscript : {t}")
        print(f"Rate       : {result['disfluency_rate']:.2%}")
        print(f"Stutter    : {result['stutter_flag']}")
        print(
            f"Details    : pauses={result['filled_pauses']} "
            f"prolongations={result['prolongations']} "
            f"repetitions={result['repetitions']}"
        )

    print("\n\nStress Combination Test")
    print("=" * 50)
    for model_out, stutter in [
        (True, True),
        (True, False),
        (False, True),
        (False, False),
    ]:
        r = combine_stress_signals(model_out, stutter)
        print(
            f"  model={model_out}, stutter={stutter} -> [{r['level']}] {r['message']}"
        )

    # To test with a real .opus file:
    # stt = SpeechToText()
    # with open("test.opus", "rb") as f:
    #     result = stt.transcribe_and_analyze(f.read())
    # print(result)

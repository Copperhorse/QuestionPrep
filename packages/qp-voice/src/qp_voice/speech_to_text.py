"""
Speech-to-Text + Disfluency Detection
---------------------------------------
Uses Qwen3 ASR to transcribe audio and counts disfluency markers
to flag stuttering/hesitation as a stress signal.

Supports:
    - Raw PCM bytes (from client WebSocket stream)
    - .opus/.webm files (converted via ffmpeg before transcription)

Requirements:
    pip install qwen-asr torch numpy soundfile
    sudo apt install ffmpeg

Fixes applied:
  FIX1 - 'import soundfile as sf' was deferred inside _transcribe_array().
         A missing soundfile package only surfaced at inference time (first
         live transcription call) instead of at startup, making the root
         cause very hard to diagnose.  Moved to the module-level imports so
         the error is raised immediately when the module is loaded.

  FIX2 - decode_opus_bytes() wrote the browser audio to a temp file with
         suffix='.opus'.  The browser's MediaRecorder produces audio/webm
         (Chrome) or audio/ogg (Firefox), not a bare Opus stream.  Some
         ffmpeg builds use the filename extension as a format hint, causing
         a decode failure.  Changed to suffix='.webm'.  ffmpeg auto-detects
         the actual container from magic bytes, so this works for both WebM
         and Ogg audio.

  FIX3 - Added a pipe-based fast path to decode_opus_bytes().  ffmpeg can
         read directly from stdin for most containers, avoiding a temp-file
         round-trip.  The disk path (FIX2) is retained as a fallback for
         containers (like some Matroska/WebM variants) that require seeking.

  B14  - __main__ test block printed result['repetitions'], a key that does
         not exist.  Fixed to print all three separate repetition keys.
"""

import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf  # FIX1: module-level — ImportError surfaces at startup, not mid-call
import torch
from qwen_asr import Qwen3ASRModel

# ─── Disfluency config ────────────────────────────────────────────────────────

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
    "like",
    "you know",
    "i mean",
    "sort of",
    "kind of",
}

PROLONGATION_RE = re.compile(r"\b\w*(.)\1{2,}\w*\b")
FALSE_START_RE = re.compile(r"\b[a-z]-\w+\b", re.IGNORECASE)
ADJACENT_RE = re.compile(r"\b(\w+)\s+\1\b", re.IGNORECASE)

DISFLUENCY_THRESHOLD = 0.10
MIN_REPEAT_WORD_LEN = 3
REPETITION_WINDOW = 8


# ══════════════════════════════════════════════════════════════════════════════
#  REPETITION HELPERS
# ══════════════════════════════════════════════════════════════════════════════


def count_adjacent_repetitions(words: list) -> int:
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
    count = 0
    for i, word in enumerate(words):
        if len(word) < MIN_REPEAT_WORD_LEN:
            continue
        ahead = words[i + 1 : i + window]
        if word.lower() in [w.lower() for w in ahead]:
            count += 1
    return count


def count_phrase_repetitions(words: list) -> int:
    count = 0
    for n in [2, 3]:
        for i in range(len(words) - n):
            phrase = " ".join(words[i : i + n]).lower()
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
    clean_text = re.sub(r"[^\w\s-]", " ", text)
    words = clean_text.split()
    total_words = max(len(words), 1)

    single_markers = {m for m in DISFLUENCY_MARKERS if " " not in m}
    multi_markers = {m for m in DISFLUENCY_MARKERS if " " in m}
    filled_pauses = sum(1 for w in words if w in single_markers)
    for marker in multi_markers:
        filled_pauses += text.count(marker)

    prolongations = len(PROLONGATION_RE.findall(text))
    false_starts = len(FALSE_START_RE.findall(text))
    adjacent_reps = count_adjacent_repetitions(words)
    windowed_reps = count_windowed_repetitions(words)
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
#  AUDIO DECODING
# ══════════════════════════════════════════════════════════════════════════════


def _pcm_from_stdout(stdout: bytes) -> np.ndarray:
    """Convert raw s16le PCM bytes (ffmpeg output) to a float32 numpy array."""
    pcm = np.frombuffer(stdout, dtype=np.int16)
    return pcm.astype(np.float32) / 32768.0


def decode_opus_bytes(opus_bytes: bytes, sample_rate: int = 16000) -> np.ndarray:
    """
    Decode browser audio bytes (WebM/Ogg container) to a float32 PCM array.

    FIX3: Tries a pipe-based fast path first (no temp file I/O).
    FIX2: Falls back to a temp file with suffix='.webm' (was '.opus').
          The browser sends a WebM or Ogg container, not a bare Opus stream.
          ffmpeg auto-detects from magic bytes, but '.opus' misled some builds
          into skipping container demuxing entirely.
    """
    _FFMPEG_ARGS = [
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

    # ── Fast path: pipe bytes straight into ffmpeg stdin ─────────────────────
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", "pipe:0"] + _FFMPEG_ARGS,
        input=opus_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode == 0 and result.stdout:
        return _pcm_from_stdout(result.stdout)

    # ── Fallback: write to a temp file ────────────────────────────────────────
    # FIX2: suffix changed from '.opus' to '.webm'
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp_in:
        tmp_in.write(opus_bytes)
        tmp_in_path = tmp_in.name

    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_in_path] + _FFMPEG_ARGS,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "ffmpeg failed to decode audio (both pipe and file path tried)"
            )
        return _pcm_from_stdout(result.stdout)
    finally:
        Path(tmp_in_path).unlink(missing_ok=True)


def decode_opus_file(file_path: str, sample_rate: int = 16000) -> np.ndarray:
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
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
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
        # FIX1: soundfile is now imported at module level, not here.
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            sf.write(tmp_path, audio_np, sample_rate, subtype="PCM_16")
            result = self.model.transcribe(audio=tmp_path, language="English")
            return result[0].text if result else ""
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def transcribe_chunk(self, byte_chunk: bytes) -> str:
        audio_np = (
            np.frombuffer(byte_chunk, dtype=np.int16).astype(np.float32) / 32768.0
        )
        return self._transcribe_array(audio_np)

    def transcribe_opus_bytes(self, opus_bytes: bytes) -> str:
        audio_np = decode_opus_bytes(opus_bytes)
        return self._transcribe_array(audio_np)

    def transcribe_opus_file(self, file_path: str) -> str:
        audio_np = decode_opus_file(file_path)
        return self._transcribe_array(audio_np)

    def transcribe_and_analyze(self, opus_bytes: bytes) -> dict:
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
# ══════════════════════════════════════════════════════════════════════════════


def combine_stress_signals(model_stressed: bool, stutter_flag: bool) -> dict:
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
        # B14 FIX: was result['repetitions'] — key doesn't exist
        print(
            f"Details    : pauses={result['filled_pauses']} "
            f"prolongations={result['prolongations']} "
            f"adjacent_reps={result['adjacent_repetitions']} "
            f"windowed_reps={result['windowed_repetitions']} "
            f"phrase_reps={result['phrase_repetitions']}"
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

"""
speech_to_text.py

Fixes applied:
  B14 - The __main__ test block printed result['repetitions'], but
        analyze_disfluencies() returns three separate keys:
          adjacent_repetitions, windowed_repetitions, phrase_repetitions.
        Running the file directly threw a KeyError. Fixed to print all three keys.

  OPT - decode_opus_bytes() previously wrote opus bytes to a NamedTemporaryFile
        so that ffmpeg could read from disk, which consumed an unnecessary file
        descriptor and added I/O latency.  Replaced with a pipe-based approach:
        ffmpeg reads from stdin (pipe:0) and the temp file is eliminated entirely.
        A fallback to the old temp-file path is retained for formats that ffmpeg
        cannot demux from a non-seekable pipe (e.g. some MPEG-TS containers).
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


def analyze_disfluencies(transcript: str) -> dict:
    """
    Count disfluency markers and return a summary dict.

    Keys returned:
        total_words, filled_pauses, prolongations, false_starts,
        adjacent_repetitions, windowed_repetitions, phrase_repetitions,
        total_disfluencies, disfluency_rate, stutter_flag
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


def _pcm_from_stdout(stdout: bytes) -> np.ndarray:
    """Convert raw s16le PCM bytes (ffmpeg output) to a float32 numpy array."""
    pcm = np.frombuffer(stdout, dtype=np.int16)
    return pcm.astype(np.float32) / 32768.0


def decode_opus_bytes(opus_bytes: bytes, sample_rate: int = 16000) -> np.ndarray:
    """Decode opus/webm bytes to a float32 PCM array.

    OPT: Tries to feed the bytes directly to ffmpeg via stdin (pipe:0 → pipe:1)
    to avoid writing a temp file.  Falls back to the temp-file path for
    containers that ffmpeg cannot demux from a non-seekable stream.
    """
    _FFMPEG_PCM_ARGS = [
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
        ["ffmpeg", "-y", "-i", "pipe:0"] + _FFMPEG_PCM_ARGS,
        input=opus_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode == 0 and result.stdout:
        return _pcm_from_stdout(result.stdout)

    # ── Fallback: write to a temp file (required by some containers) ─────────
    with tempfile.NamedTemporaryFile(suffix=".opus", delete=False) as tmp_in:
        tmp_in.write(opus_bytes)
        tmp_in_path = tmp_in.name

    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_in_path] + _FFMPEG_PCM_ARGS,
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


class SpeechToText:
    def __init__(self):
        self.model = Qwen3ASRModel.from_pretrained(
            "Qwen3-ASR-0.6B", dtype=torch.float32, device_map="cpu"
        )
        self.audio_buffer = []

    def _transcribe_array(self, audio_np: np.ndarray, sample_rate: int = 16000) -> str:
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
        # B14: was result['repetitions'] — that key does not exist.
        # The dict has three separate repetition keys:
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

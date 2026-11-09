"""Routing: voice or text -> (STT for voice) -> sentiment on the ORIGINAL text (no translation).

Voice analysis (prosody model + fusion) runs ONLY on the English / Whisper path, because the
voice model is trained on English speech. Malayalam and code-mixed audio (Sarvam path) and typed
text skip it entirely.
"""
import time

from .audio import analyze_voice
from .fusion import describe_voice, fuse
from .nlp import classify_sentiment
from .stt import detect_and_transcribe

WHISPER_PROVIDER = "groq"  # STT provider used only on the English path (see stt/detect.py)


def process(text: str = "", audio: tuple[bytes, str, float | None] | None = None) -> dict:
    """audio = (bytes, filename, duration_seconds). Returns transcript, language, sentiment, voice,
    fusion, provider, seconds and error. `language` is only known for voice."""
    t0 = time.perf_counter()
    out = {"transcript": "", "language": "", "sentiment": None, "voice": None, "fusion": None,
           "provider": "", "error": ""}

    def done() -> dict:
        out["seconds"] = round(time.perf_counter() - t0, 2)
        return out

    english_voice = False
    if audio:
        stt = detect_and_transcribe(*audio)
        out["provider"], out["language"] = stt["provider"], stt["language"]
        if not stt["text"]:
            out["error"] = stt["error"] or "Could not transcribe the audio."
            return done()
        text = out["transcript"] = stt["text"]
        english_voice = stt["provider"] == WHISPER_PROVIDER and stt["language"] == "en"
    if not text.strip():
        out["error"] = "Empty message."
        return done()
    try:
        out["sentiment"] = classify_sentiment(text.strip())
    except Exception as e:
        out["error"] = f"Sentiment failed: {e}"
        return done()

    if english_voice:  # voice analysis is optional: any failure just leaves it out
        raw = analyze_voice(audio[0])
        if raw:
            out["voice"] = describe_voice(raw)
            out["fusion"] = fuse(out["sentiment"], out["voice"])
    return done()

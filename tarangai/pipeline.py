"""Routing: voice or text -> (STT for voice) -> sentiment on the ORIGINAL text. No translation step."""
import time

from .nlp import classify_sentiment
from .stt import detect_and_transcribe


def process(text: str = "", audio: tuple[bytes, str, float | None] | None = None) -> dict:
    """audio = (bytes, filename, duration_seconds). Returns transcript, language, sentiment,
    provider, seconds and error. `language` is only known for voice (from the STT routing)."""
    t0 = time.perf_counter()
    out = {"transcript": "", "language": "", "sentiment": "", "provider": "", "error": ""}

    def done() -> dict:
        out["seconds"] = round(time.perf_counter() - t0, 2)
        return out

    if audio:
        stt = detect_and_transcribe(*audio)
        out["provider"], out["language"] = stt["provider"], stt["language"]
        if not stt["text"]:
            out["error"] = stt["error"] or "Could not transcribe the audio."
            return done()
        text = out["transcript"] = stt["text"]
    if not text.strip():
        out["error"] = "Empty message."
        return done()
    try:
        out["sentiment"] = classify_sentiment(text.strip())
    except Exception as e:
        out["error"] = f"Sentiment failed: {e}"
    return done()

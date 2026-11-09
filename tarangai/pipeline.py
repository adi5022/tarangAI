"""Routing: voice or text -> (STT) -> language -> [Sarvam normalise]. Sentiment is not wired in yet."""
import time

from .nlp import normalize
from .stt import detect_and_transcribe


def analyze_text(text: str, language: str | None = None) -> dict:
    """Text path. `language` is 'en' only when verified by Whisper (voice), so no Sarvam call is needed.
    Typed Latin text can be English, Hinglish or Manglish, so it goes through Sarvam, which
    returns plain English unchanged."""
    text = text.strip()
    if language == "en":
        return {"language": "english", "english": text, "notes": ""}
    return normalize(text)


def process(text: str = "", audio: tuple[bytes, str, float | None] | None = None) -> dict:
    """audio = (bytes, filename, duration_seconds). Returns transcript, language, english, notes,
    provider, seconds and error."""
    t0 = time.perf_counter()
    out = {"transcript": "", "language": "", "english": "", "notes": "", "provider": "", "error": ""}
    lang_hint = None
    if audio:
        stt = detect_and_transcribe(*audio)
        out["provider"] = stt["provider"]
        if not stt["text"]:
            out["error"] = stt["error"] or "Could not transcribe the audio."
            out["seconds"] = round(time.perf_counter() - t0, 2)
            return out
        text, lang_hint = stt["text"], stt["language"]
        out["transcript"] = text
    if not text.strip():
        out["error"] = "Empty message."
        out["seconds"] = round(time.perf_counter() - t0, 2)
        return out
    try:
        out.update(analyze_text(text, lang_hint))
    except Exception as e:
        out["error"] = f"Text processing failed: {e}"
    out["seconds"] = round(time.perf_counter() - t0, 2)
    return out

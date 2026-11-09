"""Language routing for voice: English goes to Whisper, everything else to Sarvam Saaras (codemix).

Adapted from Sahayi. Free-first: one Groq Whisper call listens first; if it says English, that
transcript is used and Sarvam is never called. Whisper is trusted only for "English or not".
"""
import logging
import time

from .transcribe import transcribe_audio, trim_head, whisper_detect

ML_MIN_SHARE = 0.05  # below this share of Malayalam letters -> English
EN_MIN_SHARE = 0.10  # below this share of Latin letters -> Malayalam


def classify_text(text: str) -> tuple[str, float]:
    """('en' | 'ml' | 'ml-mix' | 'unknown', malayalam_share) from the script mix of `text`.
    Script only: romanized Malayalam (Manglish) looks like English here."""
    ml = sum("ഀ" <= c <= "ൿ" for c in text)
    latin = sum(c.isascii() and c.isalpha() for c in text)
    total = ml + latin
    if total == 0:
        return "unknown", 0.0
    share = ml / total
    if share < ML_MIN_SHARE:
        return "en", share
    if 1 - share < EN_MIN_SHARE:
        return "ml", share
    return "ml-mix", share


def detect_and_transcribe(file_bytes: bytes, filename: str = "voice.ogg", duration: float | None = None) -> dict:
    """Returns {text, language, provider, error, seconds}."""
    t0 = time.perf_counter()
    long_audio = (duration or len(file_bytes) / 4500) > 20
    probe, probe_name = (trim_head(file_bytes, 8.0), "head.wav") if long_audio else (file_bytes, filename)
    wlang, wtext, werr = whisper_detect(probe, probe_name)
    if werr:  # without this, a broken Groq setup silently sends English audio to Sarvam
        logging.warning("Whisper language check failed, falling back to Sarvam: %s", werr)
    else:
        logging.info("Whisper detected language=%s", wlang or "?")

    if wlang == "english" and wtext:
        text, provider = wtext, "groq"
        if long_audio:  # the probe only covered the first seconds; transcribe it all
            full, prov, err = transcribe_audio(file_bytes, filename, "en", duration)
            if full:
                text, provider = full, prov
        return {"text": text, "language": "en", "provider": provider, "error": "",
                "seconds": round(time.perf_counter() - t0, 2)}

    text, provider, err = transcribe_audio(file_bytes, filename, "ml-mix", duration)
    if not text:
        return {"text": "", "language": "unknown", "provider": provider, "error": err,
                "seconds": round(time.perf_counter() - t0, 2)}
    lang, _ = classify_text(text)
    return {"text": text, "language": lang, "provider": provider, "error": "",
            "seconds": round(time.perf_counter() - t0, 2)}

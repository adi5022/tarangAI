"""Speech-to-text helpers, adapted from Sahayi (github.com/adi5022/sahayiAI).

English -> Groq-hosted Whisper. Malayalam / code-mix -> Sarvam Saaras.
There is no cross-language fallback: Malayalam is never sent to Whisper, which tends to
mislabel it as Tamil.
"""
import io
import os
import time

import requests
from groq import Groq

SARVAM_URL = "https://api.sarvam.ai/speech-to-text"
SARVAM_SYNC_LIMIT_ERROR = "exceeds the maximum limit"  # real-time endpoint rejects audio over 30 s

MIME_BY_EXT = {
    "wav": "audio/wav", "mp3": "audio/mpeg", "aac": "audio/aac", "m4a": "audio/mp4",
    "ogg": "audio/ogg", "opus": "audio/ogg", "webm": "audio/webm", "flac": "audio/flac",
}


def guess_mime(filename: str) -> str:
    return MIME_BY_EXT.get(filename.rsplit(".", 1)[-1].lower(), "audio/wav")


def sarvam_batch(file_bytes: bytes, filename: str, mode: str, key: str) -> tuple[str, str]:
    """Long audio via Sarvam's batch job API. Returns (text, error)."""
    import json
    import tempfile
    from pathlib import Path

    from sarvamai import SarvamAI

    try:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / (Path(filename).name or "audio.wav")
            src.write_bytes(file_bytes)
            job = SarvamAI(api_subscription_key=key.strip()).speech_to_text_job.create_job(
                model="saaras:v3", mode=mode, language_code="ml-IN", with_diarization=False)
            job.upload_files(file_paths=[str(src)])
            job.start()
            job.wait_until_complete(poll_interval=1, timeout=300)
            res = job.get_file_results()
            if res.get("failed"):
                raise RuntimeError(res["failed"][0].get("error_message") or "batch job failed")
            out = Path(tmp) / "out"
            job.download_outputs(output_dir=str(out))
            text = " ".join(
                json.loads(p.read_text(encoding="utf-8")).get("transcript", "").strip()
                for p in sorted(out.glob("*.json"))
            ).strip()
        return text, "" if text else "Sarvam batch returned an empty transcript."
    except Exception as e:
        return "", f"Sarvam batch failed: {e}"


def transcribe_audio(file_bytes: bytes, filename: str = "audio.wav", language: str = "en",
                     duration: float | None = None) -> tuple[str, str, str]:
    """language: 'en' -> Groq Whisper, 'ml' -> Sarvam transcribe, 'ml-mix' -> Sarvam codemix.
    Returns (text, provider, error)."""
    if language in ("ml", "ml-mix"):
        key = os.environ.get("SARVAM_AI")
        if not key:
            return "", "none", "SARVAM_AI key missing in .env"
        mode = "codemix" if language == "ml-mix" else "transcribe"
        if duration and duration > 28:
            text, err = sarvam_batch(file_bytes, filename, mode, key)
            return text, "sarvam", err
        try:
            r = requests.post(
                SARVAM_URL,
                headers={"api-subscription-key": key.strip()},
                files={"file": (filename, file_bytes, guess_mime(filename))},
                data={"model": "saaras:v3", "mode": mode, "language_code": "ml-IN"},
                timeout=30,
            )
        except Exception as e:
            return "", "sarvam", f"Sarvam request failed: {e}"
        if r.status_code == 400 and SARVAM_SYNC_LIMIT_ERROR in r.text:
            text, err = sarvam_batch(file_bytes, filename, mode, key)
            return text, "sarvam", err
        if r.status_code != 200:
            return "", "sarvam", f"Sarvam returned {r.status_code}: {r.text[:300]}"
        text = r.json().get("transcript", "").strip()
        return (text, "sarvam", "") if text else ("", "sarvam", "Sarvam returned an empty transcript.")

    key = os.environ.get("GROQ_API_KEY")
    if not key:
        return "", "none", "GROQ_API_KEY missing in .env"
    try:
        buf = io.BytesIO(file_bytes)
        buf.name = filename
        t = Groq(api_key=key).audio.transcriptions.create(
            file=(filename, buf), model="whisper-large-v3", language="en", response_format="json")
        return t.text.strip(), "groq", ""
    except Exception as e:
        return "", "groq", f"Groq Whisper failed: {e}"


def trim_head(file_bytes: bytes, seconds: float = 8.0) -> bytes:
    """First `seconds` of audio as WAV; returns the original bytes if the format can't be read."""
    try:
        import soundfile as sf
        data, sr = sf.read(io.BytesIO(file_bytes), frames=int(seconds * 48000), dtype="float32")
        out = io.BytesIO()
        sf.write(out, data[: int(seconds * sr)], sr, format="WAV")
        return out.getvalue()
    except Exception:
        return file_bytes


def whisper_detect(file_bytes: bytes, filename: str = "audio.wav") -> tuple[str, str, str]:
    """One Groq Whisper call (language not pinned). Returns (language, text, error).
    Only 'english' is trusted; everything else is routed to Sarvam."""
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        return "", "", "GROQ_API_KEY missing in .env"
    try:
        buf = io.BytesIO(file_bytes)
        buf.name = filename
        t = Groq(api_key=key).audio.transcriptions.create(
            file=(filename, buf), model="whisper-large-v3", response_format="verbose_json")
        return (getattr(t, "language", "") or "").strip().lower(), (t.text or "").strip(), ""
    except Exception as e:
        return "", "", f"Groq Whisper failed: {e}"

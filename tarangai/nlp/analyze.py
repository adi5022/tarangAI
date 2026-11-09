"""Code-mixed analysis: normalise Malayalam-English (or romanized Malayalam) text to clean English."""
from .sarvam_llm import chat_json

SYSTEM = """You normalise code-mixed text (Malayalam + English, Malayalam script or romanized "Manglish").
Return ONLY a JSON object with keys:
  "language": one of "english", "malayalam", "code-mixed"
  "english": a faithful, natural English translation. Keep slang meaning, emotion, sarcasm and emoji
            intent. Do not add or remove opinions. If the input is already plain English, copy it unchanged.
  "notes": one short phrase on slang/idioms you resolved (empty string if none)."""


def normalize(text: str) -> dict:
    """Returns {language, english, notes}."""
    out = chat_json(SYSTEM, text)
    return {
        "language": str(out.get("language", "code-mixed")).lower(),
        "english": str(out.get("english") or text).strip(),
        "notes": str(out.get("notes", "")).strip(),
    }

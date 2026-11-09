"""Sentiment engine: one short call that returns a pipe-separated line of fixed-vocabulary fields.

    sentiment|emotion|intensity|sarcastic      e.g.  negative|anger|strong|no

Input is the ORIGINAL text (English, Malayalam, or code-mixed; native script or romanized), with
no translation, so slang, tone and sarcasm are not lost. Every field comes from a closed list, so
output is comparable between runs and the sentiment field can be scored against 3-class datasets."""
from .sarvam_llm import chat

SENTIMENTS = ("positive", "negative", "neutral")
EMOTIONS = ("joy", "love", "surprise", "sadness", "anger", "fear", "disgust", "none")
INTENSITIES = ("mild", "moderate", "strong")
SARCASM = ("yes", "no")

SYSTEM = (
    "You analyse the sentiment of the user's text. It may be English, Malayalam, or a mix of both, "
    "in Malayalam script or romanized (Manglish); read it as written, including slang and emoji.\n"
    "Reply with ONE line of four fields separated by '|', and nothing else:\n"
    "sentiment|emotion|intensity|sarcastic\n"
    f"sentiment: {', '.join(SENTIMENTS)} (for mixed feelings give the dominant one)\n"
    f"emotion: {', '.join(EMOTIONS)} (the main emotion; 'none' if there is no clear emotion)\n"
    f"intensity: {', '.join(INTENSITIES)}\n"
    f"sarcastic: {', '.join(SARCASM)}\n"
    "If the text is sarcastic, give the sentiment it actually means, not the literal words.\n"
    "Example: negative|anger|strong|no"
)


def _pick(parts: list[str], i: int, allowed: tuple[str, ...], default: str) -> str:
    if i < len(parts):
        for a in allowed:
            if a in parts[i]:
                return a
    return default


def classify_sentiment(text: str) -> dict:
    """Returns {sentiment, emotion, intensity, sarcastic}. Raises if no valid sentiment is found."""
    raw = chat(SYSTEM, text, temperature=0.0, max_tokens=24).lower()
    line = next((ln for ln in raw.splitlines() if "|" in ln), raw)
    parts = [p.strip(" .`*") for p in line.split("|")]
    sentiment = _pick(parts, 0, SENTIMENTS, "")
    if not sentiment:  # fall back to a bare label anywhere in the output
        sentiment = next((s for s in SENTIMENTS if s in raw), "")
    if not sentiment:
        raise ValueError(f"Unexpected sentiment output: {raw!r}")
    return {
        "sentiment": sentiment,
        "emotion": _pick(parts, 1, EMOTIONS, "none"),
        "intensity": _pick(parts, 2, INTENSITIES, "moderate"),
        "sarcastic": _pick(parts, 3, SARCASM, "no"),
    }

"""Sentiment engine. Takes clean English text and returns label, confidence and a short reason.

Kept separate from normalisation so it can later be swapped for a dedicated model."""
from .sarvam_llm import chat_json

LABELS = ("positive", "negative", "neutral")

SYSTEM = """You are a sentiment classifier. Classify the sentiment of the given English text.
Return ONLY a JSON object: {"label": "positive" | "negative" | "neutral", "confidence": 0.0-1.0,
"reason": "one short sentence"}. Treat sarcasm by its intended meaning. Mixed sentiment -> the dominant one."""


def classify_sentiment(english_text: str) -> dict:
    out = chat_json(SYSTEM, english_text)
    label = str(out.get("label", "neutral")).lower().strip()
    if label not in LABELS:
        label = "neutral"
    try:
        conf = max(0.0, min(1.0, float(out.get("confidence", 0.5))))
    except (TypeError, ValueError):
        conf = 0.5
    return {"label": label, "confidence": conf, "reason": str(out.get("reason", "")).strip()}

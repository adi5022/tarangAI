"""Sentiment engine: one short call that returns a single label word.

Input is the ORIGINAL text (English, Malayalam, or code-mixed; native script or romanized), with
no translation, so slang, tone and sarcasm are not lost. max_tokens is tiny and reasoning is off,
so the model can only answer with the label."""
from .sarvam_llm import chat

LABELS = ("positive", "negative", "neutral")

SYSTEM = (
    "Classify the sentiment of the user's text. It may be English, Malayalam, or a mix of both, "
    "in Malayalam script or romanized (Manglish); read it as written, including slang and emoji. "
    "Answer with exactly one word: "
    "positive, negative, or neutral. Judge sarcasm by its intended meaning. "
    "For mixed sentiment, answer with the dominant one. No other text."
)


def classify_sentiment(english_text: str) -> str:
    """Returns 'positive' | 'negative' | 'neutral'. Raises if the model gives no valid label."""
    raw = chat(SYSTEM, english_text, temperature=0.0, max_tokens=8).lower()
    for label in LABELS:
        if label in raw:
            return label
    raise ValueError(f"Unexpected sentiment output: {raw!r}")

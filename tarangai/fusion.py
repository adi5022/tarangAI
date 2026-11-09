"""Fuse the text sentiment with the voice dimensions (English / Whisper path only).

Principle: the text labels (sentiment, emotion) are never changed. Voice may
  1. move the intensity one step when arousal clearly disagrees with it, and
  2. add a flag when voice valence points the opposite way from the text sentiment.
Every adjustment is reported, so nothing is hidden from the user.

The cut-offs are PROVISIONAL defaults, not calibrated: tune them on your own recorded clips
(see README, evaluation). Arousal and valence are roughly 0..1, around 0.5 for ordinary speech.
"""
AROUSAL_HIGH = 0.603
AROUSAL_LOW = 0.373
VALENCE_HIGH = 0.557
VALENCE_LOW = 0.342
INTENSITIES = ("mild", "moderate", "strong")


def level(value: float, low: float, high: float) -> str:
    return "high" if value >= high else "low" if value <= low else "mid"


def describe_voice(voice: dict) -> dict:
    """Adds readable levels to the raw scores."""
    return {**voice,
            "arousal_level": level(voice["arousal"], AROUSAL_LOW, AROUSAL_HIGH),
            "valence_level": level(voice["valence"], VALENCE_LOW, VALENCE_HIGH)}


def fuse(text: dict, voice: dict) -> dict:
    """text = classify_sentiment() result, voice = describe_voice() result.
    Returns {'intensity': final, 'intensity_changed_from': str|None, 'flag': str|None}."""
    intensity = text["intensity"]
    i = INTENSITIES.index(intensity)
    changed_from = None
    if voice["arousal_level"] == "high" and i < len(INTENSITIES) - 1:
        i += 1
    elif voice["arousal_level"] == "low" and i > 0:
        i -= 1
    if INTENSITIES[i] != intensity:
        changed_from = intensity

    flag = None
    s, vv = text["sentiment"], voice["valence_level"]
    if s == "positive" and vv == "low":
        flag = "Words sound positive but the voice sounds negative"
    elif s == "negative" and vv == "high":
        flag = "Words sound negative but the voice sounds positive"
    return {"intensity": INTENSITIES[i], "intensity_changed_from": changed_from, "flag": flag}

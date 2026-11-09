"""Voice emotion (arousal / valence / dominance) from the audio itself, using a Hugging Face model.

Model: audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim  (research use only, CC BY-NC-SA 4.0).
Trained on English MSP-Podcast, so it is only applied on the English (Whisper) path.
Outputs three dimensions, each roughly 0..1: arousal (calm..energetic), valence (negative..positive),
dominance (submissive..dominant). The head classes below follow the model card, which needs them to load.

Heavy imports (torch, transformers) are done lazily so the bot still starts if they are missing;
`analyze_voice` then returns None and the rest of the pipeline is unaffected.
"""
import io
import logging
import os
import threading

MODEL_NAME = os.environ.get("VOICE_EMOTION_MODEL", "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim")
SAMPLE_RATE = 16000
MIN_SECONDS = 1.0     # shorter clips give unreliable scores
MAX_SECONDS = 30.0    # keep CPU inference fast; long notes are cut to the first 30 s

_lock = threading.Lock()
_state: dict = {"loaded": False, "failed": "", "model": None, "processor": None}


def enabled() -> bool:
    return os.environ.get("VOICE_EMOTION", "1") != "0"


def load_model() -> bool:
    """Load once. Returns True if the model is ready. Never raises."""
    with _lock:
        if _state["loaded"]:
            return True
        if _state["failed"] or not enabled():
            return False
        try:
            import torch
            import torch.nn as nn
            from transformers import Wav2Vec2Model, Wav2Vec2PreTrainedModel, Wav2Vec2Processor

            class RegressionHead(nn.Module):
                def __init__(self, config):
                    super().__init__()
                    self.dense = nn.Linear(config.hidden_size, config.hidden_size)
                    self.dropout = nn.Dropout(config.final_dropout)
                    self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

                def forward(self, features):
                    x = self.dropout(features)
                    x = torch.tanh(self.dense(x))
                    return self.out_proj(self.dropout(x))

            class EmotionModel(Wav2Vec2PreTrainedModel):
                def __init__(self, config):
                    super().__init__(config)
                    self.config = config
                    self.wav2vec2 = Wav2Vec2Model(config)
                    self.classifier = RegressionHead(config)
                    self.post_init()  # transformers >= 5 requires post_init (older code used init_weights)

                def forward(self, input_values):
                    hidden = self.wav2vec2(input_values)[0]
                    return self.classifier(torch.mean(hidden, dim=1))

            logging.info("Loading voice emotion model %s (first run downloads it)...", MODEL_NAME)
            _state["processor"] = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
            _state["model"] = EmotionModel.from_pretrained(MODEL_NAME).eval()
            _state["loaded"] = True
            logging.info("Voice emotion model ready")
            return True
        except Exception as e:
            _state["failed"] = str(e)
            logging.warning("Voice emotion disabled: %s", e)
            return False


def _decode(audio_bytes: bytes):
    """Decode OGG/Opus (Telegram voice) or any libsndfile format to mono float32 at 16 kHz."""
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    from math import gcd

    data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if sr != SAMPLE_RATE:
        g = gcd(SAMPLE_RATE, sr)
        mono = resample_poly(mono, SAMPLE_RATE // g, sr // g).astype(np.float32)
    return mono


def analyze_voice(audio_bytes: bytes) -> dict | None:
    """Returns {'arousal','valence','dominance'} (floats ~0..1) or None if unavailable or too short."""
    if not load_model():
        return None
    try:
        import torch

        wav = _decode(audio_bytes)
        if len(wav) < MIN_SECONDS * SAMPLE_RATE:
            return None
        wav = wav[: int(MAX_SECONDS * SAMPLE_RATE)]
        inputs = _state["processor"](wav, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        with torch.no_grad():
            logits = _state["model"](inputs["input_values"])[0]
        arousal, dominance, valence = (float(v) for v in logits)  # order from the model card
        return {"arousal": arousal, "valence": valence, "dominance": dominance}
    except Exception as e:
        logging.warning("Voice analysis failed: %s", e)
        return None

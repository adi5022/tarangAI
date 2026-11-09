# tarangAI

Telegram bot that reads the sentiment behind **English** and **Malayalam-English code-mixed**
voice notes and text. For English voice it also listens to *how* something was said (voice
arousal and valence) and combines that with what was said.

## How it works
```
voice note ──► Groq Whisper (language check)
                 ├─ English ──► transcript ──► Sarvam sentiment ──► + voice emotion (HF model) ──► fusion
                 └─ not English ─► Sarvam Saaras (codemix) ─► transcript ─► Sarvam sentiment
text ─────────────────────────────────────────► Sarvam sentiment
```
- **Sentiment runs on the original text.** There is no translation step, so slang, tone and
  sarcasm are not flattened.
- **Voice analysis runs only on the English (Whisper) path.** The voice model is trained on
  English speech. Malayalam, code-mixed audio and typed text skip it.

### Output
```
🎙 You said: Oh great, another delay. Just what I needed.
🌐 Language: en
🔴 Sentiment: negative (strong)
😠 Emotion: anger
😏 Sarcasm detected
🎚 Voice: arousal high (0.71) · valence high (0.70)
⚠️ Words sound negative but the voice sounds positive
⏱ 2.2s
```

| Field | Values | Source |
|---|---|---|
| Sentiment | positive / negative / neutral | Sarvam chat model |
| Emotion | joy, love, surprise, sadness, anger, fear, disgust, none | Sarvam chat model |
| Intensity | mild / moderate / strong | Sarvam, then adjusted by voice |
| Sarcasm | yes / no | Sarvam chat model |
| Voice arousal, valence | about 0 to 1 each | audEERING wav2vec2 model (English audio only) |

The Sarvam model answers in one short line (`negative|anger|strong|no`) and every field comes
from a closed list, so output is comparable between runs.

### Voice and text fusion (`tarangai/fusion.py`)
The text labels (sentiment, emotion) are **never changed** by voice. Voice may:
1. move **intensity one step** when arousal clearly disagrees (high arousal raises it, low arousal lowers it), and
2. add a **flag** when voice valence points the opposite way from the text sentiment
   (for example cheerful voice over negative words, which often means sarcasm).

Every adjustment is shown in the reply.

## Models and services
| Stage | Choice |
|---|---|
| English speech-to-text and language check | Groq `whisper-large-v3` |
| Malayalam / code-mixed speech-to-text | Sarvam `saaras:v3` (codemix, `ml-IN`) |
| Sentiment | Sarvam `sarvam-105b-conversations` (chat API) |
| Voice emotion | `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim` (Hugging Face) |

Whisper is trusted only for "is this English?". Anything else goes to Sarvam. There is no
silent fallback between STT providers: Malayalam is never sent to Whisper.

## Setup
Python 3.13 is what this was developed on.

    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt
    copy .env.example .env      # then fill in the keys
    python -m tarangai

The first start downloads the voice model (several hundred MB, about 5 minutes) and loads it
(about 40 s). After that it is cached and the bot prints `tarangAI bot running` once ready.

| Key | Purpose |
|---|---|
| `TELEGRAM_TOKEN` | Bot token from @BotFather |
| `SARVAM_AI` | Saaras speech-to-text and the Sarvam chat model |
| `GROQ_API_KEY` | Whisper (`GROQ_API_KEY_2` / `_3` are used as backups if the first hits its limit) |

Optional settings:

| Variable | Effect |
|---|---|
| `VOICE_EMOTION=0` | Turn voice analysis off (the bot works without it) |
| `DEBUG_TRANSCRIPTS=1` | Log transcripts, sentiment and voice scores to the console. Off by default because transcripts are private |
| `SARVAM_MODEL` | Override the Sarvam chat model name |
| `VOICE_EMOTION_MODEL` | Override the Hugging Face model |

## Calibrating the voice cut-offs
Voice arousal and valence are continuous scores. "Low / mid / high" needs cut-offs, which differ by
speaker and microphone, so they were measured rather than guessed. `bench/app.py` is a Streamlit
tool with a guided 25-clip recording script. It averages your clips for each intended level, puts
each cut-off halfway between neighbouring averages, and can write them into `fusion.py`.

    py -m streamlit run bench/app.py

Result of the calibration run (1 speaker, 25 clips, 5 per low/high level, 15 mid):

| Dimension | low mean | mid mean | high mean | Cut-offs set (low / high) |
|---|---|---|---|---|
| Arousal | 0.275 | 0.471 | 0.735 | 0.373 / 0.603 |
| Valence | 0.282 | 0.401 | 0.714 | 0.342 / 0.557 |

The class means are in the expected order for both dimensions. The mid-arousal clips vary a lot
(std 0.12), so arousal labels near the cut-offs are noisy. Recordings are stored in
`bench/data/` (git-ignored, private).

## Layout
- `tarangai/stt/`: speech-to-text and language routing (adapted from Sahayi)
- `tarangai/nlp/`: Sarvam chat client and the sentiment classifier
- `tarangai/audio/`: voice emotion model
- `tarangai/fusion.py`: voice and text fusion rules and cut-offs
- `tarangai/pipeline.py`: routing
- `tarangai/bot.py`: Telegram handlers
- `bench/`: temporary calibration tool (can be deleted)

## Credits
The STT routing and the Sarvam/Whisper calls are adapted from
[Sahayi](https://github.com/adi5022/sahayiAI).

## Licence notes
The voice emotion model is **CC BY-NC-SA 4.0, research use only**. This project is therefore
non-commercial unless that model is replaced or licensed.

## Limitations
- **Voice model is English-trained** (MSP-Podcast). It is not applied to Malayalam or code-mixed
  speech, and its behaviour there is unknown.
- **Calibration covers one speaker and one microphone.** Other voices may land elsewhere on the
  scale. Treat voice scores as an indicative second signal, not a verdict.
- **Intensity and sarcasm are the least reliable fields.** The sentiment label is the most
  dependable output.
- **No formal evaluation yet.** Results so far come from manual spot checks, not a labelled
  test set, so no accuracy figure is claimed.
- Sarvam STT is set to `ml-IN`, so Hindi-English speech is not supported.
- Sentiment from a language model can disagree with human labels, especially on mixed feelings.

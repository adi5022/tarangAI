# tarangAI

Telegram bot that reads the sentiment behind English and Malayalam-English
code-mixed speech and text.

## How it works
```
voice note ──► Groq Whisper (language check)
                 ├─ English ──────────────────────────────► sentiment
                 └─ not English ─► Sarvam Saaras (codemix) ─► Sarvam-M normalise ─► sentiment
text ─────────► Sarvam-M normalise (plain English passes through) ─► sentiment
```
Reply: transcript, language, clean English, sentiment label with confidence and a short reason.

## Setup
    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt
    copy .env.example .env      # fill in the three keys
    python -m tarangai

| Key | Purpose |
|---|---|
| TELEGRAM_TOKEN | Bot token from @BotFather |
| SARVAM_AI | Saaras speech-to-text and Sarvam-M |
| GROQ_API_KEY | Whisper for English speech |

## Layout
- `tarangai/stt/`: speech-to-text and language routing
- `tarangai/nlp/`: Sarvam-M client, code-mix normalisation, sentiment
- `tarangai/pipeline.py`: routing logic
- `tarangai/bot.py`: Telegram handlers

## Credits
The STT routing and Sarvam/Whisper calls are adapted from
[Sahayi](https://github.com/adi5022/sahayiAI).

## Limitations
Sarcasm, slang and noisy audio reduce accuracy. Sarvam STT is set to `ml-IN`, so
Hindi-English speech is not supported. Sentiment is indicative, not authoritative.

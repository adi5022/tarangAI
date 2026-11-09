"""Telegram bot. Voice note or text in, transcript + normalised English + sentiment out."""
import asyncio
import hashlib
import logging
import os
from collections import OrderedDict

from dotenv import load_dotenv
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (ApplicationBuilder, ApplicationHandlerStop, CommandHandler, ContextTypes, MessageHandler,
                          TypeHandler, filters)

from .pipeline import process

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)  # httpx URLs contain the bot token
logging.getLogger("httpcore").setLevel(logging.WARNING)

WELCOME = (
    "Hi, I'm tarangAI.\n\n"
    "Send me a voice note or text in English or Malayalam-English (mixed) and I'll "
    "transcribe it and tell you the sentiment."
)
ERROR = "Sorry, something went wrong on my side. Please try again."
ICON = {"positive": "🟢", "negative": "🔴", "neutral": "⚪"}
EMOJI = {"joy": "😄", "love": "❤️", "surprise": "😲", "sadness": "😢", "anger": "😠", "fear": "😨", "disgust": "🤢", "none": "😐"}


def format_reply(out: dict) -> str:
    lines = []
    if out["transcript"]:  # always show what was heard, even if a later step failed
        lines.append(f"🎙 You said: {out['transcript']}")
    if out["error"]:
        lines.append(f"⚠️ {out['error']}")
        return "\n".join(lines)
    if out["language"]:  # known for voice only
        lines.append(f"🌐 Language: {out['language']}")
    s, fu, vo = out["sentiment"], out.get("fusion"), out.get("voice")
    intensity = fu["intensity"] if fu else s["intensity"]  # intensity after voice adjustment, if any
    lines.append(f"{ICON[s['sentiment']]} Sentiment: {s['sentiment']} ({intensity})")
    lines.append(f"{EMOJI.get(s['emotion'], '')} Emotion: {s['emotion']}".strip())
    if s["sarcastic"] == "yes":
        lines.append("😏 Sarcasm detected")
    if vo:  # English voice only
        lines.append(f"🎚 Voice: arousal {vo['arousal_level']} ({vo['arousal']:.2f}) · valence {vo['valence_level']} ({vo['valence']:.2f})")
        if fu["intensity_changed_from"]:
            lines.append(f"↕ Intensity adjusted by voice: {fu['intensity_changed_from']} → {fu['intensity']}")
        if fu["flag"]:
            lines.append(f"⚠️ {fu['flag']}")
    lines.append(f"⏱ {out['seconds']}s")
    return "\n".join(lines)


async def reply(update: Update, context: ContextTypes.DEFAULT_TYPE, **kw) -> None:
    await context.bot.send_chat_action(update.effective_chat.id, ChatAction.TYPING)
    try:
        out = await asyncio.to_thread(process, **kw)
    except Exception:
        logging.exception("pipeline failed")
        await update.effective_message.reply_text(ERROR)
        return
    logging.info("lang=%s sentiment=%s provider=%s %ss", out["language"] or "-", (out["sentiment"] or {}).get("sentiment", "-"),
                 out["provider"] or "-", out["seconds"])
    if os.getenv("DEBUG_TRANSCRIPTS") == "1":  # opt-in only: transcripts are private messages
        logging.info("DEBUG transcript=%r sentiment=%s voice=%s error=%s", out["transcript"], out["sentiment"],
                     {k: round(v, 3) for k, v in (out["voice"] or {}).items() if isinstance(v, float)} or None,
                     out["error"] or None)
    await update.effective_message.reply_text(format_reply(out))


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(WELCOME)


async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await reply(update, context, text=update.message.text or "")


async def on_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    media = update.message.voice or update.message.audio
    data = bytes(await (await media.get_file()).download_as_bytearray())  # in memory, nothing saved to disk
    await reply(update, context, audio=(data, "voice.ogg", getattr(media, "duration", None)))


_SEEN: "OrderedDict[int, None]" = OrderedDict()


async def drop_duplicates(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Webhook mode: if the host was asleep, Telegram may redeliver an update it thinks timed out.
    Remember recent update ids and skip repeats so one voice note is never processed twice."""
    if update.update_id in _SEEN:
        logging.info("Skipping duplicate update %s", update.update_id)
        raise ApplicationHandlerStop
    _SEEN[update.update_id] = None
    while len(_SEEN) > 500:
        _SEEN.popitem(last=False)


def main() -> None:
    load_dotenv()
    token = os.getenv("TELEGRAM_TOKEN")
    if not token:
        raise SystemExit("TELEGRAM_TOKEN missing in .env")
    app = ApplicationBuilder().token(token).build()
    app.add_handler(TypeHandler(Update, drop_duplicates), group=-1)
    app.add_handler(CommandHandler(["start", "help"], start))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, on_voice))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    from .audio import load_model  # preload so the first English voice note is not slow
    load_model()

    # Webhook mode for hosts that sleep when idle (for example Render's free tier): Telegram sends each message
    # to a public HTTPS URL, which wakes the host. Used only when a public URL is configured; otherwise polling.
    public_url = (os.getenv("WEBHOOK_URL") or os.getenv("RENDER_EXTERNAL_URL") or "").strip()
    if public_url:
        digest = hashlib.sha256(token.encode()).hexdigest()
        path = os.getenv("WEBHOOK_PATH") or digest[:32]      # unguessable URL path
        secret = os.getenv("WEBHOOK_SECRET") or digest[32:64]  # header Telegram sends; others are rejected
        port = int(os.getenv("PORT", "8080"))
        logging.info("tarangAI bot running in webhook mode on port %s", port)
        app.run_webhook(listen="0.0.0.0", port=port, url_path=path,
                        webhook_url=f"{public_url.rstrip('/')}/{path}", secret_token=secret)
    else:
        logging.info("tarangAI bot running (polling)")
        app.run_polling()


if __name__ == "__main__":
    main()

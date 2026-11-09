"""TEMPORARY voice benchmark: record clips (guided script or free), score them with the voice model,
and calibrate the fusion cut-offs from your own results.

Run:  py -m streamlit run bench/app.py

Clips and scores are stored in bench/data/ (git-ignored, private).
"""
import hashlib
import re
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tarangai import fusion  # noqa: E402
from tarangai.audio import voice_emotion as ve  # noqa: E402

DATA = ROOT / "bench" / "data"
CSV = DATA / "results.csv"
FUSION_PY = ROOT / "tarangai" / "fusion.py"
DATA.mkdir(parents=True, exist_ok=True)
COLS = ["id", "time", "speaker", "script_id", "sentence", "intended_arousal", "intended_valence",
        "arousal", "valence", "dominance", "infer_s", "file"]
LEVELS = ["low", "mid", "high"]
AR_HELP = {"low": "calm / flat / tired", "mid": "normal speaking", "high": "agitated / loud / excited"}
VA_HELP = {"low": "negative tone", "mid": "neutral tone", "high": "positive tone"}

# ------------------------------------------------------------------ guided script (25 clips)
# Part A: neutral wording, said three ways -> arousal low / mid / high   (the mid takes also serve as valence = neutral)
# Part B: emotional wording, said with natural energy -> valence low / high
NEUTRAL = ["I have a meeting at five today.", "The bus leaves from the main gate.", "Please send me the report tomorrow.",
           "The shop is next to the library.", "We are going to the market now."]
NEGATIVE = ["I can't believe how bad this service is.", "This is the worst day I have had in weeks.",
            "Nothing ever works properly here.", "I am so disappointed with the result.", "I really regret coming here."]
POSITIVE = ["That was the best meal I have had in ages!", "I am so happy with how it turned out.",
            "This is such wonderful news, thank you!", "I love how well everything worked today.", "What a fantastic surprise, I'm thrilled!"]
HOW = {
    ("low", "mid"): "Say it slowly, flat and quiet, like you are tired. No emotion in the words.",
    ("mid", "mid"): "Say it in your normal everyday voice. No emotion.",
    ("high", "mid"): "Say it LOUD, fast and urgent, like you are agitated. The words stay neutral.",
    ("mid", "low"): "Say it with a clearly NEGATIVE, unhappy tone, at your normal energy.",
    ("mid", "high"): "Say it with a clearly POSITIVE, happy tone, at your normal energy.",
}
SCRIPT = []
for n, s in enumerate(NEUTRAL, 1):
    for a in LEVELS:
        SCRIPT.append({"id": f"A{n}-{a}", "sentence": s, "arousal": a, "valence": "mid", "how": HOW[(a, "mid")]})
for n, s in enumerate(NEGATIVE, 1):
    SCRIPT.append({"id": f"B{n}-neg", "sentence": s, "arousal": "mid", "valence": "low", "how": HOW[("mid", "low")]})
for n, s in enumerate(POSITIVE, 1):
    SCRIPT.append({"id": f"B{n}-pos", "sentence": s, "arousal": "mid", "valence": "high", "how": HOW[("mid", "high")]})

st.set_page_config(page_title="Voice benchmark", page_icon="🎚", layout="wide")


@st.cache_resource(show_spinner="Loading the voice model (about 40 s the first time)...")
def get_model() -> bool:
    return ve.load_model()


def load_results() -> pd.DataFrame:
    if CSV.exists():
        df = pd.read_csv(CSV)
        for c in COLS:
            if c not in df.columns:
                df[c] = ""
        return df
    return pd.DataFrame(columns=COLS)


def save_results(df: pd.DataFrame) -> None:
    df.to_csv(CSV, index=False)


def add_clip(audio: bytes, ext: str, speaker: str, script_id: str, sentence: str, ia: str, iv: str) -> dict | None:
    t0 = time.perf_counter()
    scores = ve.analyze_voice(audio)
    dt = time.perf_counter() - t0
    if not scores:
        return None
    clip_id = int(time.time() * 1000)
    fname = f"{clip_id}_{speaker or 'anon'}.{ext}"
    (DATA / fname).write_bytes(audio)
    row = {"id": clip_id, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "speaker": speaker, "script_id": script_id,
           "sentence": sentence, "intended_arousal": ia, "intended_valence": iv, "infer_s": round(dt, 2),
           "file": fname, **scores}
    save_results(pd.concat([load_results(), pd.DataFrame([row])], ignore_index=True))
    return row


def midpoints(df: pd.DataFrame, intended: str, score: str):
    """(low_cut, high_cut, means, counts) from class means, or None if a class has too few clips."""
    g = df.groupby(intended)[score]
    means, counts = g.mean(), g.count()
    if any(counts.get(k, 0) < 2 for k in LEVELS):
        return None
    return (means["low"] + means["mid"]) / 2, (means["mid"] + means["high"]) / 2, means, counts


def accuracy(df: pd.DataFrame, intended: str, score: str, low: float, high: float) -> float:
    pred = df[score].apply(lambda v: fusion.level(v, low, high))
    return float((pred == df[intended]).mean())


def apply_to_fusion(sug: dict) -> None:
    """Rewrite the four cut-off constants in tarangai/fusion.py."""
    src = FUSION_PY.read_text(encoding="utf-8")
    for const, val in [("AROUSAL_HIGH", sug["arousal"][1]), ("AROUSAL_LOW", sug["arousal"][0]),
                       ("VALENCE_HIGH", sug["valence"][1]), ("VALENCE_LOW", sug["valence"][0])]:
        src, n = re.subn(rf"^{const} = [0-9.]+", f"{const} = {val}", src, flags=re.M)
        if n != 1:
            raise RuntimeError(f"Could not find {const} in fusion.py")
    FUSION_PY.write_text(src, encoding="utf-8")


st.title("🎚 Voice benchmark (temporary)")
st.caption("Find arousal / valence cut-offs for fusion.py from your own recordings. English speech only.")

if not get_model():
    st.error(f"Voice model failed to load: {ve._state['failed'] or 'disabled (VOICE_EMOTION=0?)'}")
    st.stop()

tab_rec, tab_res, tab_thr = st.tabs(["1 · Record", "2 · Results", "3 · Thresholds"])

# ---------------------------------------------------------------- record
with tab_rec:
    speaker = st.text_input("Speaker name", value="me")
    mode = st.radio("Mode", ["Guided script (recommended)", "Free recording"], horizontal=True)
    df_now = load_results()
    done_ids = set(df_now["script_id"].astype(str))

    if mode.startswith("Guided"):
        done = sum(item["id"] in done_ids for item in SCRIPT)
        st.progress(done / len(SCRIPT), text=f"{done} of {len(SCRIPT)} script clips recorded")
        pending = [i for i, item in enumerate(SCRIPT) if item["id"] not in done_ids]
        default = pending[0] if pending else 0
        idx = st.selectbox("Script item", range(len(SCRIPT)), index=default,
                           format_func=lambda i: f"{'✅' if SCRIPT[i]['id'] in done_ids else '⬜'} {SCRIPT[i]['id']} · {SCRIPT[i]['sentence']}")
        item = SCRIPT[idx]
        st.markdown(f"### “{item['sentence']}”")
        st.markdown(f"**How to say it:** {item['how']}")
        st.caption(f"Intended arousal: {item['arousal']} ({AR_HELP[item['arousal']]}) · "
                   f"intended valence: {item['valence']} ({VA_HELP[item['valence']]})")
        sid, sentence, ia, iv, key = item["id"], item["sentence"], item["arousal"], item["valence"], f"rec_{item['id']}"
    else:
        c1, c2 = st.columns(2)
        sentence = c1.text_input("Sentence (optional note)", value="I have a meeting at five today.")
        ia = c2.selectbox("Intended AROUSAL", LEVELS, index=1, format_func=lambda k: f"{k} · {AR_HELP[k]}")
        iv = c2.selectbox("Intended VALENCE", LEVELS, index=1, format_func=lambda k: f"{k} · {VA_HELP[k]}")
        sid, key = "", "rec_free"

    st.info("Record in the room and with the microphone you will really use. Clips should be 1 to 10 seconds "
            "(under 1 s is rejected). Exaggerate a little: the model needs clear differences.")
    rec = st.audio_input("Record a clip", key=key)
    up = st.file_uploader("...or upload a clip (wav / ogg / mp3 / flac), for example a saved Telegram voice note",
                          type=["wav", "ogg", "opus", "mp3", "flac"], key=f"up_{key}")
    audio, ext = (None, "")
    if rec is not None:
        audio, ext = rec.getvalue(), "wav"
    elif up is not None:
        audio, ext = up.getvalue(), up.name.rsplit(".", 1)[-1].lower()

    if audio:
        st.audio(audio)
        digest = hashlib.md5(audio).hexdigest()
        if st.button("Analyze and save", type="primary", disabled=st.session_state.get("last_saved") == digest):
            row = add_clip(audio, ext, speaker, sid, sentence, ia, iv)
            if row is None:
                st.warning("Could not score this clip (too short, under 1 s, or unreadable).")
            else:
                st.session_state["last_saved"] = digest
                st.session_state["last_row"] = row
                st.rerun()
        if st.session_state.get("last_saved") == digest and "last_row" in st.session_state:
            r = st.session_state["last_row"]
            m1, m2, m3 = st.columns(3)
            m1.metric("Arousal", f"{r['arousal']:.3f}", fusion.level(r["arousal"], fusion.AROUSAL_LOW, fusion.AROUSAL_HIGH))
            m2.metric("Valence", f"{r['valence']:.3f}", fusion.level(r["valence"], fusion.VALENCE_LOW, fusion.VALENCE_HIGH))
            m3.metric("Dominance", f"{r['dominance']:.3f}")
            st.caption(f"Saved. Level shown uses the current cut-offs in fusion.py. Scoring took {r['infer_s']} s. "
                       "Pick the next script item above.")

# ---------------------------------------------------------------- results
with tab_res:
    df = load_results()
    if df.empty:
        st.write("No clips yet. Record some in tab 1.")
    else:
        st.dataframe(df.drop(columns=["id", "file"]), width="stretch", hide_index=True)
        b1, b2 = st.columns([1, 5])
        if b1.button("Delete last clip"):
            last = df.iloc[-1]
            (DATA / str(last["file"])).unlink(missing_ok=True)
            save_results(df.iloc[:-1])
            st.rerun()
        b2.download_button("Download results.csv", df.to_csv(index=False), "results.csv", "text/csv")

        st.subheader("Mean score by what you intended")
        s1, s2 = st.columns(2)
        s1.write("Arousal by intended arousal")
        s1.dataframe(df.groupby("intended_arousal")["arousal"].agg(["count", "mean", "std"]).reindex(LEVELS).round(3))
        s2.write("Valence by intended valence")
        s2.dataframe(df.groupby("intended_valence")["valence"].agg(["count", "mean", "std"]).reindex(LEVELS).round(3))
        st.write("Each point is one clip. Colour = intended arousal.")
        st.scatter_chart(df, x="valence", y="arousal", color="intended_arousal")

# ---------------------------------------------------------------- thresholds
with tab_thr:
    df = load_results()
    st.write(f"Current cut-offs in `fusion.py`: arousal low **{fusion.AROUSAL_LOW}** / high **{fusion.AROUSAL_HIGH}**, "
             f"valence low **{fusion.VALENCE_LOW}** / high **{fusion.VALENCE_HIGH}**.")
    if df.empty:
        st.write("Record clips first.")
        st.stop()

    suggestion, ordered = {}, True
    for name, intended, score, cur in [("arousal", "intended_arousal", "arousal", (fusion.AROUSAL_LOW, fusion.AROUSAL_HIGH)),
                                       ("valence", "intended_valence", "valence", (fusion.VALENCE_LOW, fusion.VALENCE_HIGH))]:
        st.subheader(name.capitalize())
        res = midpoints(df, intended, score)
        if res is None:
            counts = df[intended].value_counts().reindex(LEVELS).fillna(0).astype(int).to_dict()
            st.warning(f"Need at least 2 clips for each of low, mid and high. Have: {counts}")
            continue
        lo, hi, means, counts = res
        suggestion[name] = (round(lo, 3), round(hi, 3))
        if not (means["low"] < means["mid"] < means["high"]):
            ordered = False
            st.error("Class means are not in order (low < mid < high). The model is not separating your recordings "
                     "on this dimension, so thresholds from them would be unreliable. Exaggerate more, or record more clips.")
        a, b = st.columns(2)
        a.metric("Suggested low cut-off", f"{lo:.3f}", f"current {cur[0]}", delta_color="off")
        b.metric("Suggested high cut-off", f"{hi:.3f}", f"current {cur[1]}", delta_color="off")
        st.write(f"Level accuracy on your clips, current cut-offs: **{accuracy(df, intended, score, *cur):.0%}**. "
                 f"Suggested cut-offs: **{accuracy(df, intended, score, lo, hi):.0%}** "
                 f"(measured on the same clips used to pick them, so it is optimistic).")

    if len(suggestion) == 2:
        st.subheader("Use these values")
        block = (f"AROUSAL_HIGH = {suggestion['arousal'][1]}\nAROUSAL_LOW = {suggestion['arousal'][0]}\n"
                 f"VALENCE_HIGH = {suggestion['valence'][1]}\nVALENCE_LOW = {suggestion['valence'][0]}")
        st.code(block, language="python")
        if not ordered:
            st.warning("The means are out of order for at least one dimension, so applying is disabled.")
        elif st.button("Apply to tarangai/fusion.py", type="primary"):
            apply_to_fusion(suggestion)
            st.success("fusion.py updated. Restart the bot to use the new cut-offs. "
                       "(This page still shows the old values until you refresh it.)")
        st.caption("Calibrated on your voice and microphone only. A different speaker may land elsewhere, "
                   "which is a limitation worth stating in the report.")

"""Thin client for the Sarvam chat model (Sarvam-M)."""
import json
import os
import re

import requests

URL = "https://api.sarvam.ai/v1/chat/completions"
# sarvam-m is deprecated. The "-conversations" model answers directly; plain sarvam-105b spends small
# max_tokens budgets on hidden reasoning and can return an empty answer. The API rejects reasoning_effort "none".
MODEL = os.environ.get("SARVAM_MODEL", "sarvam-105b-conversations")


def chat(system: str, user: str, temperature: float = 0.0, max_tokens: int = 600) -> str:
    key = os.environ.get("SARVAM_AI")
    if not key:
        raise RuntimeError("SARVAM_AI key missing in .env")
    r = requests.post(
        URL,
        headers={"api-subscription-key": key.strip(), "Content-Type": "application/json"},
        json={"model": MODEL, "temperature": temperature, "max_tokens": max_tokens,
              "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]},
        timeout=45,
    )
    if not r.ok:  # surface Sarvam's own error message, not just the status code
        raise RuntimeError(f"Sarvam chat {r.status_code}: {r.text[:300]}")
    text = r.json()["choices"][0]["message"]["content"] or ""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()  # drop reasoning block if present


def chat_json(system: str, user: str, **kw) -> dict:
    """Chat call that must return a JSON object; tolerant of code fences and extra text, one retry."""
    for attempt in range(2):
        raw = chat(system, user, **kw)
        m = re.search(r"\{.*\}", raw, flags=re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
        user = user + "\n\nReturn ONLY one valid JSON object, nothing else."
    raise ValueError("Sarvam did not return valid JSON")

"""Flexible model swap-point: gemini | strands | none.

Never logs or returns API keys. Offline pytest and missing credentials
use provider=none (deterministic parse + BM25).
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

from cancer_output_atlas.runtime import STRANDS_AVAILABLE, build_agent


def _offline() -> bool:
    return os.environ.get("COA_OFFLINE", "").strip().lower() in {"1", "true", "yes"}


def gemini_api_key() -> str:
    return (
        os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
    ).strip()


def provider() -> str:
    """Resolved provider. Default gemini if a Gemini key is present, else none."""
    if _offline():
        return "none"
    explicit = (os.environ.get("COA_MODEL_PROVIDER") or "").strip().lower()
    if explicit == "none":
        return "none"
    if explicit == "strands":
        return "strands" if STRANDS_AVAILABLE else "none"
    if explicit == "gemini":
        return "gemini" if gemini_api_key() else "none"
    if gemini_api_key():
        return "gemini"
    return "none"


def model_id() -> str:
    explicit = (os.environ.get("COA_MODEL") or os.environ.get("STRANDS_MODEL") or "").strip()
    if explicit:
        return explicit
    if provider() == "gemini":
        return "gemini-3.5-flash"
    return ""


def model_ready() -> bool:
    return provider() in {"gemini", "strands"}


def _agent_text(result: Any) -> str:
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    for attr in ("text", "output", "content"):
        val = getattr(result, attr, None)
        if isinstance(val, str) and val.strip():
            return val
    msg = getattr(result, "message", None)
    if isinstance(msg, str):
        return msg
    if isinstance(msg, dict):
        content = msg.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            bits = []
            for part in content:
                if isinstance(part, dict) and part.get("text"):
                    bits.append(str(part["text"]))
                elif isinstance(part, str):
                    bits.append(part)
            if bits:
                return "\n".join(bits)
    return str(result)


def _gemini_generate(system: str, user: str, timeout: float) -> str | None:
    key = gemini_api_key()
    if not key:
        return None
    mid = model_id() or "gemini-3.5-flash"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{mid}:generateContent"
    body = json.dumps(
        {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"temperature": 0.0, "maxOutputTokens": 256},
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json; charset=utf-8",
            "x-goog-api-key": key,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return None
    texts: list[str] = []
    for cand in payload.get("candidates") or []:
        content = (cand or {}).get("content") or {}
        for part in content.get("parts") or []:
            t = part.get("text")
            if t:
                texts.append(str(t))
    out = "\n".join(texts).strip()
    return out or None


def _strands_generate(system: str, user: str, timeout: float) -> str | None:
    if not STRANDS_AVAILABLE:
        return None
    mid = model_id() or None

    def _call() -> str:
        agent = build_agent(tools=[], system_prompt=system, model=mid)
        return _agent_text(agent(user))

    try:
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return (pool.submit(_call).result(timeout=timeout) or "").strip() or None
    except Exception:
        return None


def generate_text(system: str, user: str, *, timeout: float = 8.0) -> str | None:
    """Generate text via the configured provider. None on skip/failure. Never logs keys."""
    prov = provider()
    if prov == "gemini":
        return _gemini_generate(system, user, timeout)
    if prov == "strands":
        return _strands_generate(system, user, timeout)
    return None

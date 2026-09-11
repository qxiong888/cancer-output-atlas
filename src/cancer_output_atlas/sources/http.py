"""Tiny HTTP helper. Only used after safety.one_click_allowed."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from cancer_output_atlas.safety import refuse_download, user_agent


class FetchError(RuntimeError):
    pass


def get_text(url: str, timeout: float = 20.0) -> str:
    refuse_download(url)
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": user_agent(),
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read(12_000_000)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise FetchError(f"fetch failed: {url} ({exc})") from exc
    return raw.decode("utf-8", errors="replace")


def get_json(url: str, timeout: float = 20.0) -> Any:
    text = get_text(url, timeout=timeout)
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise FetchError(f"non-JSON response from {url}") from exc


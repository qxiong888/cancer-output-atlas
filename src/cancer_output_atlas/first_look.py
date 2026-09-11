"""Lazy-path picks: short why + top semantic scores only."""

from __future__ import annotations

import re
from typing import Any

_CJK = re.compile(r"[\u3400-\u9fff]")

_STOP_OVERLAP = frozenset(
    {
        "find",
        "public",
        "resources",
        "resource",
        "related",
        "dataset",
        "datasets",
        "i",
        "can",
        "the",
        "a",
        "an",
        "of",
        "and",
        "or",
        "to",
        "for",
        "with",
        "in",
        "on",
        "reuse",
        "use",
        "used",
        "using",
        "找",
        "的",
        "和",
        "与",
        "资源",
        "公开",
        "可以",
        "我",
        "要",
        "做",
        "能用",
        "相关",
    }
)


def why_short(hit: dict[str, Any]) -> str:
    why = str(hit.get("why") or "").strip()
    if (
        why
        and not why.startswith("profile_fit=")
        and "goal-token overlap" not in why
        and not _CJK.search(why)
    ):
        return why
    overlap = [
        str(x)
        for x in (hit.get("goal_overlap") or [])
        if x and str(x).lower() not in _STOP_OVERLAP and not _CJK.search(str(x))
    ][:4]
    if overlap:
        return "Matches " + " ".join(overlap)
    title = str(hit.get("title") or "").strip()
    return title[:40]


def first_look_hits(by_ods: dict[str, list[dict[str, Any]]], n: int = 3) -> list[dict[str, Any]]:
    """Top topic scores only. Callability must never outrank a better topic match."""
    pool: list[dict[str, Any]] = []
    seen: set[str] = set()
    for hits in by_ods.values():
        for h in hits:
            oid = str(h.get("output_id") or "")
            if oid in seen:
                continue
            seen.add(oid)
            pool.append(h)
    pool.sort(key=lambda h: (-float(h.get("rank_score") or 0), str(h.get("output_id") or "")))
    return pool[:n]


"""Optional model slot parse + retrieved-only why (Strict mode).

Provider is selected by model_config (gemini | strands | none).
The model never searches, never invents GEO/NCT/DOI/IDs, and never adds
resources that were not retrieved. Offline pytest and missing keys use
the lexical parser in rank.py.
"""

from __future__ import annotations

import json
import re
from typing import Any

_ODS = frozenset(
    {"data", "software", "tool", "method", "model", "trial_result", "biospecimen"}
)
_ID_LOOKS = re.compile(
    r"\b(?:GSE|GSM|GPL|NCT|PMID|PMC|SRR|SRP|PRJNA|E-MTAB|E-GEOD)[-_]?\d+\b"
    r"|\b10\.\d{4,}/[^\s]+",
    re.I,
)
_JSON_OBJ = re.compile(r"\{.*\}", re.S)

SLOT_SYSTEM = """You parse a reuse-search goal into JSON slots. You do not search or rank.
You do not invent GEO, NCT, DOI, SRA, PMID, or any accession.
Return ONLY a JSON object:
{"topic_phrases": ["virtual cell"], "ods_types": ["data"], "must_not": []}
Rules:
- topic_phrases are distinctive multiword or entity strings from the user's text
  (virtual cell, 虚拟细胞, pembrolizumab, nsclc). Never emit filler bigrams
  such as "cell related", "related dataset", or "cell dataset".
- related/relating/regarding/about/dataset/datasets/resources/resource are not topics.
  dataset/data/geo/数据 → ods_types "data". trial/nct/试验 → trial_result.
- Lone "cell" or journal "Cell" is not a topic. "virtual cell" is.
- If the leftover after removing filler and type words is empty, topic_phrases is [].
- Echo an accession only if the user typed it. Otherwise omit.
"""

WHY_SYSTEM = """You write one short Chinese reason for why a retrieved atlas row
matches the user's goal. Use ONLY the provided row fields (title, summary, ids,
landing). Do not add identifiers, papers, or facts that are not in those fields.
Return ONLY a short line like: 标题含 virtual cell
Never invent GEO/NCT/DOI IDs.
"""


def _extract_json(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    raw = text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        m = _JSON_OBJ.search(raw)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return obj if isinstance(obj, dict) else None


def _invented_id(phrase: str, goal: str) -> bool:
    goal_u = goal.upper()
    for m in _ID_LOOKS.finditer(phrase or ""):
        if m.group(0).upper() not in goal_u:
            return True
    return False


def sanitize_slots(raw: dict[str, Any] | None, goal: str) -> dict[str, Any] | None:
    if not raw:
        return None
    phrases: list[str] = []
    for p in raw.get("topic_phrases") or []:
        s = " ".join(str(p).lower().split())
        if not s or _invented_id(s, goal):
            continue
        phrases.append(s)
    types: list[str] = []
    for t in raw.get("ods_types") or []:
        slug = str(t).strip().lower()
        if slug in _ODS and slug not in types:
            types.append(slug)
    must_not: list[str] = []
    for p in raw.get("must_not") or []:
        s = " ".join(str(p).lower().split())
        if not s or _invented_id(s, goal) or s in phrases:
            continue
        must_not.append(s)
    if not phrases and not types and not must_not:
        return None
    return {"topic_phrases": phrases, "ods_types": types, "must_not": must_not}


def try_llm_slots(goal: str) -> dict[str, Any] | None:
    from cancer_output_atlas.model_config import generate_text, model_ready

    if not model_ready():
        return None
    text = generate_text(SLOT_SYSTEM, f"Parse this goal into slots.\nGoal: {goal}", timeout=8.0)
    return sanitize_slots(_extract_json(text or ""), goal)


def try_llm_why(goal: str, row: dict[str, Any], fallback: str) -> str:
    from cancer_output_atlas.model_config import generate_text, model_ready

    if not model_ready():
        return fallback
    allowed = " ".join(
        str(row.get(k) or "")
        for k in ("title", "summary", "landing_url", "output_id", "why")
    )
    ids = " ".join(str(x) for x in (row.get("ids") or []))
    allowed = f"{allowed} {ids}"
    user = (
        "Goal: "
        + goal
        + "\nRow title: "
        + str(row.get("title") or "")
        + "\nRow ids: "
        + ids
        + "\nRow landing: "
        + str(row.get("landing_url") or "")
        + "\nLexical why: "
        + str(row.get("why") or fallback)
    )
    text = generate_text(WHY_SYSTEM, user, timeout=6.0)
    if not text:
        return fallback
    text = " ".join(text.split())
    if _invented_id(text, allowed):
        return fallback
    return text[:80] or fallback


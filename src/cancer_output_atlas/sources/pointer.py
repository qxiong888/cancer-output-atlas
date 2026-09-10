"""Object-store / landing pointers. Never list or download the bucket."""

from __future__ import annotations

import json

from cancer_output_atlas.node_features import structured_topics
from cancer_output_atlas.schema import Identifier, OutputRecord
from cancer_output_atlas.sources.catalog import fixtures_dir
from cancer_output_atlas.sources.http import FetchError


def _summary_with_keywords(note: dict) -> str:
    summary = str(note.get("summary") or "")
    keywords = note.get("keywords") or []
    if keywords:
        summary = summary + "\nKeywords: " + ", ".join(str(k) for k in keywords)
    return summary


def _merged_catalog_notes() -> dict:
    """catalog_notes.json first; extra file adds keys only (never overwrites)."""
    notes: dict = {}
    for name in ("catalog_notes.json", "catalog_notes_extra.json", "quinn_catalog_notes.json"):
        path = fixtures_dir() / name
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        for key, val in payload.items():
            if key not in notes:
                notes[key] = val
    return notes


def load_pointer(uri: str) -> OutputRecord:
    notes = _merged_catalog_notes()
    if not notes:
        raise FetchError("pointer catalog missing")
    note = notes.get(uri)
    if not note:
        raise FetchError(f"no public catalog note for pointer {uri!r}")
    kind = str(note.get("kind_hint") or "dataset")
    if uri.startswith("gs://"):
        scheme = "gs_uri"
        output_id = f"{kind}:gs_uri:{uri}"
    else:
        scheme = "url"
        output_id = f"{kind}:url:{uri}"
    idents = [Identifier(scheme, uri, "catalog_pointer")]
    landing = str(note.get("landing_url") or uri)
    if landing and landing != uri:
        idents.append(Identifier("url", landing, "catalog_pointer"))
    doi = str(note.get("doi") or "").strip()
    if doi.startswith("10."):
        idents.append(Identifier("doi", doi, "catalog_pointer"))
    return OutputRecord(
        output_id=output_id,
        kind=kind,
        title=str(note["title"]),
        summary=_summary_with_keywords(note),
        identifiers=idents,
        landing_url=landing,
        source_status="pointer",
        license_hint=note.get("license_hint"),
        evidence_urls=list(note.get("source_urls") or []),
        reuse={
            "public_metadata": True,
            "matrices_downloaded": False,
            "object_store_listed": False,
            "download": "forbidden",
            "apply_yourself": bool(note.get("apply_yourself")),
        },
        topics=structured_topics(note.get("keywords")),
    )


def load_all_catalog_notes() -> list[OutputRecord]:
    """Every catalog_notes (+ extra) pointer, public landing only. Empty if notes missing."""
    notes = _merged_catalog_notes()
    if not notes:
        return []
    out: list[OutputRecord] = []
    for uri in notes:
        try:
            out.append(load_pointer(str(uri)))
        except FetchError:
            continue
    return out

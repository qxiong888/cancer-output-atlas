"""Persist typed nullable node features for the semantic-adapter contract.

Bind to source fields and existing helpers only. Empty > guessed.
Do not mine leftover title tokens. Do not write embeddings.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

from cancer_output_atlas.ods import ODS, ods_category, ods_zh
from cancer_output_atlas.schema import (
    ACCESS_VALUES,
    CALLABILITY_VALUES,
    CATALOG_VALUES,
    OutputRecord,
)

_YEAR = re.compile(r"^(\d{4})(?:[-/T].*)?$")

CATALOG_SCHEME = {
    "geo": "geo",
    "cbioportal": "cbioportal",
    "nct": "nct",
    "gdc_project": "gdc",
    "tcia_collection": "tcia",
    "figshare": "figshare",
    "github": "github",
    "dockstore": "dockstore",
    "nfcore": "nfcore",
    "europepmc": "europepmc",
    "gs_uri": "pointer",
}

_CONTROLLED_HINTS = (
    "controlled",
    "dbgap",
    "apply-yourself",
    "apply yourself",
    "user_applies",
)


def year_from_structured(raw: Any) -> int | None:
    """Parse a year from a source date/year field. Not from titles."""
    if raw is None or raw == "":
        return None
    if isinstance(raw, int):
        return raw if 1900 <= raw <= 2100 else None
    text = str(raw).strip()
    if text.isdigit() and len(text) == 4:
        year = int(text)
        return year if 1900 <= year <= 2100 else None
    m = _YEAR.match(text)
    if not m:
        # GEO pdat is YYYY/MM/DD
        if len(text) >= 4 and text[:4].isdigit() and text[4:5] in {"/", "-", "T"}:
            year = int(text[:4])
            return year if 1900 <= year <= 2100 else None
        return None
    year = int(m.group(1))
    return year if 1900 <= year <= 2100 else None


def catalog_of(rec: OutputRecord) -> str:
    schemes = [i.scheme for i in rec.identifiers]
    for scheme in schemes:
        slug = CATALOG_SCHEME.get(scheme)
        if slug:
            return slug
    if rec.source_status == "pointer":
        return "pointer"
    return "other"


def access_of(rec: OutputRecord) -> str:
    reuse = rec.reuse or {}
    if reuse.get("apply_yourself"):
        return "apply_yourself"
    blob = " ".join(
        p
        for p in (
            rec.license_hint or "",
            str(reuse.get("download") or ""),
            rec.source_status or "",
        )
        if p
    ).lower()
    if rec.source_status == "pointer" and any(h in blob for h in _CONTROLLED_HINTS):
        return "controlled_pointer"
    if rec.source_status == "pointer" and reuse.get("download") == "forbidden":
        if any(h in (rec.license_hint or "").lower() for h in ("controlled", "dbgap", "authorization")):
            return "controlled_pointer"
    return "public_metadata"


def persist_ods(rec: OutputRecord) -> OutputRecord:
    """Store one of the seven NCI ODS types. Empty never."""
    if rec.ods not in ODS:
        rec.ods = ods_category(
            kind=rec.kind,
            title=rec.title,
            summary=rec.summary,
            ids=[i.key() for i in rec.identifiers],
            landing_url=rec.landing_url,
            reuse=rec.reuse,
        )
    rec.ods_zh = ods_zh(rec.ods)
    return rec


def persist_contract(rec: OutputRecord) -> OutputRecord:
    """Fill typed contract fields that are still empty. Never guess cancer type."""
    persist_ods(rec)
    if rec.callability not in CALLABILITY_VALUES:
        from cancer_output_atlas.find import callability_of

        rec.callability = callability_of(rec)
    if rec.access not in ACCESS_VALUES:
        rec.access = access_of(rec)
    if rec.catalog not in CATALOG_VALUES:
        rec.catalog = catalog_of(rec)
    return rec


def apply_reuse_degree(records: list[OutputRecord], links: Iterable[Any]) -> None:
    counts: dict[str, int] = {rec.output_id: 0 for rec in records}
    for ln in links or []:
        src = getattr(ln, "source", None) or (ln.get("source") if isinstance(ln, dict) else None)
        tgt = getattr(ln, "target", None) or (ln.get("target") if isinstance(ln, dict) else None)
        target_is_node = getattr(ln, "target_is_node", True)
        if isinstance(ln, dict):
            target_is_node = bool(ln.get("target_is_node", True))
        if src in counts:
            counts[src] += 1
        if target_is_node and tgt in counts:
            counts[tgt] += 1
    for rec in records:
        rec.reuse_degree = counts.get(rec.output_id, 0)


def structured_topics(*groups: Any) -> list[str]:
    """Copy structured keyword lists only. No title 2-grams."""
    out: list[str] = []
    seen: set[str] = set()
    for group in groups:
        if not group:
            continue
        if isinstance(group, str):
            group = [group]
        for item in group:
            text = str(item).strip()
            if not text:
                continue
            key = text.casefold()
            if key in seen:
                continue
            seen.add(key)
            out.append(text)
    return out


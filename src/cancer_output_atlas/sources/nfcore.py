"""nf-core pipeline index — cancer / onco / tumour / tumor / somatic only."""

from __future__ import annotations

from typing import Any

from cancer_output_atlas.schema import Identifier, OutputRecord
from cancer_output_atlas.sources.http import FetchError, get_json

API = "https://nf-co.re/pipelines.json"
CANCER_TERMS = ("cancer", "onco", "tumour", "tumor", "somatic")


def _is_cancer_pipeline(raw: dict[str, Any]) -> bool:
    blob = " ".join(
        [
            str(raw.get("name") or ""),
            str(raw.get("description") or ""),
            " ".join(str(t) for t in (raw.get("topics") or [])),
        ]
    ).lower()
    return any(t in blob for t in CANCER_TERMS)


def record_from_pipeline(payload: dict[str, Any], source: str) -> OutputRecord:
    name = str(payload.get("name") or "").strip()
    full = str(payload.get("full_name") or (f"nf-core/{name}" if name else "")).strip()
    if not name:
        raise FetchError("nf-core pipeline missing name")
    desc = str(payload.get("description") or "")
    topics = payload.get("topics") or []
    if topics:
        desc = (desc + "\n" if desc else "") + "Topics: " + ", ".join(str(t) for t in topics)
    landing = str(payload.get("homepage") or f"https://nf-co.re/{name}")
    repo = str(payload.get("repository_url") or f"https://github.com/{full}")
    idents = [Identifier("nfcore", name, source), Identifier("github", full, "nfcore.full_name")]
    return OutputRecord(
        output_id=f"workflow:nfcore:{name}",
        kind="workflow",
        title=full or name,
        summary=desc,
        identifiers=idents,
        landing_url=landing,
        source_status="fetched" if source.startswith("nfcore") else "fixture",
        license_hint="nf-core public pipeline index",
        evidence_urls=[landing, repo],
        reuse={"public_metadata": True, "matrices_downloaded": False},
    )


def fetch_cancer_pipelines() -> list[OutputRecord]:
    payload = get_json(API)
    wfs = (payload or {}).get("remote_workflows") or []
    out = []
    for raw in wfs:
        if not _is_cancer_pipeline(raw):
            continue
        try:
            out.append(record_from_pipeline(raw, source="nfcore_api"))
        except FetchError:
            continue
    return out


def records_from_pipelines_fixture(payload: Any, source: str = "fixture") -> list[OutputRecord]:
    pipes = payload.get("pipelines") if isinstance(payload, dict) else payload
    out: list[OutputRecord] = []
    for raw in pipes or []:
        try:
            out.append(record_from_pipeline(raw, source=source))
        except FetchError:
            continue
    return out

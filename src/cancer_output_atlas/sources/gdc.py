"""GDC /projects catalog. Pointer nodes only — never /data, /slicing, BAM."""

from __future__ import annotations

from typing import Any

from cancer_output_atlas.oncotree import types_from_source
from cancer_output_atlas.schema import Identifier, OutputRecord
from cancer_output_atlas.sources.http import FetchError, get_json

API = "https://api.gdc.cancer.gov/projects"


def record_from_project(payload: dict[str, Any], source: str) -> OutputRecord:
    pid = str(payload.get("project_id") or payload.get("id") or "").strip()
    if not pid:
        raise FetchError("GDC project missing project_id")
    sites = payload.get("primary_site") or []
    if isinstance(sites, str):
        sites = [sites]
    diseases = payload.get("disease_type") or []
    if isinstance(diseases, str):
        diseases = [diseases]
    parts = []
    if sites:
        parts.append("primary_site: " + "; ".join(str(s) for s in sites))
    if diseases:
        parts.append("disease_type: " + "; ".join(str(d) for d in diseases))
    parts.append(
        "GDC project metadata only. BAM and /data downloads are forbidden; "
        "controlled files require an apply-yourself GDC/dbGaP process."
    )
    # Never copy dbgap_accession_number onto identifiers (phs###### is refused).
    type_pairs = [(pid, "gdc.project_id"), (payload.get("name"), "gdc.name")]
    for site in sites:
        type_pairs.append((site, "gdc.primary_site"))
    for disease in diseases:
        type_pairs.append((disease, "gdc.disease_type"))
    return OutputRecord(
        output_id=f"dataset:gdc_project:{pid}",
        kind="dataset",
        title=str(payload.get("name") or pid),
        summary="\n".join(parts),
        identifiers=[Identifier("gdc_project", pid, source)],
        landing_url=f"https://portal.gdc.cancer.gov/projects/{pid}",
        source_status="fetched" if source.startswith("gdc") else "fixture",
        license_hint="GDC project catalog; controlled raw/BAM not fetched",
        evidence_urls=[f"{API}/{pid}"],
        reuse={
            "public_metadata": True,
            "matrices_downloaded": False,
            "gdc_data": False,
            "gdc_slicing": False,
            "bam_download": False,
            "download": "forbidden",
        },
        disease_types=types_from_source(type_pairs),
    )


def fetch_projects() -> list[OutputRecord]:
    payload = get_json(f"{API}?size=200&from=0")
    hits = ((payload or {}).get("data") or {}).get("hits") or []
    out = []
    for raw in hits:
        try:
            out.append(record_from_project(raw, source="gdc_api"))
        except FetchError:
            continue
    return out


def records_from_projects_fixture(payload: Any, source: str = "fixture") -> list[OutputRecord]:
    projects = payload.get("projects") if isinstance(payload, dict) else payload
    out: list[OutputRecord] = []
    for raw in projects or []:
        try:
            out.append(record_from_project(raw, source=source))
        except FetchError:
            continue
    return out

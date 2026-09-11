"""Human Cell Atlas Data Portal project metadata (Azul). No matrix download."""

from __future__ import annotations

from typing import Any

from cancer_output_atlas.schema import Identifier, OutputRecord
from cancer_output_atlas.sources.http import FetchError


def record_from_portal(raw: dict[str, Any], source: str) -> OutputRecord:
    url = str(raw.get("landing_url") or "").strip()
    if not url.startswith("https://"):
        raise FetchError("HCA portal missing landing_url")
    return OutputRecord(
        output_id=f"dataset:url:{url}",
        kind="dataset",
        title=str(raw.get("title") or "Human Cell Atlas Data Portal"),
        summary=str(raw.get("summary") or "HCA Data Portal landing; matrices not downloaded."),
        identifiers=[Identifier("url", url, source)],
        landing_url=url,
        source_status="pointer",
        license_hint="HCA DCP public landing; matrices not downloaded",
        evidence_urls=[url],
        reuse={
            "public_metadata": True,
            "matrices_downloaded": False,
            "download": "forbidden",
            "hca": True,
        },
        topics=["human cell atlas", "single-cell"],
        catalog="other",
    )


def record_from_project(raw: dict[str, Any], source: str) -> OutputRecord:
    pid = str(raw.get("project_id") or "").strip()
    if not pid:
        raise FetchError("HCA project missing project_id")
    landing = str(raw.get("landing_url") or f"https://data.humancellatlas.org/explore/projects/{pid}")
    parts = [str(raw.get("summary") or "")]
    organs = raw.get("organs") or []
    diseases = raw.get("diseases") or []
    if organs:
        parts.append("organs: " + "; ".join(str(o) for o in organs))
    if diseases:
        parts.append("diseases: " + "; ".join(str(d) for d in diseases))
    reasons = raw.get("match_reasons") or []
    if reasons:
        parts.append("match_reasons: " + ", ".join(str(r) for r in reasons))
    dois = [str(d) for d in (raw.get("dois") or []) if str(d).startswith("10.")]
    if dois:
        parts.append("publication_doi_observed: " + "; ".join(dois))
    parts.append(
        "HCA project metadata only (Azul catalog). Matrices and controlled files are not downloaded."
    )
    apply_yourself = bool(raw.get("apply_yourself"))
    n_samples = raw.get("estimated_cell_count")
    try:
        n_samples = int(n_samples) if n_samples is not None else None
    except (TypeError, ValueError):
        n_samples = None
    return OutputRecord(
        output_id=f"dataset:hca_project:{pid}",
        kind="dataset",
        title=str(raw.get("title") or pid),
        summary="\n".join(p for p in parts if p)[:4000],
        identifiers=[
            Identifier("hca_project", pid, source),
            Identifier("url", landing, source),
        ],
        landing_url=landing,
        source_status="fixture" if source == "fixture" else "fetched",
        license_hint="HCA DCP project catalog; matrices not downloaded",
        n_samples=n_samples,
        evidence_urls=[landing],
        reuse={
            "public_metadata": True,
            "matrices_downloaded": False,
            "download": "forbidden",
            "hca": True,
            "apply_yourself": apply_yourself,
        },
        topics=[str(r) for r in reasons] or ["human cell atlas"],
        catalog="other",
    )


def records_from_fixture(payload: Any, source: str = "fixture") -> list[OutputRecord]:
    if not isinstance(payload, dict):
        return []
    out: list[OutputRecord] = []
    portal = payload.get("portal")
    if isinstance(portal, dict):
        try:
            out.append(record_from_portal(portal, source))
        except FetchError:
            pass
    for raw in payload.get("projects") or []:
        try:
            out.append(record_from_project(raw, source))
        except FetchError:
            continue
    return out


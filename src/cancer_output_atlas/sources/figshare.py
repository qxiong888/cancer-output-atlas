"""Figshare public article metadata. Lists file names; never downloads payloads."""

from __future__ import annotations

import re
from typing import Any

from cancer_output_atlas.safety import is_omics_filename
from cancer_output_atlas.node_features import structured_topics, year_from_structured
from cancer_output_atlas.schema import Identifier, OutputRecord
from cancer_output_atlas.sources.http import FetchError, get_json

FIGSHARE_API = "https://api.figshare.com/v2/articles"
DOI_ARTICLE_RE = re.compile(r"figshare(?:\.plus)?\.(\d+)")


def article_id_from_doi(doi: str) -> str:
    m = DOI_ARTICLE_RE.search(doi)
    if not m:
        raise FetchError(f"cannot parse figshare article id from {doi!r}")
    return m.group(1)


def fetch_figshare(doi: str) -> OutputRecord:
    article_id = article_id_from_doi(doi)
    payload = get_json(f"{FIGSHARE_API}/{article_id}")
    return record_from_figshare(doi, payload, source="figshare_api")


def record_from_figshare(doi: str, payload: dict[str, Any], source: str) -> OutputRecord:
    identifiers = [
        Identifier("doi", doi, source),
        Identifier("figshare", str(payload.get("id") or article_id_from_doi(doi)), source),
    ]
    api_doi = str(payload.get("doi") or "").strip()
    if api_doi and api_doi != doi:
        identifiers.append(Identifier("doi", api_doi, "figshare.doi"))
    resource_doi = str(payload.get("resource_doi") or "").strip()
    if resource_doi.startswith("10."):
        identifiers.append(Identifier("doi", resource_doi, "figshare.resource_doi"))
    for ref in payload.get("references") or []:
        ref_s = str(ref)
        if "10.1016/" in ref_s or ref_s.startswith("10."):
            val = ref_s.replace("https://doi.org/", "").strip()
            if val.startswith("10."):
                identifiers.append(Identifier("doi", val, "figshare.references"))
    landing = (
        payload.get("figshare_url")
        or payload.get("url_public_html")
        or f"https://doi.org/{doi}"
    )
    files = payload.get("files") or []
    omics_names = [f.get("name") for f in files if is_omics_filename(str(f.get("name") or ""))]
    license_obj = payload.get("license") or {}
    license_hint = license_obj.get("name") if isinstance(license_obj, dict) else None
    keywords = payload.get("keywords") or payload.get("tags") or []
    summary = str(payload.get("description") or "")
    if keywords:
        summary = summary + "\nKeywords: " + ", ".join(str(k) for k in keywords)
    rec = OutputRecord(
        output_id=f"dataset:doi:{doi}",
        kind="dataset",
        title=str(payload.get("title") or doi),
        summary=summary,
        identifiers=_dedupe(identifiers),
        landing_url=str(landing),
        source_status="fetched" if source == "figshare_api" else "fixture",
        license_hint=license_hint,
        evidence_urls=[str(landing)],
        reuse={
            "public_metadata": True,
            "matrices_downloaded": False,
            "listed_omics_filenames": omics_names,
            "note": "h5ad/omics filenames listed from metadata; never downloaded",
        },
        topics=structured_topics(keywords),
        year=year_from_structured(
            payload.get("published_date") or payload.get("created_date") or payload.get("timeline")
        ),
    )
    return rec


def _dedupe(idents: list[Identifier]) -> list[Identifier]:
    seen: set[str] = set()
    out: list[Identifier] = []
    for i in idents:
        if i.key() in seen:
            continue
        seen.add(i.key())
        out.append(i)
    return out


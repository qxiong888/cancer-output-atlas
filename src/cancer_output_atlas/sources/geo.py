"""NCBI GEO / GDS public metadata via E-utilities. Metadata only."""

from __future__ import annotations

from typing import Any

from cancer_output_atlas.node_features import year_from_structured
from cancer_output_atlas.oncotree import geo_structured_types
from cancer_output_atlas.schema import Identifier, OutputRecord
from cancer_output_atlas.sources.http import FetchError, get_json

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def _series_uid(payload: dict[str, Any]) -> str | None:
    ids = payload.get("esearchresult", {}).get("idlist") or []
    for uid in ids:
        if str(uid).startswith("200"):
            return str(uid)
    return str(ids[0]) if ids else None


def fetch_gse(accession: str) -> OutputRecord:
    acc = accession.strip().upper()
    if not acc.startswith("GSE") or not acc[3:].isdigit():
        raise FetchError(f"not a GEO series accession: {accession!r}")
    search = get_json(f"{EUTILS}/esearch.fcgi?db=gds&term={acc}[ACCN]&retmode=json")
    uid = _series_uid(search)
    if not uid:
        raise FetchError(f"GEO esearch empty for {acc}")
    summary = get_json(f"{EUTILS}/esummary.fcgi?db=gds&id={uid}&retmode=json")
    rec = (summary.get("result") or {}).get(uid) or {}
    if rec.get("error") or rec.get("accession") != acc:
        # Accept if accession missing but uid matched a document without error.
        if rec.get("error") or not rec:
            raise FetchError(f"GEO esummary failed for {acc} uid={uid}")
    return record_from_esummary(acc, rec, source="ncbi_eutils")


def record_from_esummary(acc: str, rec: dict[str, Any], source: str) -> OutputRecord:
    identifiers = [Identifier("geo", acc, source)]
    pmid_list = rec.get("pubmedids") or []
    if isinstance(pmid_list, str):
        pmid_list = [pmid_list]
    for pmid in pmid_list:
        pmid = str(pmid).strip()
        if pmid.isdigit():
            identifiers.append(Identifier("pmid", pmid, "geo_esummary.pubmedids"))
    for rel in rec.get("extrelations") or []:
        if (rel.get("relationtype") or "").upper() == "SRA":
            sra = str(rel.get("targetobject") or "").strip()
            if sra.startswith("SRP") and sra[3:].isdigit():
                identifiers.append(Identifier("sra", sra, "geo_esummary.extrelations"))
    bp = str(rec.get("bioproject") or "").strip()
    if bp.startswith("PRJNA") and bp[5:].isdigit():
        identifiers.append(Identifier("bioproject", bp, "geo_esummary.bioproject"))
    n_samples = rec.get("n_samples")
    try:
        n_samples_i = int(n_samples) if n_samples not in (None, "") else None
    except (TypeError, ValueError):
        n_samples_i = None
    return OutputRecord(
        output_id=f"dataset:geo:{acc}",
        kind="dataset",
        title=str(rec.get("title") or acc),
        summary=str(rec.get("summary") or ""),
        identifiers=identifiers,
        landing_url=f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={acc}",
        source_status="fetched" if source == "ncbi_eutils" else "fixture",
        license_hint="GEO public series metadata; check series record for data-use terms",
        organism=str(rec.get("taxon") or "") or None,
        assay=str(rec.get("gdstype") or "") or None,
        n_samples=n_samples_i,
        evidence_urls=[
            f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={acc}",
        ],
        disease_types=geo_structured_types(rec),
        year=year_from_structured(rec.get("pdat") or rec.get("pdyear")),
    )

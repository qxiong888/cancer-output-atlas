"""Europe PMC article metadata (optional paper-with-data node)."""

from __future__ import annotations

from typing import Any
from urllib.parse import quote

from cancer_output_atlas.node_features import year_from_structured
from cancer_output_atlas.schema import Identifier, OutputRecord
from cancer_output_atlas.sources.http import FetchError, get_json

API = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"


def record_from_result(payload: dict[str, Any], source: str) -> OutputRecord:
    eid = str(payload.get("id") or "").strip()
    doi = str(payload.get("doi") or "").strip()
    if not eid and not doi:
        raise FetchError("Europe PMC result missing id and doi")
    idents: list[Identifier] = []
    if eid:
        idents.append(Identifier("europepmc", eid, source))
    if doi.startswith("10."):
        idents.append(Identifier("doi", doi, source))
    pmid = str(payload.get("pmid") or "").strip()
    if pmid.isdigit():
        idents.append(Identifier("pmid", pmid, "europepmc.pmid"))
    landing = f"https://doi.org/{doi}" if doi.startswith("10.") else (
        f"https://europepmc.org/article/{payload.get('source') or 'PPR'}/{eid}"
    )
    abstract = str(payload.get("abstractText") or "")
    return OutputRecord(
        output_id=f"publication:doi:{doi}" if doi.startswith("10.") else f"publication:europepmc:{eid}",
        kind="publication",
        title=str(payload.get("title") or doi or eid),
        summary=abstract,
        identifiers=idents,
        landing_url=landing,
        source_status="fetched" if source.startswith("europepmc") else "fixture",
        license_hint="Europe PMC public article metadata; closed full text not downloaded",
        evidence_urls=[landing],
        reuse={"public_metadata": True, "matrices_downloaded": False, "full_text": False},
        year=year_from_structured(
            payload.get("pubYear") or payload.get("firstPublicationDate")
        ),
    )


def fetch_doi(doi: str) -> OutputRecord:
    q = f"DOI:{doi.strip()}"
    payload = get_json(f"{API}?query={quote(q)}&format=json&resultType=core")
    results = ((payload or {}).get("resultList") or {}).get("result") or []
    if not results:
        raise FetchError(f"Europe PMC empty for {doi}")
    return record_from_result(results[0], source="europepmc_api")


def records_from_fixture(payload: Any, source: str = "fixture") -> list[OutputRecord]:
    results = payload.get("results") if isinstance(payload, dict) else payload
    out: list[OutputRecord] = []
    for raw in results or []:
        try:
            out.append(record_from_result(raw, source=source))
        except FetchError:
            continue
    return out

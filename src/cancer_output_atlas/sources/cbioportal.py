"""cBioPortal public study metadata. Never fetches mutation/CNA matrices."""

from __future__ import annotations

from typing import Any

from cancer_output_atlas.oncotree import types_from_source
from cancer_output_atlas.schema import Identifier, OutputRecord
from cancer_output_atlas.sources.http import FetchError, get_json

API = "https://www.cbioportal.org/api/studies"

# Endpoints this adapter will never request (matrix / patient-level).
REFUSED_MATRIX_PATHS = (
    "/api/molecular-profiles",
    "/mutations",
    "/discrete-copy-number",
    "/copy-number-segments",
    "/clinical-data",
)


def record_from_study(payload: dict[str, Any], source: str) -> OutputRecord:
    study_id = str(payload.get("studyId") or "").strip()
    if not study_id:
        raise FetchError("cBioPortal study missing studyId")
    identifiers = [Identifier("cbioportal", study_id, source)]
    pmid_raw = payload.get("pmid")
    if pmid_raw:
        for part in str(pmid_raw).split(","):
            pmid = part.strip()
            if pmid.isdigit():
                identifiers.append(Identifier("pmid", pmid, "cbioportal.pmid"))
    desc = str(payload.get("description") or "")
    extras = []
    if payload.get("cancerTypeId"):
        extras.append(f"cancerTypeId: {payload['cancerTypeId']}")
    if payload.get("citation"):
        extras.append(f"citation: {payload['citation']}")
    if payload.get("publicStudy") is True:
        extras.append("publicStudy: true")
    n_samples = payload.get("allSampleCount")
    try:
        n_samples_i = int(n_samples) if n_samples not in (None, "") else None
    except (TypeError, ValueError):
        n_samples_i = None
    if extras:
        desc = (desc + "\n" if desc else "") + " | ".join(extras)
    return OutputRecord(
        output_id=f"dataset:cbioportal:{study_id}",
        kind="dataset",
        title=str(payload.get("name") or study_id),
        summary=desc,
        identifiers=identifiers,
        landing_url=f"https://www.cbioportal.org/study/summary?id={study_id}",
        source_status="fetched" if source.startswith("cbioportal_api") else "fixture",
        license_hint="cBioPortal public study metadata; raw sequencing may be controlled elsewhere",
        n_samples=n_samples_i,
        evidence_urls=[
            f"https://www.cbioportal.org/study/summary?id={study_id}",
            f"{API}/{study_id}",
        ],
        reuse={
            "public_metadata": True,
            "matrices_downloaded": False,
            "mutation_matrix": False,
            "cna_matrix": False,
        },
        disease_types=types_from_source(
            [(payload.get("cancerTypeId"), "cbioportal.cancerTypeId")]
        ),
    )


def fetch_study(study_id: str) -> OutputRecord:
    sid = study_id.strip()
    if not sid or "/" in sid or " " in sid:
        raise FetchError(f"not a cBioPortal study id: {study_id!r}")
    payload = get_json(f"{API}/{sid}")
    if not isinstance(payload, dict):
        raise FetchError(f"cBioPortal non-object for {sid}")
    return record_from_study(payload, source="cbioportal_api")


def records_from_studies_fixture(payload: Any, source: str = "fixture") -> list[OutputRecord]:
    studies = payload.get("studies") if isinstance(payload, dict) else payload
    out: list[OutputRecord] = []
    for raw in studies or []:
        try:
            out.append(record_from_study(raw, source=source))
        except FetchError:
            continue
    return out


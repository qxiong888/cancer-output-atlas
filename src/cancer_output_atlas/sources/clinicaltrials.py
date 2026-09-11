"""ClinicalTrials.gov API v2 — protocol metadata only."""

from __future__ import annotations

import re
from typing import Any

from cancer_output_atlas.node_features import year_from_structured
from cancer_output_atlas.oncotree import types_from_source
from cancer_output_atlas.schema import Identifier, OutputRecord
from cancer_output_atlas.sources.http import FetchError, get_json

NCT_RE = re.compile(r"^NCT\d{8}$")
API = "https://clinicaltrials.gov/api/v2/studies"


def _proto(payload: dict[str, Any]) -> dict[str, Any]:
    return payload.get("protocolSection") or payload


def record_from_study(payload: dict[str, Any], source: str) -> OutputRecord:
    proto = _proto(payload)
    ident = proto.get("identificationModule") or {}
    nct = str(ident.get("nctId") or "").strip().upper()
    if not NCT_RE.match(nct):
        raise FetchError(f"ClinicalTrials payload missing NCT id")
    cond = proto.get("conditionsModule") or {}
    arms = proto.get("armsInterventionsModule") or {}
    status = proto.get("statusModule") or {}
    design = proto.get("designModule") or {}
    conditions = [str(c) for c in (cond.get("conditions") or []) if c]
    interventions = []
    for item in arms.get("interventions") or []:
        name = str(item.get("name") or "").strip()
        if name:
            interventions.append(name)
    title = str(ident.get("briefTitle") or nct)
    official = str(ident.get("officialTitle") or "")
    parts = []
    if official and official != title:
        parts.append(official)
    if conditions:
        parts.append("Conditions: " + "; ".join(conditions))
    if interventions:
        parts.append("Interventions: " + "; ".join(interventions))
    overall = status.get("overallStatus")
    if overall:
        parts.append(f"Status: {overall}")
    phases = design.get("phases") or []
    if phases:
        parts.append("Phases: " + ", ".join(str(p) for p in phases))
    year = None
    for key in ("startDateStruct", "lastUpdatePostDateStruct", "statusVerifiedDate"):
        raw = status.get(key)
        if isinstance(raw, dict):
            year = year_from_structured(raw.get("date"))
        else:
            year = year_from_structured(raw)
        if year is not None:
            break
    reuse = {
        "public_metadata": True,
        "matrices_downloaded": False,
        "attachments_downloaded": False,
    }
    flag = payload.get("hasResults")
    if flag is None:
        flag = payload.get("has_results")
    if isinstance(flag, bool):
        reuse["has_results"] = flag
    return OutputRecord(
        output_id=f"trial:nct:{nct}",
        kind="trial",
        title=title,
        summary="\n".join(parts),
        identifiers=[Identifier("nct", nct, source)],
        landing_url=f"https://clinicaltrials.gov/study/{nct}",
        source_status="fetched" if source.startswith("clinicaltrials") else "fixture",
        license_hint="ClinicalTrials.gov public protocol metadata",
        evidence_urls=[f"https://clinicaltrials.gov/study/{nct}", f"{API}/{nct}"],
        reuse=reuse,
        disease_types=types_from_source((c, "nct.condition") for c in conditions),
        interventions=list(interventions),
        year=year,
    )


def fetch_nct(nct: str) -> OutputRecord:
    acc = nct.strip().upper()
    if not NCT_RE.match(acc):
        raise FetchError(f"not an NCT identifier: {nct!r}")
    payload = get_json(f"{API}/{acc}")
    if not isinstance(payload, dict):
        raise FetchError(f"ClinicalTrials non-object for {acc}")
    return record_from_study(payload, source="clinicaltrials_api")


def records_from_query_fixture(payload: dict[str, Any], source: str) -> list[OutputRecord]:
    out: list[OutputRecord] = []
    for study in payload.get("studies") or []:
        try:
            out.append(record_from_study(study, source=source))
        except FetchError:
            continue
    return out


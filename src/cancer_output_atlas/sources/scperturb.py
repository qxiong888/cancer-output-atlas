"""scPerturb portal / GitHub / Zenodo public landings. Never downloads h5ad."""

from __future__ import annotations

from typing import Any

from cancer_output_atlas.schema import Identifier, OutputRecord
from cancer_output_atlas.sources.http import FetchError


def record_from_portal(raw: dict[str, Any], source: str) -> OutputRecord:
    url = str(raw.get("landing_url") or "").strip()
    if not url.startswith("http"):
        raise FetchError("scPerturb portal missing landing_url")
    return OutputRecord(
        output_id=f"dataset:url:{url}",
        kind="dataset",
        title=str(raw.get("title") or url),
        summary=str(raw.get("summary") or "scPerturb public landing; h5ad not downloaded."),
        identifiers=[Identifier("url", url, source)],
        landing_url=url,
        source_status="pointer",
        license_hint="scPerturb portal; h5ad not downloaded",
        evidence_urls=[url],
        reuse={
            "public_metadata": True,
            "matrices_downloaded": False,
            "download": "forbidden",
            "scperturb": True,
        },
        topics=["scperturb", "perturbation", "single-cell"],
        catalog="other",
    )


def record_from_zenodo(raw: dict[str, Any], source: str) -> OutputRecord:
    rid = str(raw.get("record_id") or "").strip()
    landing = str(raw.get("landing_url") or "").strip()
    if not rid or not landing.startswith("https://"):
        raise FetchError("Zenodo record missing id or landing")
    doi = raw.get("doi")
    idents = [
        Identifier("zenodo", rid, source),
        Identifier("url", landing, source),
    ]
    extra = []
    if doi and str(doi).startswith("10."):
        extra.append(f"doi_observed: {doi}")
    if raw.get("version"):
        extra.append(f"version: {raw['version']}")
    desc = str(raw.get("description") or raw.get("note") or "")
    summary = desc
    if extra:
        summary = (summary + "\n" if summary else "") + "\n".join(extra)
    summary = (summary + "\n" if summary else "") + (
        "Zenodo landing metadata only. h5ad / ATAC payloads are not downloaded."
    )
    return OutputRecord(
        output_id=f"dataset:zenodo:{rid}",
        kind="dataset",
        title=str(raw.get("title") or f"Zenodo {rid}"),
        summary=summary[:3000],
        identifiers=idents,
        landing_url=landing,
        source_status="fixture" if source == "fixture" else "fetched",
        license_hint="Zenodo public landing; files not downloaded",
        evidence_urls=[landing],
        reuse={
            "public_metadata": True,
            "matrices_downloaded": False,
            "download": "forbidden",
            "scperturb": True,
        },
        topics=["scperturb", "zenodo"],
        catalog="other",
    )


def record_from_github(raw: dict[str, Any], source: str) -> OutputRecord:
    full = str(raw.get("full_name") or "").strip()
    landing = str(raw.get("html_url") or "").strip()
    if not full or "/" not in full:
        raise FetchError("scPerturb GitHub missing full_name")
    if not landing:
        landing = f"https://github.com/{full}"
    desc = str(raw.get("description") or "")
    return OutputRecord(
        output_id=f"software:github:{full}",
        kind="software",
        title=full,
        summary=desc + "\nscPerturb source repository. h5ad payloads are not cloned or downloaded.",
        identifiers=[
            Identifier("github", full, source),
            Identifier("url", landing, source),
        ],
        landing_url=landing,
        source_status="fixture" if source == "fixture" else "fetched",
        license_hint=str(raw.get("license") or "See repository LICENSE"),
        evidence_urls=[landing],
        reuse={
            "public_metadata": True,
            "matrices_downloaded": False,
            "clone": False,
            "download": "forbidden",
            "scperturb": True,
        },
        topics=["scperturb", "github"],
        catalog="other",
    )


def records_from_fixture(payload: Any, source: str = "fixture") -> list[OutputRecord]:
    if not isinstance(payload, dict):
        return []
    out: list[OutputRecord] = []
    for raw in payload.get("portals") or []:
        try:
            out.append(record_from_portal(raw, source))
        except FetchError:
            continue
    gh = payload.get("github")
    if isinstance(gh, dict):
        try:
            out.append(record_from_github(gh, source))
        except FetchError:
            pass
    for raw in payload.get("zenodo") or []:
        try:
            out.append(record_from_zenodo(raw, source))
        except FetchError:
            continue
    return out

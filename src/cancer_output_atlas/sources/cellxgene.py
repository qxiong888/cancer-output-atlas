"""CZ CELLxGENE Discover / Census public collection + dataset metadata.

Landing pages and IDs only. Never follows datasets.cellxgene *.h5ad assets.
"""

from __future__ import annotations

from typing import Any

from cancer_output_atlas.schema import Identifier, OutputRecord
from cancer_output_atlas.sources.http import FetchError


def _labels(items: Any) -> list[str]:
    out: list[str] = []
    for it in items or []:
        if isinstance(it, dict) and it.get("label"):
            out.append(str(it["label"]))
        elif isinstance(it, str):
            out.append(it)
    return out


def record_from_portal(raw: dict[str, Any], source: str) -> OutputRecord:
    url = str(raw.get("landing_url") or "").strip()
    if not url.startswith("https://"):
        raise FetchError("CELLxGENE portal missing landing_url")
    pid = str(raw.get("id") or url)
    return OutputRecord(
        output_id=f"dataset:url:{url}",
        kind=str(raw.get("kind") or "dataset"),
        title=str(raw.get("title") or pid),
        summary=str(raw.get("summary") or "CZ CELLxGENE public landing; matrices not downloaded."),
        identifiers=[Identifier("url", url, source)],
        landing_url=url,
        source_status="fixture" if source == "fixture" else "pointer",
        license_hint="CELLxGENE Discover / Census landing; h5ad not downloaded",
        evidence_urls=[url],
        reuse={
            "public_metadata": True,
            "matrices_downloaded": False,
            "download": "forbidden",
            "cellxgene": True,
        },
        topics=["cellxgene", "single-cell"],
        catalog="other",
    )


def record_from_collection(raw: dict[str, Any], source: str) -> OutputRecord:
    cid = str(raw.get("collection_id") or "").strip()
    if not cid:
        raise FetchError("CELLxGENE collection missing collection_id")
    landing = str(raw.get("collection_url") or f"https://cellxgene.cziscience.com/collections/{cid}")
    desc = str(raw.get("description") or "")
    reasons = [str(r) for r in (raw.get("match_reasons") or [])]
    n_ds = raw.get("n_datasets")
    extra = []
    if reasons:
        extra.append("match_reasons: " + ", ".join(reasons))
    if n_ds is not None:
        extra.append(f"n_datasets: {n_ds}")
    consortia = [str(c) for c in (raw.get("consortia") or [])]
    if consortia:
        extra.append("consortia: " + ", ".join(consortia))
    doi = raw.get("doi")
    if doi:
        extra.append(f"collection_doi_observed: {doi}")
    summary = desc
    if extra:
        summary = (summary + "\n" if summary else "") + "\n".join(extra)
    summary = (summary + "\n" if summary else "") + (
        "CELLxGENE collection metadata only. h5ad / Census matrices are not downloaded."
    )
    idents = [
        Identifier("cellxgene_collection", cid, source),
        Identifier("url", landing, source),
    ]
    return OutputRecord(
        output_id=f"dataset:cellxgene_collection:{cid}",
        kind="dataset",
        title=str(raw.get("name") or cid),
        summary=summary[:4000],
        identifiers=idents,
        landing_url=landing,
        source_status="fixture" if source == "fixture" else "fetched",
        license_hint="CELLxGENE Discover collection; matrices not downloaded",
        evidence_urls=[landing],
        reuse={
            "public_metadata": True,
            "matrices_downloaded": False,
            "download": "forbidden",
            "cellxgene": True,
        },
        topics=reasons or ["cellxgene"],
        catalog="other",
    )


def record_from_dataset(raw: dict[str, Any], source: str, collection_url: str | None = None) -> OutputRecord:
    did = str(raw.get("dataset_id") or "").strip()
    if not did:
        raise FetchError("CELLxGENE dataset missing dataset_id")
    landing = str(raw.get("explorer_url") or collection_url or "").strip()
    if not landing.startswith("https://"):
        raise FetchError(f"CELLxGENE dataset {did} missing landing")
    if landing.lower().endswith((".h5ad", ".h5", ".bam")):
        raise FetchError(f"refused omics landing for dataset {did}")
    tissue = raw.get("tissue") if isinstance(raw.get("tissue"), list) else _labels(raw.get("tissue"))
    disease = raw.get("disease") if isinstance(raw.get("disease"), list) else _labels(raw.get("disease"))
    organism = raw.get("organism") if isinstance(raw.get("organism"), list) else _labels(raw.get("organism"))
    assay = raw.get("assay") if isinstance(raw.get("assay"), list) else _labels(raw.get("assay"))
    parts = []
    if raw.get("collection_name"):
        parts.append(f"collection: {raw['collection_name']}")
    if tissue:
        parts.append("tissue: " + "; ".join(str(t) for t in tissue))
    if disease:
        parts.append("disease: " + "; ".join(str(d) for d in disease))
    if organism:
        parts.append("organism: " + "; ".join(str(o) for o in organism))
    if assay:
        parts.append("assay: " + "; ".join(str(a) for a in assay))
    if raw.get("cell_count") is not None:
        parts.append(f"cell_count: {raw['cell_count']}")
    if raw.get("landing_fallback"):
        parts.append(f"landing_fallback: {raw['landing_fallback']}")
    parts.append("CELLxGENE dataset metadata only. h5ad assets are not downloaded.")
    n_samples = raw.get("cell_count")
    try:
        n_samples = int(n_samples) if n_samples is not None else None
    except (TypeError, ValueError):
        n_samples = None
    return OutputRecord(
        output_id=f"dataset:cellxgene_dataset:{did}",
        kind="dataset",
        title=str(raw.get("title") or did),
        summary="\n".join(parts),
        identifiers=[
            Identifier("cellxgene_dataset", did, source),
            Identifier("url", landing, source),
        ],
        landing_url=landing,
        source_status="fixture" if source == "fixture" else "fetched",
        license_hint="CELLxGENE Discover dataset; matrices not downloaded",
        organism="; ".join(str(o) for o in organism) if organism else None,
        assay="; ".join(str(a) for a in assay) if assay else None,
        n_samples=n_samples,
        evidence_urls=[landing],
        reuse={
            "public_metadata": True,
            "matrices_downloaded": False,
            "download": "forbidden",
            "cellxgene": True,
        },
        topics=["cellxgene", "single-cell"],
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
    for col in payload.get("collections") or []:
        try:
            out.append(record_from_collection(col, source))
        except FetchError:
            continue
        curl = str(col.get("collection_url") or "")
        for ds in col.get("datasets") or []:
            try:
                out.append(record_from_dataset(ds, source, collection_url=curl))
            except FetchError:
                continue
    return out

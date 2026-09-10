"""TCIA collection *names* as pointer nodes. Never download DICOM."""

from __future__ import annotations

from typing import Any

from cancer_output_atlas.schema import Identifier, OutputRecord
from cancer_output_atlas.sources.http import FetchError, get_json

API = "https://services.cancerimagingarchive.net/nbia-api/services/v1/getCollectionValues"


def record_from_collection(name: str, source: str) -> OutputRecord:
    coll = name.strip()
    if not coll:
        raise FetchError("empty TCIA collection name")
    return OutputRecord(
        output_id=f"dataset:tcia_collection:{coll}",
        kind="dataset",
        title=f"TCIA collection {coll}",
        summary=(
            f"Public TCIA collection name {coll} from getCollectionValues. "
            "Pointer only — DICOM / pixel data are never downloaded."
        ),
        identifiers=[Identifier("tcia_collection", coll, source)],
        landing_url="https://www.cancerimagingarchive.net/",
        source_status="pointer" if source.endswith("pointer") else (
            "fetched" if source.startswith("tcia_api") else "fixture"
        ),
        license_hint="TCIA collection name only; images not fetched",
        evidence_urls=[API],
        reuse={
            "public_metadata": True,
            "matrices_downloaded": False,
            "dicom_download": False,
            "download": "forbidden",
        },
    )


def fetch_collections() -> list[OutputRecord]:
    payload = get_json(API)
    names: list[str] = []
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict) and item.get("Collection"):
                names.append(str(item["Collection"]))
            elif isinstance(item, str):
                names.append(item)
    return [record_from_collection(n, source="tcia_api") for n in names]


def records_from_collections_fixture(payload: Any, source: str = "fixture") -> list[OutputRecord]:
    names = payload.get("collections") if isinstance(payload, dict) else payload
    out: list[OutputRecord] = []
    for name in names or []:
        if isinstance(name, dict):
            name = name.get("Collection") or name.get("name")
        try:
            out.append(record_from_collection(str(name), source=source))
        except FetchError:
            continue
    return out

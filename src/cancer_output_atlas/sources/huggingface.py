"""Hugging Face public dataset/model *cards* only. Never lists parquet/h5ad/weights."""

from __future__ import annotations

from typing import Any

from cancer_output_atlas.schema import Identifier, OutputRecord
from cancer_output_atlas.sources.http import FetchError


def record_from_card(raw: dict[str, Any], source: str) -> OutputRecord:
    repo_id = str(raw.get("id") or raw.get("repo_id") or "").strip()
    url = str(raw.get("landing_url") or "").strip()
    if not repo_id or not url.startswith("https://huggingface.co/"):
        raise FetchError("Hugging Face card missing id or landing_url")
    if raw.get("private") or raw.get("disabled"):
        raise FetchError(f"Hugging Face card not public: {repo_id}")
    repo_type = str(raw.get("repo_type") or "dataset")
    kind = "model" if repo_type == "model" else "dataset"
    desc = str(raw.get("description") or "").strip()
    card_text = str(raw.get("card_text") or "").strip()
    tags = [str(t) for t in (raw.get("tags") or [])]
    summary_parts = []
    if desc:
        summary_parts.append(desc)
    if card_text and card_text not in desc:
        summary_parts.append(card_text[:2000])
    if tags:
        summary_parts.append("Keywords: " + ", ".join(tags[:30]))
    summary_parts.append(
        "Hugging Face public card text only. Weights, parquet, and h5ad files are not downloaded."
    )
    license_hint = raw.get("license")
    return OutputRecord(
        output_id=f"{kind}:huggingface:{repo_id}",
        kind=kind,
        title=str(raw.get("title") or repo_id),
        summary="\n\n".join(summary_parts)[:5000],
        identifiers=[
            Identifier("huggingface", repo_id, source),
            Identifier("url", url, source),
        ],
        landing_url=url,
        source_status="fixture" if source == "fixture" else "fetched",
        license_hint=str(license_hint) if license_hint else "Hugging Face public card; payloads not downloaded",
        evidence_urls=[url],
        reuse={
            "public_metadata": True,
            "matrices_downloaded": False,
            "download": "forbidden",
            "huggingface_card": True,
        },
        topics=tags[:20],
        catalog="other",
    )


def records_from_fixture(payload: Any, source: str = "fixture") -> list[OutputRecord]:
    cards = payload.get("cards") if isinstance(payload, dict) else payload
    out: list[OutputRecord] = []
    for raw in cards or []:
        if not isinstance(raw, dict):
            continue
        try:
            out.append(record_from_card(raw, source))
        except FetchError:
            continue
    return out

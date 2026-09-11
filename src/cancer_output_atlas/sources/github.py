"""GitHub public repository metadata. Prefer search; /repos may 403."""

from __future__ import annotations

from typing import Any
from urllib.parse import quote

from cancer_output_atlas.node_features import structured_topics, year_from_structured
from cancer_output_atlas.schema import Identifier, OutputRecord
from cancer_output_atlas.sources.http import FetchError, get_json

SEARCH = "https://api.github.com/search/repositories"


def record_from_repo(payload: dict[str, Any], source: str) -> OutputRecord:
    full = str(payload.get("full_name") or "").strip()
    if not full or "/" not in full:
        raise FetchError("GitHub repo missing full_name")
    desc = str(payload.get("description") or "")
    topics = payload.get("topics") or []
    if topics:
        desc = (desc + "\n" if desc else "") + "Topics: " + ", ".join(str(t) for t in topics)
    license_hint = payload.get("license")
    if isinstance(license_hint, dict):
        license_hint = license_hint.get("spdx_id") or license_hint.get("name")
    landing = str(payload.get("html_url") or f"https://github.com/{full}")
    stars = payload.get("stargazers_count")
    if stars is not None:
        desc = (desc + "\n" if desc else "") + f"stargazers: {stars}"
    return OutputRecord(
        output_id=f"software:github:{full}",
        kind="software",
        title=full,
        summary=desc,
        identifiers=[
            Identifier("github", full, source),
            Identifier("url", landing, source),
        ],
        landing_url=landing,
        source_status="fetched" if source.startswith("github") else "fixture",
        license_hint=str(license_hint) if license_hint else None,
        evidence_urls=[landing],
        reuse={"public_metadata": True, "matrices_downloaded": False, "clone": False},
        topics=structured_topics(topics),
        year=year_from_structured(
            payload.get("created_at") or payload.get("pushed_at") or payload.get("updated_at")
        ),
    )


def fetch_repo(full_name: str) -> OutputRecord:
    name = full_name.strip()
    if name.count("/") != 1:
        raise FetchError(f"not owner/repo: {full_name!r}")
    # Search is more reliable from this box than GET /repos (often 403).
    payload = get_json(f"{SEARCH}?q={quote('repo:' + name)}")
    items = (payload or {}).get("items") if isinstance(payload, dict) else None
    if items:
        return record_from_repo(items[0], source="github_search")
    raise FetchError(f"GitHub search empty for {name}")


def records_from_repos_fixture(payload: Any, source: str = "fixture") -> list[OutputRecord]:
    items = payload.get("items") if isinstance(payload, dict) else payload
    if isinstance(payload, dict) and "full_name" in payload:
        items = [payload]
    out: list[OutputRecord] = []
    for raw in items or []:
        try:
            out.append(record_from_repo(raw, source=source))
        except FetchError:
            continue
    return out


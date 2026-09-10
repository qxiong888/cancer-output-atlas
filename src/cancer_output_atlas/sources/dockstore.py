"""Dockstore GA4GH TRS — workflow descriptors, not run data."""

from __future__ import annotations

from typing import Any
from urllib.parse import quote

from cancer_output_atlas.schema import Identifier, OutputRecord
from cancer_output_atlas.sources.http import FetchError, get_json

TRS = "https://dockstore.org/api/ga4gh/trs/v2/tools"


def _normalize_trs_id(value: str) -> str:
    v = value.strip()
    if v.startswith("#workflow/"):
        return v
    if v.startswith("github.com/"):
        return f"#workflow/{v}"
    return v


def record_from_tool(payload: dict[str, Any], source: str) -> OutputRecord:
    tid = str(payload.get("id") or "").strip()
    if not tid:
        raise FetchError("Dockstore tool missing id")
    name = str(payload.get("name") or payload.get("toolname") or tid)
    desc = str(payload.get("description") or "")
    org = payload.get("organization")
    if org:
        desc = (desc + "\n" if desc else "") + f"organization: {org}"
    landing = f"https://dockstore.org/workflows/{tid.removeprefix('#workflow/')}"
    idents = [Identifier("dockstore", tid, source)]
    if tid.startswith("#workflow/github.com/"):
        repo = tid.removeprefix("#workflow/")
        idents.append(Identifier("github", repo.removeprefix("github.com/"), "dockstore.trs_id"))
    return OutputRecord(
        output_id=f"workflow:dockstore:{tid}",
        kind="workflow",
        title=name,
        summary=desc,
        identifiers=idents,
        landing_url=landing,
        source_status="fetched" if source.startswith("dockstore") else "fixture",
        license_hint="Dockstore public TRS metadata; workflow run data not fetched",
        evidence_urls=[landing, f"{TRS}/{quote(tid, safe='')}"],
        reuse={"public_metadata": True, "matrices_downloaded": False, "run_data": False},
    )


def fetch_tool(value: str) -> OutputRecord:
    tid = _normalize_trs_id(value)
    # Prefer direct TRS id; fall back to name search.
    try:
        payload = get_json(f"{TRS}/{quote(tid, safe='')}")
        if isinstance(payload, dict) and payload.get("id"):
            return record_from_tool(payload, source="dockstore_trs")
    except FetchError:
        pass
    name = tid.rsplit("/", 1)[-1]
    found = get_json(f"{TRS}?name={quote(name)}")
    if isinstance(found, list):
        for tool in found:
            if str(tool.get("id") or "") == tid or str(tool.get("name") or "") == name:
                return record_from_tool(tool, source="dockstore_trs_search")
    raise FetchError(f"Dockstore tool not found: {value!r}")


def records_from_tools_fixture(payload: Any, source: str = "fixture") -> list[OutputRecord]:
    tools = payload.get("tools") if isinstance(payload, dict) else payload
    out: list[OutputRecord] = []
    for raw in tools or []:
        try:
            out.append(record_from_tool(raw, source=source))
        except FetchError:
            continue
    return out

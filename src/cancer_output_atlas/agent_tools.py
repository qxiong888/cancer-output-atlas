"""Official Strands tool wrappers around the deterministic atlas steps.

Each function uses the real decorator when strands-agents is installed.
They remain ordinary callables for the unattended pipeline and offline tests.
"""

from __future__ import annotations

import json
from typing import Any

from cancer_output_atlas.classify import GOAL_DEFAULT, classify
from cancer_output_atlas.link import link_outputs
from cancer_output_atlas.rank import rank_records
from cancer_output_atlas.runtime import tool
from cancer_output_atlas.safety import one_click_allowed, refuse_reason
from cancer_output_atlas.schema import Identifier, OutputRecord, record_to_dict


@tool
def refuse_unsafe(request: str) -> str:
    """Refuse dbGaP, PHI, controlled access, or non-public one-click calls.

    Args:
        request: URL, accession, or free text the caller wants to fetch or open.
    """
    reason = refuse_reason(request)
    if reason:
        return json.dumps({"allowed": False, "reason": reason})
    if request.startswith(("http://", "https://")) and not one_click_allowed(request):
        return json.dumps(
            {
                "allowed": False,
                "reason": "one-click only for already-public metadata APIs",
            }
        )
    if request.startswith("gs://"):
        return json.dumps(
            {
                "allowed": False,
                "reason": "object-store URIs are pointer-only; never downloaded",
            }
        )
    return json.dumps({"allowed": True, "reason": "no safety hit on this text"})


@tool
def classify_output_json(record_json: str, goal: str = GOAL_DEFAULT) -> str:
    """Classify one public output from a JSON record. Does not invent IDs.

    Args:
        record_json: JSON object with title, summary, identifiers, kind, landing_url.
        goal: Reuse goal used for profile_fit.
    """
    raw = json.loads(record_json)
    idents = [
        Identifier(i["scheme"], i["value"], i.get("source", "tool"))
        for i in raw.get("identifiers") or []
    ]
    rec = OutputRecord(
        output_id=raw.get("output_id") or "output:unknown",
        kind=raw.get("kind") or "dataset",
        title=raw.get("title") or "",
        summary=raw.get("summary") or "",
        identifiers=idents,
        landing_url=raw.get("landing_url") or "",
        source_status=raw.get("source_status") or "fixture",
        organism=raw.get("organism"),
        biosystems=list(raw.get("biosystems") or []),
        assay=raw.get("assay"),
        perturbation=raw.get("perturbation"),
    )
    classify(rec, goal=goal)
    return json.dumps(record_to_dict(rec))


@tool
def link_outputs_json(records_json: str) -> str:
    """Build reuse links among classified output records.

    Args:
        records_json: JSON list of output records.
    """
    from cancer_output_atlas.pipeline import records_from_json

    records = records_from_json(json.loads(records_json))
    for rec in records:
        if rec.classification is None and rec.source_status != "skipped":
            classify(rec)
    links = [l.__dict__ for l in link_outputs(records)]
    return json.dumps(links)


@tool
def rank_by_goal_json(records_json: str, goal: str = GOAL_DEFAULT, top: int = 10) -> str:
    """Rank classified outputs for a reuse goal.

    Args:
        records_json: JSON list of output records.
        goal: Reuse goal text.
        top: Maximum rows to return.
    """
    from cancer_output_atlas.pipeline import records_from_json

    records = records_from_json(json.loads(records_json))
    for rec in records:
        if rec.classification is None and rec.source_status != "skipped":
            classify(rec, goal=goal)
    return json.dumps(rank_records(records, goal=goal, top=top))


@tool
def emit_digest_text(what: str, why: str, ids: str) -> str:
    """Format a what / why / ids digest block.

    Args:
        what: What the atlas produced this run.
        why: Why those outputs were kept or ranked.
        ids: Observed public identifiers only (never invented).
    """
    return f"## What\n{what}\n\n## Why\n{why}\n\n## IDs\n{ids}\n"


SYSTEM_PROMPT = """You are the Cancer Output Atlas agent.
You classify and link PUBLIC cancer research outputs
(dataset, software, workflow, model, trial, sample).
Hard rules:
- Never invent GEO, SRA, DOI, or NCT identifiers.
- If a fetch fails, skip. Do not guess the record.
- Public metadata only. No PHI, no dbGaP, no controlled access.
- Do not download h5ad or other omics payloads. gs:// is a pointer only.
- One-click only for already-public remotely callable metadata APIs.
- Prefer findability and reuse over narrative claims.
"""


def tool_list() -> list[Any]:
    return [
        refuse_unsafe,
        classify_output_json,
        link_outputs_json,
        rank_by_goal_json,
        emit_digest_text,
    ]


def make_find_tool(records, *, graph_path=None, cache=None, top_default: int = 40):
    """Strands tool wrapping find_by_goal on the baked graph. Ranking is find_by_goal."""
    from cancer_output_atlas.find import find_by_goal

    cache = cache if cache is not None else {}
    cache.setdefault("trace", [])
    cache.setdefault("tools_used", [])

    @tool
    def find_public_outputs(goal: str, top: int = 40) -> str:
        """Rank baked-graph public cancer outputs for a reuse goal. Never invents IDs.

        Args:
            goal: User goal text.
            top: Maximum hits to keep per ODS class.
        """
        n = int(top) if top else top_default
        n = max(1, min(n, 80))
        payload = find_by_goal(records, goal, top=n, graph_path=str(graph_path) if graph_path else None)
        cache["payload"] = payload
        if "find_public_outputs" not in cache["tools_used"]:
            cache["tools_used"].append("find_public_outputs")
        cache["trace"].append(
            {
                "tool": "find_public_outputs",
                "args": {"goal": (goal or "")[:240], "top": n},
            }
        )
        return json.dumps(
            {
                "matched_total": payload.get("matched_total"),
                "matched_by_ods": payload.get("matched_by_ods"),
                "abstain": payload.get("abstain"),
                "generated_at_utc": payload.get("generated_at_utc"),
            },
            ensure_ascii=False,
        )

    return find_public_outputs

"""What / why / ids digest — no invented identifiers."""

from __future__ import annotations

from collections import defaultdict

from cancer_output_atlas.schema import AtlasGraph, OutputRecord


def collect_ids(records: list[OutputRecord]) -> dict[str, list[str]]:
    by_scheme: dict[str, set[str]] = defaultdict(set)
    for rec in records:
        if rec.source_status == "skipped":
            continue
        for ident in rec.identifiers:
            by_scheme[ident.scheme].add(ident.value)
    return {k: sorted(v) for k, v in sorted(by_scheme.items())}


def render_digest(graph: AtlasGraph, ranked: list[dict] | None = None) -> str:
    live = [n for n in graph.nodes if n.source_status != "skipped"]
    ids = collect_ids(live)
    kinds: dict[str, int] = defaultdict(int)
    for n in live:
        kinds[n.kind] += 1
    lines = [
        "# Cancer Output Atlas digest",
        "",
        "## What",
        f"- Goal: {graph.goal}",
        f"- Classified outputs: {len(live)}",
        f"- Skipped: {len(graph.skipped)}",
        f"- Reuse links: {len(graph.links)}",
        f"- Kinds: " + (", ".join(f"{k}={v}" for k, v in sorted(kinds.items())) or "none"),
        "",
        "## Why",
        "- This atlas classifies *research outputs* and links them for findability/reuse.",
        "- It is not an evidence graph, not RAG, and not a drug–gene–disease spine.",
        "- Classification is rule-based on public metadata text (no invented GEO/SRA/DOI/NCT IDs).",
        "- Fetch failures are skipped. Large omics files and object-store payloads are never downloaded.",
        "- One-click is allowed only for already-public remotely callable metadata APIs.",
        "- Rank-by-goal prefers public cancer research outputs (NSCLC / immuno-oncology / CRISPRi-in-cancer).",
        "- Every public catalog record we can snapshot is ingested; controlled resources are apply-yourself pointers.",
        "",
        "## IDs",
    ]
    pin: set[str] = set()
    try:
        from cancer_output_atlas.sources.catalog import SEEDS

        pin = {s.value for s in SEEDS}
    except Exception:
        pin = set()
    if not ids:
        lines.append("- none (no live records)")
    else:
        for scheme, values in ids.items():
            shown = values
            extra = ""
            if len(values) > 40:
                pinned = [v for v in values if v in pin]
                rest = [v for v in values if v not in pin][:20]
                shown = pinned + rest
                extra = f" … +{len(values)-len(shown)} more"
            lines.append(f"- {scheme} ({len(values)}): {', '.join(shown)}{extra}")
    if graph.skipped:
        lines.extend(["", "## Skipped"])
        for item in graph.skipped:
            lines.append(f"- {item.get('seed', '?')}: {item.get('reason', 'skipped')}")
    if ranked:
        lines.extend(["", "## Ranked for goal"])
        for row in ranked[:25]:
            id_s = ", ".join(row.get("ids") or [])
            lines.append(
                f"- #{row['rank']} ({row['rank_score']}) {row['title']} [{id_s}] — {row['why']}"
            )
    lines.append("")
    return "\n".join(lines)


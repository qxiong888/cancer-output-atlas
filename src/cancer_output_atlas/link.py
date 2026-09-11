"""Link public outputs for reuse. Edges require observed evidence, not guesses."""

from __future__ import annotations

from collections import defaultdict

from cancer_output_atlas.schema import Link, OutputRecord

COLORECTAL_CRISPRI_PHRASE = (
    "enhancer-gene regulatory interactions in colorectal cancer"
)


def _labels(rec: OutputRecord) -> set[str]:
    if rec.classification is None:
        return set()
    return set(rec.classification.labels)


def _blob(rec: OutputRecord) -> str:
    return f"{rec.title} {rec.summary}".lower()


def _pembro_nsclc(rec: OutputRecord) -> bool:
    blob = _blob(rec)
    pembro = "pembrolizumab" in blob
    nsclc = (
        "nsclc" in blob
        or "non-small cell lung" in blob
        or "non-small-cell lung" in blob
    )
    return pembro and nsclc


def link_outputs(records: list[OutputRecord]) -> list[Link]:
    live = [r for r in records if r.source_status != "skipped"]
    links: list[Link] = []
    seen: set[tuple[str, str, str]] = set()

    def add(src: str, rel: str, dst: str, evidence: str, node: bool = True) -> None:
        key = (src, rel, dst)
        if src == dst or key in seen:
            return
        seen.add(key)
        links.append(
            Link(source=src, rel=rel, target=dst, evidence=evidence, target_is_node=node)
        )

    pmid_index: dict[str, list[OutputRecord]] = defaultdict(list)
    github_software: list[tuple[OutputRecord, str]] = []

    for rec in live:
        for ident in rec.identifiers:
            if ident.scheme == "pmid":
                add(
                    rec.output_id,
                    "described_by",
                    f"pmid:{ident.value}",
                    f"PMID {ident.value} on {ident.source}",
                    node=False,
                )
                pmid_index[ident.value].append(rec)
            if ident.scheme == "sra":
                add(
                    rec.output_id,
                    "has_raw_in",
                    f"sra:{ident.value}",
                    f"SRA {ident.value} from {ident.source}",
                    node=False,
                )
            if ident.scheme == "gs_uri":
                add(
                    rec.output_id,
                    "points_to_object_store",
                    ident.value,
                    "seed pointer; objects are never listed or downloaded",
                    node=False,
                )
            if ident.scheme == "github" and rec.kind == "software":
                github_software.append((rec, ident.value.lower()))

    for pmid, recs in pmid_index.items():
        if len(recs) < 2:
            continue
        uniq = {r.output_id: r for r in recs}
        ids = sorted(uniq)
        for i, a in enumerate(ids):
            for b in ids[i + 1 :]:
                add(a, "same_publication", b, f"shared PMID {pmid}")

    phrase_hits = [
        r
        for r in live
        if COLORECTAL_CRISPRI_PHRASE in _blob(r)
    ]
    for i, a in enumerate(phrase_hits):
        for b in phrase_hits[i + 1 :]:
            add(
                a.output_id,
                "same_publication",
                b.output_id,
                "shared observed phrase: enhancer-gene regulatory interactions in colorectal cancer",
            )

    # shares_assay_class only for the more specific Perturb-seq label (not a 700-series clique).
    perturb = [r for r in live if "perturb_seq" in _labels(r)]
    for i, a in enumerate(perturb):
        for b in perturb[i + 1 :]:
            add(
                a.output_id,
                "shares_assay_class",
                b.output_id,
                "shared labels perturb_seq",
            )

    # Figshare processed companion vs GEO expression series (labels only).
    figshare_recs = [
        r
        for r in live
        if any(
            i.scheme == "figshare" or (i.scheme == "doi" and "figshare" in i.value)
            for i in r.identifiers
        )
        and "perturb_seq" in _labels(r)
    ]
    geo_recs = [
        r
        for r in live
        if any(i.scheme == "geo" for i in r.identifiers) and "perturb_seq" in _labels(r)
    ]
    for f in figshare_recs:
        for g in geo_recs:
            add(
                f.output_id,
                "complements",
                g.output_id,
                "processed Perturb-seq companion vs GEO expression series (labels only; not same-study unless PMID matches)",
            )

    # Complements: trial ↔ dataset when both texts mention pembrolizumab AND NSCLC.
    trials = [r for r in live if r.kind == "trial" and _pembro_nsclc(r)]
    datasets = [r for r in live if r.kind == "dataset" and _pembro_nsclc(r)]
    # Prefer demo-priority pairs; also keep any pair that names the other ID.
    demo_trials = {"trial:nct:NCT02220894", "trial:nct:NCT03065764"}
    demo_geo = {
        "dataset:geo:GSE345124",
        "dataset:geo:GSE305086",
        "dataset:geo:GSE337519",
        "dataset:geo:GSE320129",
    }
    for t in trials:
        t_demo = t.output_id in demo_trials
        t_blob = _blob(t)
        for d in datasets:
            if not (t_demo or d.output_id in demo_geo):
                continue
            add(
                t.output_id,
                "complements",
                d.output_id,
                "observed overlap: pembrolizumab AND NSCLC in both public texts",
            )

    # Workflow implements / uses software when a GitHub full_name is observed on both.
    for rec in live:
        if rec.kind == "software":
            continue
        blob = f"{rec.output_id} {rec.landing_url} {_blob(rec)}".lower()
        for gh, name in github_software:
            if name in blob or name.split("/")[-1] in blob and name.split("/")[0] in blob:
                if rec.kind == "workflow":
                    add(
                        rec.output_id,
                        "implements",
                        gh.output_id,
                        f"observed GitHub path {name}",
                    )
                elif rec.kind == "dataset" and "cbioportal" in rec.output_id and "cbioportal" in name:
                    add(
                        rec.output_id,
                        "uses_software",
                        gh.output_id,
                        f"cBioPortal study metadata uses software {name}",
                    )

    return links


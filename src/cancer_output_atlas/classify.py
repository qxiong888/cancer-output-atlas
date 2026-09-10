"""Rule-based output classifier. Labels come from observed text only."""

from __future__ import annotations

import re
from typing import Iterable

from cancer_output_atlas.node_features import persist_ods
from cancer_output_atlas.schema import Classification, OutputRecord

KIND_HINTS: list[tuple[str, tuple[str, ...]]] = [
    ("trial", ("clinical trial", "nct0", "randomized trial")),
    ("software", ("github.com", "python package", "software tool", "command-line")),
    ("workflow", ("nextflow", "snakemake", "wdl workflow", "cwl workflow", "dockstore", "nf-core")),
    ("publication", ("preprint", "wellcome open research", "europe pmc", "abstract:")),
    ("model", ("foundation model", "trained model", "checkpoint", "virtual cell model")),
    ("sample", ("biosample", "gsm", "geo sample")),
    ("dataset", ("expression profiling", "dataset", "perturb-seq", "rna-seq", "geo series")),
]

LABEL_TERMS: dict[str, tuple[str, ...]] = {
    "crispri": ("crispri", "crispr interference", "dcas9-krab", "krabb"),
    "crispra": ("crispra", "crispr activation"),
    "crispr": ("crispr", "cas9", "perturb-seq", "crop-seq"),
    "gene_expression": (
        "gene expression",
        "rna-seq",
        "rnaseq",
        "transcriptom",
        "expression profiling",
        "scrna",
        "scRNA",
    ),
    "single_cell": ("single cell", "single-cell", "scrna", "perturb-seq", "crop-seq"),
    "perturb_seq": ("perturb-seq", "perturbseq", "crop-seq"),
    "nsclc": (
        "nsclc",
        "non-small cell lung",
        "non-small-cell lung",
        "non small cell lung",
        "lung adenocarcinoma",
    ),
    "immunotherapy": (
        "pembrolizumab",
        "immunotherapy",
        "immuno-oncology",
        "immune checkpoint",
        "immune-checkpoint",
        "pd-1",
        "pd-l1",
        "chemo-immunotherapy",
        "chemoimmunotherapy",
    ),
    "colorectal": ("colorectal",),
    "cancer": (
        "leukemia",
        "tumour",
        "tumor",
        "carcinoma",
        "hepg2",
        "jurkat",
        "melanoma",
        "oncolog",
        "hepatocellular",
        "chronic myelogenous",
        "chronic myeloid",
        "nsclc",
        "non-small cell lung",
        "non-small-cell lung",
        "lung adenocarcinoma",
        "lung cancer",
        "colorectal",
        "pembrolizumab",
        "immunotherapy",
        "cancer",
        "neoplasm",
        "somatic",
    ),
    "neuron": ("neuron", "ipsc-derived", "ipsc derived", "neural"),
    "public": ("public", "open-access", "open access", "cc by", "cc0"),
}

GOAL_DEFAULT = "NCI ODS output types: data, software, tools, methods, models, trial results, biospecimens (NSCLC / immuno-oncology / CRISPRi-in-cancer)"

# Cell-line / biosystem tokens observed in text. a549 is a lung line; k562 is
# recorded as a biosystem when present, never used as a cancer claim.
BIO_TERMS = (
    "a549",
    "k562",
    "hepg2",
    "jurkat",
    "rpe1",
    "h1 hesc",
    "hesc",
    "ipsc",
    "neuron",
    "ht29",
)


def _blob(rec: OutputRecord) -> str:
    parts = [rec.title, rec.summary, rec.kind, rec.landing_url]
    parts.extend(i.value for i in rec.identifiers)
    parts.extend(rec.biosystems)
    if rec.assay:
        parts.append(rec.assay)
    return " ".join(p for p in parts if p).lower()


def _find(blob: str, terms: Iterable[str]) -> list[str]:
    hit = []
    for t in terms:
        if t.lower() in blob:
            hit.append(t)
    return hit


def infer_kind(rec: OutputRecord) -> str:
    if rec.kind in {
        "dataset",
        "software",
        "workflow",
        "model",
        "trial",
        "sample",
        "publication",
    }:
        return rec.kind
    blob = _blob(rec)
    for kind, terms in KIND_HINTS:
        if _find(blob, terms):
            return kind
    schemes = {i.scheme for i in rec.identifiers}
    if "nct" in schemes:
        return "trial"
    if "github" in schemes:
        return "software"
    if "dockstore" in schemes or "nfcore" in schemes:
        return "workflow"
    if "europepmc" in schemes:
        return "publication"
    if "geo" in schemes or "doi" in schemes or "gs_uri" in schemes or "cbioportal" in schemes:
        return "dataset"
    return "dataset"


def infer_fields(rec: OutputRecord) -> None:
    blob = _blob(rec)
    if "perturb-seq" in blob or "perturbseq" in blob:
        rec.assay = rec.assay or "Perturb-seq"
        rec.modality = rec.modality or "single-cell RNA-seq"
    elif "crop-seq" in blob:
        rec.assay = rec.assay or "CROP-seq"
        rec.modality = rec.modality or "single-cell RNA-seq"
    elif "rna-seq" in blob or "expression profiling" in blob or "transcriptom" in blob:
        rec.assay = rec.assay or "RNA-seq"
        rec.modality = rec.modality or rec.modality or "RNA-seq"
    if "crispri" in blob or "crispr interference" in blob:
        rec.perturbation = rec.perturbation or "CRISPRi"
    elif "crispra" in blob:
        rec.perturbation = rec.perturbation or "CRISPRa"
    elif "crispr" in blob:
        rec.perturbation = rec.perturbation or "CRISPR"
    bios = list(rec.biosystems)
    for term in BIO_TERMS:
        if term in blob and term not in {b.lower() for b in bios}:
            bios.append(term)
    rec.biosystems = bios


def profile_fit(labels: list[str], rec: OutputRecord, goal: str) -> float:
    """How well this output matches the demo reuse goal. Rule score in [0, 1]."""
    score = 0.0
    lab = set(labels)
    if "public" in lab or rec.source_status in {"fetched", "fixture", "pointer"}:
        score += 0.10
    if "nsclc" in lab:
        score += 0.22
    if "immunotherapy" in lab:
        score += 0.16
    if "colorectal" in lab:
        score += 0.10
    if "crispri" in lab and "cancer" in lab:
        score += 0.22  # CRISPRi-in-cancer
    elif "crispri" in lab:
        score += 0.10
    elif "crispr" in lab and "cancer" in lab:
        score += 0.10
    elif "crispr" in lab:
        score += 0.06
    if "gene_expression" in lab:
        score += 0.10
    if "single_cell" in lab or "perturb_seq" in lab:
        score += 0.08
    if "cancer" in lab:
        score += 0.12
    goal_l = goal.lower()
    blob = _blob(rec)
    if "nsclc" in goal_l and ("nsclc" in blob or "non-small cell lung" in blob or "lung adenocarcinoma" in blob):
        score += 0.06
    if "immuno" in goal_l and ("immunotherap" in blob or "pembrolizumab" in blob):
        score += 0.04
    if rec.source_status == "pointer":
        score -= 0.05  # reusable in principle, but this atlas does not fetch objects
    return round(max(0.0, min(1.0, score)), 3)


def classify(rec: OutputRecord, goal: str = GOAL_DEFAULT) -> OutputRecord:
    if rec.source_status == "skipped":
        rec.classification = Classification(
            kind=rec.kind or "dataset",
            labels=[],
            profile_fit=0.0,
            rationale=rec.skip_reason or "skipped",
        )
        persist_ods(rec)
        return rec
    infer_fields(rec)
    rec.kind = infer_kind(rec)
    blob = _blob(rec)
    labels: list[str] = []
    matched: list[str] = []
    for label, terms in LABEL_TERMS.items():
        hits = _find(blob, terms)
        if hits:
            labels.append(label)
            matched.extend(hits)
    if rec.source_status in {"fetched", "fixture", "pointer"} and "public" not in labels:
        labels.append("public")
    rec.classification = Classification(
        kind=rec.kind,
        labels=labels,
        profile_fit=profile_fit(labels, rec, goal),
        rationale=(
            f"kind={rec.kind}; labels={','.join(labels) or 'none'}; "
            f"terms={','.join(sorted(set(matched))) or 'none'}"
        ),
        matched_terms=sorted(set(matched)),
    )
    rec.reuse = {
        **(rec.reuse or {}),
        "public_metadata": True,
        "matrices_downloaded": False,
        "one_click": "metadata APIs only",
    }
    persist_ods(rec)
    return rec

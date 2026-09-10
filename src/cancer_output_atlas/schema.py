"""Research-output atlas schema (not an evidence / RAG graph).

Nodes are public *outputs* (dataset, software, workflow, model, trial, sample, publication).
Edges are reuse links (deposited, described-by, same-study, complements, …).
Identifiers are only those observed on a public landing or API — never invented.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class OutputKind(str, Enum):
    DATASET = "dataset"
    SOFTWARE = "software"
    WORKFLOW = "workflow"
    MODEL = "model"
    TRIAL = "trial"
    SAMPLE = "sample"
    PUBLICATION = "publication"


class IdScheme(str, Enum):
    GEO = "geo"
    SRA = "sra"
    DOI = "doi"
    NCT = "nct"
    PMID = "pmid"
    BIOPROJECT = "bioproject"
    GITHUB = "github"
    GS_URI = "gs_uri"
    FIGSHARE = "figshare"
    CBIOPORTAL = "cbioportal"
    DOCKSTORE = "dockstore"
    GDC_PROJECT = "gdc_project"
    TCIA_COLLECTION = "tcia_collection"
    NFCORE = "nfcore"
    EUROPEPMC = "europepmc"
    CELLXGENE_COLLECTION = "cellxgene_collection"
    CELLXGENE_DATASET = "cellxgene_dataset"
    HUGGINGFACE = "huggingface"
    HCA_PROJECT = "hca_project"
    ZENODO = "zenodo"
    URL = "url"
    TESTLOCAL = "testlocal"


class LinkRel(str, Enum):
    DESCRIBED_BY = "described_by"
    HAS_RAW_IN = "has_raw_in"
    SAME_PUBLICATION = "same_publication"
    SHARES_ASSAY_CLASS = "shares_assay_class"
    COMPLEMENTS = "complements"
    POINTS_TO_OBJECT_STORE = "points_to_object_store"
    IMPLEMENTS = "implements"
    USES_SOFTWARE = "uses_software"
    GENERATED_BY_WORKFLOW = "generated_by_workflow"


class SourceStatus(str, Enum):
    FETCHED = "fetched"
    FIXTURE = "fixture"
    POINTER = "pointer"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class Identifier:
    scheme: str
    value: str
    source: str

    def key(self) -> str:
        return f"{self.scheme}:{self.value}"


@dataclass
class Classification:
    kind: str
    labels: list[str]
    profile_fit: float
    rationale: str
    matched_terms: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class DiseaseType:
    """One ontology-backed disease type from a public source field.

    Disease-agnostic contract: this atlas stores ontology="oncotree" (official
    MSK codes). A later autoimmune atlas may store ontology="mondo" in the
    same field without a schema rewrite. Do not invent codes.
    """

    ontology: str
    code: str
    name: str
    source: str


# Backward-compatible name for OncoTree-only callers.
CancerType = DiseaseType


# NCI ODS Track 1 slugs. Never invent an 8th type.
ODS_VALUES = (
    "data",
    "software",
    "tool",
    "method",
    "model",
    "trial_result",
    "biospecimen",
)
CALLABILITY_VALUES = (
    "metadata_api",
    "runnable_workflow",
    "hosted_tool",
    "repo_only",
    "pointer_apply",
)
ACCESS_VALUES = (
    "public_metadata",
    "apply_yourself",
    "controlled_pointer",
)
CATALOG_VALUES = (
    "geo",
    "cbioportal",
    "nct",
    "gdc",
    "tcia",
    "figshare",
    "github",
    "dockstore",
    "nfcore",
    "europepmc",
    "pointer",
    "other",
)


@dataclass
class OutputRecord:
    """Public research-output node.

    The typed nullable fields below are the semantic-adapter contract for a
    future adapter. Bind to these fields only — never leftover title tokens,
    never embeddings, never classification.profile_fit as a product feature.
    Empty is better than a guessed value.
    """

    output_id: str
    kind: str
    title: str
    summary: str
    identifiers: list[Identifier]
    landing_url: str
    source_status: str
    skip_reason: str | None = None
    license_hint: str | None = None
    organism: str | None = None
    biosystems: list[str] = field(default_factory=list)
    assay: str | None = None
    perturbation: str | None = None
    modality: str | None = None
    n_samples: int | None = None
    classification: Classification | None = None
    reuse: dict[str, Any] = field(default_factory=dict)
    evidence_urls: list[str] = field(default_factory=list)
    # --- semantic-adapter contract (typed, nullable; empty > guessed) ---
    disease_types: list[DiseaseType] = field(default_factory=list)
    ods: str | None = None
    ods_zh: str | None = None
    callability: str | None = None
    access: str | None = None
    catalog: str | None = None
    year: int | None = None
    interventions: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    reuse_degree: int = 0
    link_ok: bool | None = None
    checked_at: str | None = None

    def identifier_values(self, scheme: str | None = None) -> list[str]:
        return [
            i.value
            for i in self.identifiers
            if scheme is None or i.scheme == scheme
        ]

    def all_id_keys(self) -> set[str]:
        return {i.key() for i in self.identifiers}

    @property
    def cancer_types(self) -> list[DiseaseType]:
        """OncoTree subset of disease_types (backward-compatible alias)."""
        return [d for d in self.disease_types if d.ontology == "oncotree"]


@dataclass
class Link:
    source: str
    rel: str
    target: str
    evidence: str
    target_is_node: bool = True


@dataclass
class AtlasGraph:
    goal: str
    generated_at: str
    nodes: list[OutputRecord]
    links: list[Link]
    skipped: list[dict[str, str]]
    policy: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": "cancer_output_atlas.v1",
            "goal": self.goal,
            "generated_at": self.generated_at,
            "policy": self.policy,
            "nodes": [record_to_dict(n) for n in self.nodes],
            "links": [asdict(l) for l in self.links],
            "skipped": self.skipped,
        }


def record_to_dict(rec: OutputRecord) -> dict[str, Any]:
    d = asdict(rec)
    d["identifiers"] = [asdict(i) for i in rec.identifiers]
    if rec.classification is not None:
        d["classification"] = asdict(rec.classification)
    d["disease_types"] = [asdict(x) for x in rec.disease_types]
    d["cancer_types"] = [asdict(x) for x in rec.cancer_types]
    return d


def classification_row(rec: OutputRecord) -> dict[str, Any]:
    from cancer_output_atlas.ods import ODS, ods_category, ods_zh

    c = rec.classification
    if rec.ods in ODS:
        ods = rec.ods
        zh = rec.ods_zh or ods_zh(ods)
    else:
        ods = ods_category(
            kind=rec.kind,
            title=rec.title,
            summary=rec.summary,
            ids=[i.key() for i in rec.identifiers],
            landing_url=rec.landing_url,
            reuse=rec.reuse,
        )
        zh = ods_zh(ods)
    return {
        "output_id": rec.output_id,
        "kind": rec.kind,
        "ods": ods,
        "ods_zh": zh,
        "title": rec.title,
        "profile_fit": None if c is None else c.profile_fit,
        "labels": [] if c is None else c.labels,
        "perturbation": rec.perturbation,
        "assay": rec.assay,
        "modality": rec.modality,
        "organism": rec.organism,
        "biosystems": rec.biosystems,
        "landing_url": rec.landing_url,
        "ids": [i.key() for i in rec.identifiers],
        "source_status": rec.source_status,
        "rationale": None if c is None else c.rationale,
    }


POLICY = {
    "public_metadata_only": True,
    "invent_ids": False,
    "skip_on_fetch_failure": True,
    "phi": False,
    "dbgap": False,
    "download_h5ad_or_omics": False,
    "one_click": "already-public remotely callable metadata APIs only",
}

# Schemes that identify *this* output (not a related paper).
IDENTITY_SCHEMES = frozenset(
    {
        "geo",
        "nct",
        "doi",
        "cbioportal",
        "dockstore",
        "gdc_project",
        "tcia_collection",
        "figshare",
        "nfcore",
        "europepmc",
        "gs_uri",
        "cellxgene_collection",
        "cellxgene_dataset",
        "huggingface",
        "hca_project",
        "zenodo",
    }
)

"""Seed catalog of *known public* starting points plus bulk catalog loaders.

Demo seeds are the verified cancer pack. Old CRISPRi / Arc pointers live in
BACKGROUND_SEEDS and are not ingested by default.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# User-supplied public starting points. May be discovered; never invent extras.
SEEDS: list["Seed"]


@dataclass(frozen=True)
class Seed:
    scheme: str
    value: str
    fetcher: str  # geo | figshare | clinicaltrials | cbioportal | dockstore | github | pointer | europepmc
    landing_url: str


SEEDS = [
    Seed(
        "nct",
        "NCT02220894",
        "clinicaltrials",
        "https://clinicaltrials.gov/study/NCT02220894",
    ),
    Seed(
        "nct",
        "NCT03065764",
        "clinicaltrials",
        "https://clinicaltrials.gov/study/NCT03065764",
    ),
    Seed(
        "geo",
        "GSE320129",
        "geo",
        "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE320129",
    ),
    Seed(
        "geo",
        "GSE305086",
        "geo",
        "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE305086",
    ),
    Seed(
        "geo",
        "GSE345124",
        "geo",
        "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE345124",
    ),
    Seed(
        "geo",
        "GSE337519",
        "geo",
        "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE337519",
    ),
    Seed(
        "doi",
        "10.6084/m9.figshare.32743536.v1",
        "figshare",
        "https://figshare.com/articles/dataset/Extended_data_for_Perturb-seq_analysis/32743536",
    ),
    Seed(
        "cbioportal",
        "luad_tcga_pan_can_atlas_2018",
        "cbioportal",
        "https://www.cbioportal.org/study/summary?id=luad_tcga_pan_can_atlas_2018",
    ),
    Seed(
        "dockstore",
        "github.com/nf-core/oncoanalyser",
        "dockstore",
        "https://dockstore.org/workflows/github.com/nf-core/oncoanalyser",
    ),
    Seed(
        "github",
        "cBioPortal/cbioportal",
        "github",
        "https://github.com/cBioPortal/cbioportal",
    ),
]

# Method-background CRISPRi / Arc pointer. Not the cancer demo.
BACKGROUND_SEEDS: list[Seed] = [
    Seed(
        "geo",
        "GSE264667",
        "geo",
        "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE264667",
    ),
    Seed(
        "geo",
        "GSE90546",
        "geo",
        "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE90546",
    ),
    Seed(
        "geo",
        "GSE124703",
        "geo",
        "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE124703",
    ),
    Seed(
        "geo",
        "GSE152988",
        "geo",
        "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE152988",
    ),
    Seed(
        "doi",
        "10.25452/figshare.plus.20029387",
        "figshare",
        "https://doi.org/10.25452/figshare.plus.20029387",
    ),
    Seed(
        "gs_uri",
        "gs://arc-institute-virtual-cell-atlas/virtual-cell-challenge/",
        "pointer",
        "https://github.com/ArcInstitute/arc-virtual-cell-atlas",
    ),
]

ARC_POINTER = BACKGROUND_SEEDS[-1]

BACKGROUND_GEO = frozenset(s.value for s in BACKGROUND_SEEDS if s.scheme == "geo")

# Quinn 2026-08-28: these BACKGROUND GEO series must enter the default offline graph.
DEFAULT_OFFLINE_BACKGROUND_GEO = frozenset({"GSE90546", "GSE264667"})

# Apply-yourself pointers (not scraped / not downloaded).
POINTER_CARDS: list[Seed] = [
    Seed(
        "url",
        "https://cancer.sanger.ac.uk/cosmic",
        "pointer",
        "https://cancer.sanger.ac.uk/cosmic",
    ),
    Seed(
        "url",
        "https://docs.humantumoratlas.org/data_access/portal/",
        "pointer",
        "https://docs.humantumoratlas.org/data_access/portal/",
    ),
    Seed(
        "url",
        "https://docs.gdc.cancer.gov/API/Users_Guide/Getting_Started/",
        "pointer",
        "https://docs.gdc.cancer.gov/API/Users_Guide/Getting_Started/",
    ),
]

# Named catalog snapshots ingested in full (deduped by observed ID).
CATALOG_NAMES = (
    "cbioportal_studies",
    "clinicaltrials",
    "gdc_projects",
    "tcia_collections",
    "nfcore_cancer",
    "dockstore_tools",
    "geo_cancer_crispri",
    "geo_virtual_cell",
    "geo_lung_cancer",
    "geo_cancer_extra",
    "github_repos",
    "europepmc_papers",
    "catalog_notes",
    "huggingface_cards",
    "cellxgene_cancer",
    "hf_virtual_cell",
    "hca_cancer",
    "scperturb",
)


def fixtures_dir() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        cand = parent / "fixtures"
        if cand.is_dir() and any(cand.glob("*.json")):
            return cand
    return here.parents[3] / "fixtures"


def load_fixture_json(name: str) -> Any:
    path = fixtures_dir() / name
    return json.loads(path.read_text(encoding="utf-8"))


def seed_key(seed: Seed) -> str:
    return f"{seed.scheme}:{seed.value}"


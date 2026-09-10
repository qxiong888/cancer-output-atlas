"""Map atlas nodes onto NCI ODS Track 1 output types.

Official list (NCI ODS Impact Prize): data, software, tools,
methods and protocols, models, clinical trial results, biospecimens.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Mapping
from urllib.parse import urlparse

# English slug used in tables / JSON
ODS = (
    "data",
    "software",
    "tool",
    "method",
    "model",
    "trial_result",
    "biospecimen",
)

ODS_ZH = {
    "data": "数据",
    "software": "软件",
    "tool": "工具",
    "method": "方法",
    "model": "模型",
    "trial_result": "试验结果",
    "biospecimen": "生物样本",
}

TOOL_HINTS = (
    "portal",
    "viewer",
    "catalog",
    "browser",
    "workbench",
    "commons",
    "itcr",
    "informatics tool",
    "analysis tool",
    "web tool",
    "dashboard",
)

METHOD_HINTS = (
    "protocol",
    "pipeline",
    "workflow",
    "nextflow",
    "snakemake",
    "wdl",
    "cwl",
    "dockstore",
    "nf-core",
    "method",
)

# Publications are not methods unless they describe a protocol/pipeline/workflow.
PUBLICATION_METHOD_HINTS = (
    "protocol",
    "pipeline",
    "workflow",
    "nextflow",
    "snakemake",
    "wdl",
    "cwl",
    "dockstore",
    "nf-core",
    "methods and protocols",
)

# Bare "specimen" matches GEO series titles ("clinical specimens") and is too wide.
# "hcmi" is a models program/catalog, not a biospecimen class by itself.
BIOSPECIMEN_HINTS = (
    "biospecimen bank",
    "biobank",
    "tissue bank",
    "cell line catalog",
)
_TITLE_BIOSPECIMEN = (
    "tissue bank",
    "biobank",
    "biospecimen bank",
    "moonshot biobank",
    "chernobyl tissue",
)

_GSE_RE = re.compile(r"^GSE\d+$", re.I)
_TRUE = frozenset({"true", "yes", "1"})
_FALSE = frozenset({"false", "no", "0"})
_RESULTS_PHRASES = (
    "has results",
    "has_results",
    "hasresults",
    "results posted",
    "study results posted",
    "results first posted",
    "posted study results",
    "posted results",
    "results submitted and posted",
    "has results: true",
    "hasresults: true",
)

# Portal *homepages* (not a specific collection/project/dataset id).
_PORTAL_ROOT_HOST_PATHS = frozenset(
    {
        "www.humancellatlas.org",
        "humancellatlas.org",
        "data.humancellatlas.org",
        "cellxgene.cziscience.com",
        "chanzuckerberg.github.io/cellxgene-census",
        "arcinstitute.org/tools/virtualcellatlas",
        "depmap.org",
        "depmap.org/portal",
        "virtualcellchallenge.org",
        "lamin.ai/laminlabs/arc-virtual-cell-atlas",
        "www.lamin.ai/laminlabs/arc-virtual-cell-atlas",
        "www.cbioportal.org",
        "cbioportal.org",
        "portal.gdc.cancer.gov",
        "galaxyproject.org",
        "www.galaxyproject.org",
        "galaxyp.org",
        "www.galaxyp.org",
    }
)
_PORTAL_TITLES = frozenset(
    {
        "human cell atlas",
        "human cell atlas data portal",
        "cz cellxgene discover",
        "cellxgene discover",
        "cz cellxgene discover census (documentation landing)",
        "arc virtual cell atlas",
        "the cancer dependency map portal",
        "depmap portal",
        "virtual cell challenge",
        "arc virtual cell atlas (lamindb mirror)",
        "cbioportal for cancer genomics",
        "gdc data portal",
        "galaxy",
        "galaxy-p-multi-omics",
    }
)
_SPECIFIC_PATH = re.compile(
    r"/collections/|/projects/|/e/[^/]+\.cxg|/study/|/studies/"
    r"|/datasets/|/acc\.cgi|/explore/projects/",
    re.I,
)


def _blob(*parts: str) -> str:
    return " ".join(p for p in parts if p).lower()


def _schemes(ids: Iterable[str]) -> set[str]:
    out: set[str] = set()
    for raw in ids:
        if ":" in raw:
            out.add(raw.split(":", 1)[0].lower())
        else:
            out.add(raw.lower())
    return out


def _id_values(ids: Iterable[str]) -> list[str]:
    out: list[str] = []
    for raw in ids:
        if ":" in raw:
            out.append(raw.split(":", 1)[1])
        else:
            out.append(raw)
    return out


def is_geo_series(ids: Iterable[str], schemes: set[str] | None = None) -> bool:
    schemes = schemes if schemes is not None else _schemes(ids)
    if "geo" in schemes:
        return True
    return any(_GSE_RE.match((v or "").strip()) for v in _id_values(ids))


def _truthy(val: Any) -> bool | None:
    if val is True:
        return True
    if val is False:
        return False
    if isinstance(val, (int, float)) and val in {0, 1}:
        return bool(val)
    if isinstance(val, str):
        key = val.strip().lower()
        if key in _TRUE:
            return True
        if key in _FALSE:
            return False
    return None


def trial_results_status(
    *,
    title: str = "",
    summary: str = "",
    reuse: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> bool | None:
    """True = posted results, False = known none, None = unknown (do not drop)."""
    for src in (reuse, extra):
        if not src:
            continue
        for key in ("has_results", "hasResults", "has_results_posted"):
            flag = _truthy(src.get(key))
            if flag is not None:
                return flag
    blob = _blob(title, summary)
    if any(p in blob for p in _RESULTS_PHRASES):
        return True
    return None


def has_posted_trial_results(
    *,
    title: str = "",
    summary: str = "",
    reuse: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> bool:
    """NCI ODS 'clinical trial results' means HAS results, not a protocol card."""
    status = trial_results_status(title=title, summary=summary, reuse=reuse, extra=extra)
    return status is True


_CTGOV_STATUS_EN = {
    "RECRUITING": "Recruiting",
    "NOT_YET_RECRUITING": "Not yet recruiting",
    "ACTIVE_NOT_RECRUITING": "Active, not recruiting",
    "ENROLLING_BY_INVITATION": "Enrolling by invitation",
    "COMPLETED": "Completed",
    "TERMINATED": "Terminated",
    "WITHDRAWN": "Withdrawn",
    "SUSPENDED": "Suspended",
    "UNKNOWN": "Status unknown",
    "NO_LONGER_AVAILABLE": "No longer available",
    "AVAILABLE": "Available",
    "APPROVED_FOR_MARKETING": "Approved for marketing",
    "TEMPORARILY_NOT_AVAILABLE": "Temporarily not available",
}


def trial_status_label(
    *,
    title: str = "",
    summary: str = "",
    reuse: Mapping[str, Any] | None = None,
) -> str:
    """English badge: CT.gov overallStatus plus whether results are posted."""
    reuse = reuse or {}
    raw = reuse.get("overall_status") or reuse.get("overallStatus") or ""
    key = str(raw).strip().upper().replace(" ", "_")
    phase = _CTGOV_STATUS_EN.get(key) or (str(raw).replace("_", " ").title() if raw else "")
    posted = trial_results_status(title=title, summary=summary, reuse=reuse)
    if posted is True:
        return f"{phase}. Results posted" if phase else "Results posted"
    if posted is False:
        return f"{phase}. No results posted" if phase else "No results posted"
    return phase


def _host_path(url: str) -> tuple[str, str]:
    parsed = urlparse((url or "").strip())
    host = (parsed.netloc or "").lower()
    host_bare = host[4:] if host.startswith("www.") else host
    path = (parsed.path or "").rstrip("/")
    return f"{host}{path}", f"{host_bare}{path}"


def is_portal_root(*, kind: str = "", title: str = "", landing_url: str = "", text: str = "") -> bool:
    """True for a portal/catalog homepage, not a specific dataset or study page."""
    kind = (kind or "").lower()
    title_l = (title or "").strip().lower()
    if title_l in _PORTAL_TITLES:
        return True
    url = (landing_url or "").strip()
    if url:
        if _SPECIFIC_PATH.search(url):
            return False
        full, bare = _host_path(url)
        if full in _PORTAL_ROOT_HOST_PATHS or bare in _PORTAL_ROOT_HOST_PATHS:
            return True
        parsed = urlparse(url)
        path = (parsed.path or "").rstrip("/") or "/"
        if kind in {"catalog", "landing"} and path in {"", "/"}:
            return True
    blob = text or _blob(kind, title, landing_url)
    if kind in {"catalog", "landing"} and any(h in blob for h in ("portal", "catalog", "discover")):
        if not _SPECIFIC_PATH.search(landing_url or ""):
            return True
    return False


def ods_category(
    *,
    kind: str = "",
    title: str = "",
    summary: str = "",
    ids: Iterable[str] = (),
    landing_url: str = "",
    reuse: Mapping[str, Any] | None = None,
) -> str:
    """Return one ODS slug. Never invent a new category."""
    kind = (kind or "").lower()
    id_list = list(ids)
    schemes = _schemes(id_list)
    text = _blob(kind, title, summary, landing_url, " ".join(id_list))

    geo = is_geo_series(id_list, schemes)
    title_l = (title or "").strip().lower()
    url_l = (landing_url or "").strip().lower()

    # GEO series / GSE accessions are data, unless this node is actually a sample/biobank.
    if geo:
        if kind == "sample":
            return "biospecimen"
        if kind in {"biobank"} or any(h in title_l for h in _TITLE_BIOSPECIMEN):
            return "biospecimen"
        return "data"

    if is_portal_root(kind=kind, title=title, landing_url=landing_url, text=text):
        return "tool"

    # ODS "clinical trial results" = an NCT record. A trial-shaped commons or CDAS
    # listing with no NCT is not trial_result (no 8th class).
    if "nct" in schemes:
        if has_posted_trial_results(title=title, summary=summary, reuse=reuse):
            return "trial_result"
        # Protocol-only NCT is hidden in find (see find_by_goal). Keep the slug.
        return "trial_result"

    # Study/project catalog pages are data even if the title mentions a biobank or HCMI.
    if schemes & {
        "cbioportal",
        "gdc_project",
        "cellxgene_collection",
        "cellxgene_dataset",
        "hca_project",
    }:
        return "data"

    if kind == "model":
        return "model"

    # Actual sample collections. Do not use summary mentions of "biospecimen" on hub pages.
    if kind in {"biobank"} or any(h in title_l for h in _TITLE_BIOSPECIMEN):
        return "biospecimen"
    if kind == "sample":
        if "tetramer" in title_l or "tetramer" in url_l or "reagent" in title_l:
            return "tool"
        return "biospecimen"

    if kind in {"workflow"} or "dockstore" in schemes or "nfcore" in schemes:
        return "method"
    if kind == "publication":
        # Incidental "pipeline"/"protocol" in the abstract does not make a paper a method.
        if any(h in title_l for h in PUBLICATION_METHOD_HINTS):
            return "method"
        # Not a protocol/pipeline paper. No 8th "paper" class.
        return "data"
    if kind == "software":
        if any(h in text for h in TOOL_HINTS):
            return "tool"
        return "software"
    if kind in {"dataset", "catalog"} or schemes & {
        "geo",
        "cbioportal",
        "gdc_project",
        "tcia_collection",
        "figshare",
        "sra",
        "bioproject",
        "cellxgene_collection",
        "cellxgene_dataset",
        "hca_project",
        "huggingface",
    }:
        if any(h in text for h in METHOD_HINTS) and kind == "workflow":
            return "method"
        return "data"
    if any(h in text for h in METHOD_HINTS):
        return "method"
    if any(h in text for h in TOOL_HINTS):
        return "tool"
    if kind == "pointer" and any(h in text for h in BIOSPECIMEN_HINTS):
        return "biospecimen"
    return "data"


def ods_zh(slug: str) -> str:
    return ODS_ZH.get(slug, slug)

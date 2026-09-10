"""Function/safety boundary gates for AFH find (not clinical advice)."""

from __future__ import annotations

import re
from dataclasses import dataclass

_WS = re.compile(r"\s+")
_NON = re.compile(r"[^a-z0-9\u4e00-\u9fff]+")

CLINICAL_ABSTAIN = (
    "This tool finds public cancer-research catalog metadata only. "
    "It does not diagnose, treat, prescribe, or give clinical advice. "
    "No public entries are returned for clinical-care goals."
)

OUT_OF_DOMAIN_ABSTAIN = (
    "No public cancer-catalog entries match that goal. "
    "This finder is scoped to public cancer-related resources on the baked graph."
)

WEAK_TOKEN_ABSTAIN = (
    "No public entries on this graph match that goal "
    "(query too generic or out of catalog scope)."
)

# Care / clinical intent (fail-closed).
_CLINICAL_RE = re.compile(
    r"(?i)\b("
    r"diagnos(?:e|is|ing|tic)|"
    r"prescrib(?:e|ing|ed)|"
    r"treatment\s+plan|"
    r"care\s+plan|"
    r"best\s+treatment|"
    r"treatment\s+for\s+my|"
    r"right\s+for\s+my|"
    r"what\s+to\s+take|"
    r"tell\s+me\s+what\s+to\s+take|"
    r"my\s+patient|"
    r"the\s+patient|"
    r"start\s+tomorrow|"
    r"cure\s+rate|"
    r"what\s+drug\s+should|"
    r"should\s+i\s+(?:give|use|start)|"
    r"how\s+(?:do|should)\s+i\s+treat|"
    r"clinical\s+management|"
    r"medical\s+advice"
    r")\b"
)

# Injection / path probes (not catalog goals).
_PATH_PROBE_RE = re.compile(r"(?i)(/etc/|\betc/passwd\b|\bpasswd\b)")

# Strong cancer-domain signals (not exhaustive; prefer abstain if missing + non-cancer disease).
_CANCER_RE = re.compile(
    r"(?i)\b("
    r"cancer|tumor|tumour|oncolog|carcinoma|sarcoma|lymphoma|leukemia|leukaemia|"
    r"melanoma|glioma|gbm|glioblastoma|nsclc|sclc|luad|lusc|"
    r"breast|ovarian|prostate|pancreatic|colorectal|bladder|renal|kidney|"
    r"hepatocellular|hcc|thyroid|cervical|endometrial|myeloma|"
    r"pembrolizumab|keytruda|nivolumab|checkpoint|immunotherapy|"
    r"oncotree|tcga|cbioportal"
    r")\b"
)

# Non-cancer diseases that historically leaked via assay tokens.
_NONCANCER_DISEASE_RE = re.compile(
    r"(?i)\b("
    r"rheumatoid\s+arthritis|\barthritis\b|"
    r"diabetes|diabetic|metformin|"
    r"alzheimer|parkinson|"
    r"asthma|"
    r"multiple\s+sclerosis|\bms\b|"
    r"crohn|ulcerative\s+colitis|lupus|psoriasis|"
    r"hypertension|copd|"
    r"non[\s-]?cancer"
    r")\b"
)

# Ultra-short / catalog crumbs that must not solely drive recall.
WEAK_SOLO = frozenset(
    {
        "bound",
        "open",
        "non",
        "rate",
        "car",
        "rna",
        "seq",
        "clinical",
        "title",
        "link",
        "data",
        "dataset",
        "study",
        "public",
        "resource",
        "resources",
        "related",
        "find",
        "etc",
        "passwd",
        "please",
        "hart",
        "canary",
    }
)

_ASSAY_ONLY = frozenset(
    {
        "rna",
        "rna-seq",
        "rnaseq",
        "scrna",
        "scRNA".lower(),
        "bulk",
        "sequencing",
        "assay",
        "omics",
        "transcriptome",
        "expression",
    }
)


@dataclass(frozen=True)
class BoundaryDecision:
    abstain: bool
    reason: str | None
    message: str | None
    gate: str | None = None


def _norm(text: str) -> str:
    s = (text or "").lower()
    s = _NON.sub(" ", s)
    return _WS.sub(" ", s).strip()


def _tokens(text: str) -> list[str]:
    return [t for t in _norm(text).split(" ") if t]



def path_probe(goal: str) -> bool:
    return bool(_PATH_PROBE_RE.search(goal or ""))


def clinical_intent(goal: str) -> bool:
    return bool(_CLINICAL_RE.search(goal or ""))


def has_cancer_signal(goal: str) -> bool:
    # Strip explicit non-cancer phrasing so "non-cancer diabetes" is not cancer+.
    g = re.sub(r"(?i)\bnon[\s-]?cancer\b", " ", goal or "")
    return bool(_CANCER_RE.search(g))


def noncancer_disease(goal: str) -> bool:
    g = goal or ""
    if re.search(r"(?i)\bnon[\s-]?cancer\b", g):
        return True
    return bool(_NONCANCER_DISEASE_RE.search(g))


def weak_only_goal(goal: str) -> bool:
    toks = _tokens(goal)
    if not toks:
        return True
    content = [t for t in toks if t not in WEAK_SOLO and t not in _ASSAY_ONLY and len(t) > 2]
    # strip pure digits
    content = [t for t in content if not t.isdigit()]
    return len(content) == 0


def evaluate_goal_boundary(goal: str) -> BoundaryDecision:
    """Return abstain decision before ranking. Prefer empty over wrong."""
    g = (goal or "").strip()
    if not g:
        return BoundaryDecision(True, "empty_goal", WEAK_TOKEN_ABSTAIN, "empty")
    if clinical_intent(g):
        return BoundaryDecision(True, "clinical_intent", CLINICAL_ABSTAIN, "clinical")
    if path_probe(g) and not has_cancer_signal(g):
        return BoundaryDecision(True, "path_probe", WEAK_TOKEN_ABSTAIN, "weak")
    if noncancer_disease(g) and not has_cancer_signal(g):
        return BoundaryDecision(True, "noncancer_domain", OUT_OF_DOMAIN_ABSTAIN, "noncancer")
    if weak_only_goal(g):
        return BoundaryDecision(True, "weak_tokens", WEAK_TOKEN_ABSTAIN, "weak")
    return BoundaryDecision(False, None, None, None)

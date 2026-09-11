"""Hard safety gates: public metadata only; refuse everything else."""

from __future__ import annotations

import os
import re
from urllib.parse import urlparse

# Already-public, remotely callable metadata endpoints (one-click allow-list).
PUBLIC_METADATA_HOSTS = frozenset(
    {
        "eutils.ncbi.nlm.nih.gov",
        "www.ncbi.nlm.nih.gov",
        "api.figshare.com",
        "api.github.com",
        "clinicaltrials.gov",
        "www.cbioportal.org",
        "dockstore.org",
        "api.gdc.cancer.gov",
        "services.cancerimagingarchive.net",
        "nf-co.re",
        "www.ebi.ac.uk",
    }
)

DBGAP_RE = re.compile(r"\b(dbgap|phs\d{3,}|\bpha\d{6,})", re.I)
PHI_RE = re.compile(r"\b(phi|protected health information|hipaa|individual-level genotype)\b", re.I)
CONTROLLED_RE = re.compile(r"\b(controlled access|authorized access|data use limitation|duo:)\b", re.I)

OMICS_SUFFIXES = (
    ".h5ad",
    ".h5",
    ".hdf5",
    ".bam",
    ".fastq",
    ".fastq.gz",
    ".fq",
    ".fq.gz",
    ".mtx",
    ".mtx.gz",
    ".bigwig",
    ".bw",
    ".vcf",
    ".vcf.gz",
    ".dcm",
    ".dicom",
)

GS_RE = re.compile(r"^gs://", re.I)

# Path fragments that are never one-click, even on an allow-listed host.
REFUSED_PATH_TOKS = (
    "/dbgap/",
    "/gap/",
    "/sra/sra-instant/",
    "/slicing",
    "/getimage",
    "/getdicom",
    "/wado",
)


class SafetyError(ValueError):
    """Raised when a request is refused by policy."""


def offline() -> bool:
    return os.environ.get("COA_OFFLINE", "").strip() in {"1", "true", "yes", "on"}


def refuse_reason(text: str | None) -> str | None:
    blob = text or ""
    if DBGAP_RE.search(blob):
        return "refused: dbGaP / controlled-accession pattern"
    if PHI_RE.search(blob):
        return "refused: PHI / protected-health wording"
    if CONTROLLED_RE.search(blob) and "public" not in blob.lower():
        return "refused: controlled-access wording"
    return None


def assert_public_identifier(scheme: str, value: str) -> None:
    blob = f"{scheme}:{value}"
    reason = refuse_reason(blob)
    if reason:
        raise SafetyError(reason)
    if scheme == "gs_uri" or GS_RE.match(value):
        return  # pointer only; download is a separate check


def is_omics_filename(name: str) -> bool:
    lower = name.lower().split("?")[0]
    return any(lower.endswith(suf) for suf in OMICS_SUFFIXES)


def _host_path_allowed(host: str, path: str) -> bool:
    """Host-specific metadata-only path rules."""
    path_l = (path or "").lower()
    if host == "api.gdc.cancer.gov":
        p = path_l.rstrip("/")
        # Project catalog / status only. Never /data, /slicing, /files, /manifest.
        return p in {"/projects", "/status"} or p.startswith("/projects/")
    if host == "www.cbioportal.org":
        # Study list + study metadata + portal info. Never mutation/CNA matrices.
        if any(
            tok in path_l
            for tok in (
                "molecular-profile",
                "/mutations",
                "discrete-copy-number",
                "copy-number",
                "gene-panel-data",
                "/clinical-data",
            )
        ):
            return False
        return "/api/studies" in path_l or path_l.rstrip("/").endswith("/api/info")
    if host == "services.cancerimagingarchive.net":
        return "getcollectionvalues" in path_l
    if host == "dockstore.org":
        return "/api/ga4gh/trs/v2/tools" in path_l
    if host == "clinicaltrials.gov":
        return "/api/v2/studies" in path_l or path_l.rstrip("/").endswith("/api/v2")
    if host == "nf-co.re":
        return path_l.rstrip("/").endswith("/pipelines.json")
    if host == "www.ebi.ac.uk":
        return "/europepmc/webservices/rest/" in path_l
    return True


def one_click_allowed(url: str) -> bool:
    """True only for already-public remotely callable metadata tools."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return False
    host = (parsed.hostname or "").lower()
    if host not in PUBLIC_METADATA_HOSTS:
        return False
    path = parsed.path or ""
    path_l = path.lower()
    if any(tok in path_l for tok in REFUSED_PATH_TOKS):
        return False
    # GDC /data is a download endpoint (path == /data or /data/...).
    if host == "api.gdc.cancer.gov" and (
        path_l.rstrip("/") == "/data" or path_l.startswith("/data/")
    ):
        return False
    if is_omics_filename(path):
        return False
    if not _host_path_allowed(host, path):
        return False
    return True


def refuse_download(url_or_name: str) -> None:
    if is_omics_filename(url_or_name) or GS_RE.match(url_or_name):
        raise SafetyError(
            f"refused: never download omics / object-store payloads ({url_or_name!r})"
        )
    parsed = urlparse(url_or_name)
    if parsed.scheme in {"http", "https"} and not one_click_allowed(url_or_name):
        raise SafetyError(
            f"refused: one-click only for public metadata APIs ({url_or_name!r})"
        )


def user_agent() -> str:
    email = os.environ.get("NCBI_EMAIL", "").strip()
    base = os.environ.get("COA_USER_AGENT", "CancerOutputAtlas/0.1 (public metadata only)")
    return f"{base} {email}".strip()


"""Unattended atlas pipeline: fetch (or fixtures) → classify → link → emit."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cancer_output_atlas.classify import GOAL_DEFAULT, classify
from cancer_output_atlas.digest import render_digest
from cancer_output_atlas.link import link_outputs
from cancer_output_atlas.rank import rank_records
from cancer_output_atlas.safety import SafetyError, offline, refuse_reason
from cancer_output_atlas.node_features import apply_reuse_degree, persist_contract
from cancer_output_atlas.schema import (
    IDENTITY_SCHEMES,
    POLICY,
    AtlasGraph,
    Classification,
    DiseaseType,
    Identifier,
    OutputRecord,
    classification_row,
)
from cancer_output_atlas.sources.catalog import (
    BACKGROUND_GEO,
    DEFAULT_OFFLINE_BACKGROUND_GEO,
    POINTER_CARDS,
    SEEDS,
    Seed,
    fixtures_dir,
    seed_key,
)
from cancer_output_atlas.sources.figshare import article_id_from_doi, record_from_figshare
from cancer_output_atlas.sources.geo import record_from_esummary
from cancer_output_atlas.sources.http import FetchError
from cancer_output_atlas.sources.pointer import load_pointer

ISO = "%Y-%m-%dT%H:%M:%SZ"



def _disease_types_from_raw(raw: dict[str, Any]) -> list[DiseaseType]:
    items = raw.get("disease_types") or raw.get("cancer_types") or []
    out: list[DiseaseType] = []
    seen: set[tuple[str, str]] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        code = str(item.get("code") or "").strip()
        if not code:
            continue
        ontology = str(item.get("ontology") or "oncotree").strip() or "oncotree"
        key = (ontology, code)
        if key in seen:
            continue
        seen.add(key)
        out.append(
            DiseaseType(
                ontology=ontology,
                code=code,
                name=str(item.get("name") or code),
                source=str(item.get("source") or ""),
            )
        )
    return out


def records_from_json(payload: Any) -> list[OutputRecord]:
    if isinstance(payload, dict) and "nodes" in payload:
        payload = payload["nodes"]
    out: list[OutputRecord] = []
    for raw in payload:
        idents = [
            Identifier(i["scheme"], i["value"], i.get("source", "json"))
            for i in raw.get("identifiers") or []
        ]
        klass = None
        if raw.get("classification"):
            c = raw["classification"]
            klass = Classification(
                kind=c.get("kind") or raw.get("kind") or "dataset",
                labels=list(c.get("labels") or []),
                profile_fit=float(c.get("profile_fit") or 0.0),
                rationale=c.get("rationale") or "",
                matched_terms=list(c.get("matched_terms") or []),
            )
        rec = OutputRecord(
            output_id=raw["output_id"],
            kind=raw.get("kind") or "dataset",
            title=raw.get("title") or "",
            summary=raw.get("summary") or "",
            identifiers=idents,
            landing_url=raw.get("landing_url") or "",
            source_status=raw.get("source_status") or "fixture",
            skip_reason=raw.get("skip_reason"),
            license_hint=raw.get("license_hint"),
            organism=raw.get("organism"),
            biosystems=list(raw.get("biosystems") or []),
            assay=raw.get("assay"),
            perturbation=raw.get("perturbation"),
            modality=raw.get("modality"),
            n_samples=raw.get("n_samples"),
            classification=klass,
            reuse=dict(raw.get("reuse") or {}),
            evidence_urls=list(raw.get("evidence_urls") or []),
            disease_types=_disease_types_from_raw(raw),
            ods=raw.get("ods"),
            ods_zh=raw.get("ods_zh"),
            callability=raw.get("callability"),
            access=raw.get("access"),
            catalog=raw.get("catalog"),
            year=raw.get("year"),
            interventions=list(raw.get("interventions") or []),
            topics=list(raw.get("topics") or []),
            reuse_degree=int(raw.get("reuse_degree") or 0),
            link_ok=raw.get("link_ok"),
            checked_at=raw.get("checked_at"),
        )
        out.append(rec)
    return out


def _read_json(name: str) -> Any:
    return json.loads((fixtures_dir() / name).read_text(encoding="utf-8"))


def _geo_summary_fixture_paths() -> list[Path]:
    """All GEO series-summary fixtures, including expand harvests.

    Load order: background, main catalog, explicit expand, then any other
    ``geo_summaries_*.json`` in the fixtures root (not ``_partial/``).
    Later files overlay earlier keys.
    """
    fix = fixtures_dir()
    preferred = (
        "geo_summaries_background.json",
        "geo_summaries.json",
        "geo_summaries_expand.json",
    )
    seen: set[str] = set()
    paths: list[Path] = []
    for name in preferred:
        path = fix / name
        if path.is_file():
            paths.append(path)
            seen.add(name)
    for path in sorted(fix.glob("geo_summaries_*.json")):
        if path.name in seen:
            continue
        if path.name.endswith("_partial.json"):
            continue
        if path.is_file():
            paths.append(path)
            seen.add(path.name)
    # Quinn-owned GEO dump only (never treat quinn_catalog_notes / quinn_pointers as GEO).
    qpath = fix / "quinn_geo_summaries.json"
    if qpath.is_file() and qpath.name not in seen:
        paths.append(qpath)
        seen.add(qpath.name)
    return paths


def _load_geo_summaries() -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for path in _geo_summary_fixture_paths():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict):
            merged.update(data)
    return merged


def _load_geo_fixture(acc: str) -> OutputRecord:
    data = _load_geo_summaries()
    rec = data.get(acc)
    if not rec:
        raise FetchError(f"no GEO fixture for {acc}")
    return record_from_esummary(acc, rec, source="fixture")


def _load_figshare_fixture(doi: str) -> OutputRecord:
    article_id = article_id_from_doi(doi)
    path = fixtures_dir() / f"figshare_{article_id}.json"
    if not path.exists():
        raise FetchError(f"no Figshare fixture for article {article_id}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return record_from_figshare(doi, data, source="fixture")


def _load_nct_fixture(nct: str) -> OutputRecord:
    from cancer_output_atlas.sources.clinicaltrials import record_from_study

    acc = nct.strip().upper()
    for name in ("nct_nsclc_pembrolizumab.json", "nct_cancer_immunotherapy.json"):
        path = fixtures_dir() / name
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for study in payload.get("studies") or []:
            ident = ((study.get("protocolSection") or {}).get("identificationModule") or {})
            if str(ident.get("nctId") or "").upper() == acc:
                return record_from_study(study, source="fixture")
    raise FetchError(f"no NCT fixture for {acc}")


def _load_cbioportal_fixture(study_id: str) -> OutputRecord:
    from cancer_output_atlas.sources.cbioportal import record_from_study

    path = fixtures_dir() / f"cbioportal_{study_id}.json"
    if path.exists():
        return record_from_study(json.loads(path.read_text(encoding="utf-8")), source="fixture")
    catalog = _read_json("cbioportal_studies.json")
    for raw in catalog.get("studies") or []:
        if raw.get("studyId") == study_id:
            return record_from_study(raw, source="fixture")
    raise FetchError(f"no cBioPortal fixture for {study_id}")


def _load_dockstore_fixture(value: str) -> OutputRecord:
    from cancer_output_atlas.sources.dockstore import _normalize_trs_id, record_from_tool

    tid = _normalize_trs_id(value)
    path = fixtures_dir() / "dockstore_oncoanalyser.json"
    if path.exists():
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw.get("id") == tid or value.rsplit("/", 1)[-1] in str(raw.get("id") or ""):
            return record_from_tool(raw, source="fixture")
    catalog = _read_json("dockstore_tools.json")
    for raw in catalog.get("tools") or []:
        if raw.get("id") == tid or value.rsplit("/", 1)[-1] in str(raw.get("id") or ""):
            return record_from_tool(raw, source="fixture")
    raise FetchError(f"no Dockstore fixture for {value}")


def _load_github_fixture(full_name: str) -> OutputRecord:
    from cancer_output_atlas.sources.github import record_from_repo

    catalog = _read_json("github_repos.json")
    for raw in catalog.get("items") or []:
        if raw.get("full_name") == full_name:
            return record_from_repo(raw, source="fixture")
    slug = full_name.replace("/", "_").lower()
    path = fixtures_dir() / f"github_{slug}.json"
    if path.exists():
        return record_from_repo(json.loads(path.read_text(encoding="utf-8")), source="fixture")
    raise FetchError(f"no GitHub fixture for {full_name}")


def fetch_seed(seed: Seed, *, use_offline: bool) -> OutputRecord:
    blocked = refuse_reason(f"{seed.scheme}:{seed.value}")
    if blocked:
        raise SafetyError(blocked)
    if seed.fetcher == "pointer":
        return load_pointer(seed.value)
    if use_offline:
        if seed.fetcher == "geo":
            return _load_geo_fixture(seed.value)
        if seed.fetcher == "figshare":
            return _load_figshare_fixture(seed.value)
        if seed.fetcher == "clinicaltrials":
            return _load_nct_fixture(seed.value)
        if seed.fetcher == "cbioportal":
            return _load_cbioportal_fixture(seed.value)
        if seed.fetcher == "dockstore":
            return _load_dockstore_fixture(seed.value)
        if seed.fetcher == "github":
            return _load_github_fixture(seed.value)
        if seed.fetcher == "europepmc":
            from cancer_output_atlas.sources.europepmc import records_from_fixture

            recs = records_from_fixture(_read_json("europepmc_ppr1288154.json"), source="fixture")
            for rec in recs:
                if seed.value in rec.all_id_keys() or any(
                    seed.value in i.value for i in rec.identifiers
                ):
                    return rec
            if recs:
                return recs[0]
        raise FetchError(f"no offline fixture for {seed_key(seed)}")
    if seed.fetcher == "geo":
        from cancer_output_atlas.sources.geo import fetch_gse

        return fetch_gse(seed.value)
    if seed.fetcher == "figshare":
        from cancer_output_atlas.sources.figshare import fetch_figshare

        return fetch_figshare(seed.value)
    if seed.fetcher == "clinicaltrials":
        from cancer_output_atlas.sources.clinicaltrials import fetch_nct

        return fetch_nct(seed.value)
    if seed.fetcher == "cbioportal":
        from cancer_output_atlas.sources.cbioportal import fetch_study

        return fetch_study(seed.value)
    if seed.fetcher == "dockstore":
        from cancer_output_atlas.sources.dockstore import fetch_tool

        return fetch_tool(seed.value)
    if seed.fetcher == "github":
        from cancer_output_atlas.sources.github import fetch_repo

        return fetch_repo(seed.value)
    if seed.fetcher == "europepmc":
        from cancer_output_atlas.sources.europepmc import fetch_doi

        return fetch_doi(seed.value)
    raise FetchError(f"unknown fetcher {seed.fetcher}")


def _load_catalogs_offline() -> tuple[list[OutputRecord], list[dict[str, str]]]:
    from cancer_output_atlas.sources.cbioportal import records_from_studies_fixture
    from cancer_output_atlas.sources.clinicaltrials import records_from_query_fixture
    from cancer_output_atlas.sources.dockstore import records_from_tools_fixture
    from cancer_output_atlas.sources.europepmc import records_from_fixture as epmc_from_fixture
    from cancer_output_atlas.sources.gdc import records_from_projects_fixture
    from cancer_output_atlas.sources.github import records_from_repos_fixture
    from cancer_output_atlas.sources.nfcore import records_from_pipelines_fixture
    from cancer_output_atlas.sources.tcia import records_from_collections_fixture

    records: list[OutputRecord] = []
    skipped: list[dict[str, str]] = []

    def add_catalog(name: str, loader) -> None:
        try:
            records.extend(loader())
        except (OSError, KeyError, json.JSONDecodeError, FetchError) as exc:
            skipped.append({"seed": f"catalog:{name}", "reason": f"skipped: {exc}"})

    add_catalog(
        "cbioportal_studies",
        lambda: records_from_studies_fixture(_read_json("cbioportal_studies.json"), "fixture"),
    )
    for nct_file in ("nct_nsclc_pembrolizumab.json", "nct_cancer_immunotherapy.json"):
        add_catalog(
            nct_file,
            lambda n=nct_file: records_from_query_fixture(_read_json(n), "fixture"),
        )
    add_catalog(
        "gdc_projects",
        lambda: records_from_projects_fixture(_read_json("gdc_projects.json"), "fixture"),
    )
    add_catalog(
        "tcia_collections",
        lambda: records_from_collections_fixture(_read_json("tcia_collections.json"), "fixture"),
    )
    add_catalog(
        "nfcore_cancer",
        lambda: records_from_pipelines_fixture(_read_json("nfcore_cancer_pipelines.json"), "fixture"),
    )
    add_catalog(
        "dockstore_tools",
        lambda: records_from_tools_fixture(_read_json("dockstore_tools.json"), "fixture"),
    )
    add_catalog(
        "github_repos",
        lambda: records_from_repos_fixture(_read_json("github_repos.json"), "fixture"),
    )
    add_catalog(
        "europepmc_papers",
        lambda: epmc_from_fixture(_read_json("europepmc_ppr1288154.json"), "fixture"),
    )
    for epmc_name in sorted(p.name for p in fixtures_dir().glob("europepmc_*.json")):
        if epmc_name == "europepmc_ppr1288154.json":
            continue
        add_catalog(
            epmc_name,
            lambda n=epmc_name: epmc_from_fixture(_read_json(n), "fixture"),
        )

    def _geo_catalog() -> list[OutputRecord]:
        summaries = _load_geo_summaries()
        keep = {
            acc
            for acc in summaries
            if str(acc).upper().startswith("GSE")
            and (
                acc not in BACKGROUND_GEO
                or acc in DEFAULT_OFFLINE_BACKGROUND_GEO
            )
        }
        out = []
        for acc in sorted(keep):
            rec = summaries.get(acc)
            if rec:
                out.append(record_from_esummary(acc, rec, source="fixture"))
        return out

    add_catalog("geo_cancer_crispri", _geo_catalog)

    fig_path = fixtures_dir() / "figshare_32743536.json"
    if fig_path.exists():
        try:
            records.append(
                record_from_figshare(
                    "10.6084/m9.figshare.32743536.v1",
                    json.loads(fig_path.read_text(encoding="utf-8")),
                    source="fixture",
                )
            )
        except (FetchError, OSError, json.JSONDecodeError) as exc:
            skipped.append({"seed": "catalog:figshare_32743536", "reason": f"skipped: {exc}"})
    return records, skipped


def _load_catalogs_live() -> tuple[list[OutputRecord], list[dict[str, str]]]:
    records: list[OutputRecord] = []
    skipped: list[dict[str, str]] = []

    def add(name: str, fn) -> None:
        try:
            records.extend(fn())
        except (FetchError, SafetyError, OSError, KeyError) as exc:
            skipped.append({"seed": f"catalog:{name}", "reason": f"skipped: {exc}"})

    def nct_query(cond: str, intr: str | None = None) -> list[OutputRecord]:
        from cancer_output_atlas.sources.clinicaltrials import record_from_study
        from urllib.parse import urlencode

        from cancer_output_atlas.sources.http import get_json

        studies: list[OutputRecord] = []
        token = None
        while True:
            q = {
                "query.cond": cond,
                "pageSize": "1000",
                "countTotal": "true",
                "fields": "NCTId,BriefTitle,OfficialTitle,Condition,InterventionName,Phase,OverallStatus",
            }
            if intr:
                q["query.intr"] = intr
            if token:
                q["pageToken"] = token
            payload = get_json(
                "https://clinicaltrials.gov/api/v2/studies?" + urlencode(q),
                timeout=60.0,
            )
            for raw in payload.get("studies") or []:
                try:
                    studies.append(record_from_study(raw, source="clinicaltrials_api"))
                except FetchError:
                    continue
            token = payload.get("nextPageToken")
            if not token:
                break
        return studies

    add("cbioportal_studies", lambda: __import__(
        "cancer_output_atlas.sources.cbioportal", fromlist=["records_from_studies_fixture"]
    ).records_from_studies_fixture(
        {"studies": __import__("cancer_output_atlas.sources.http", fromlist=["get_json"]).get_json(
            "https://www.cbioportal.org/api/studies", timeout=60.0
        )},
        source="cbioportal_api",
    ))
    add("nct_nsclc_pembrolizumab", lambda: nct_query("non-small cell lung cancer", "pembrolizumab"))
    add("nct_cancer_immunotherapy", lambda: nct_query("cancer", "immunotherapy"))
    add("gdc_projects", lambda: __import__(
        "cancer_output_atlas.sources.gdc", fromlist=["fetch_projects"]
    ).fetch_projects())
    add("tcia_collections", lambda: __import__(
        "cancer_output_atlas.sources.tcia", fromlist=["fetch_collections"]
    ).fetch_collections())
    add("nfcore_cancer", lambda: __import__(
        "cancer_output_atlas.sources.nfcore", fromlist=["fetch_cancer_pipelines"]
    ).fetch_cancer_pipelines())
    add("dockstore_tools", lambda: __import__(
        "cancer_output_atlas.sources.dockstore", fromlist=["records_from_tools_fixture"]
    ).records_from_tools_fixture(
        {"tools": __import__("cancer_output_atlas.sources.http", fromlist=["get_json"]).get_json(
            "https://dockstore.org/api/ga4gh/trs/v2/tools?name=oncoanalyser", timeout=30.0
        )},
        source="dockstore_trs",
    ))
    add("github_repos", lambda: [
        __import__("cancer_output_atlas.sources.github", fromlist=["fetch_repo"]).fetch_repo(n)
        for n in ("cBioPortal/cbioportal", "nf-core/oncoanalyser")
    ])
    return records, skipped



def load_default_catalog_notes() -> list[OutputRecord]:
    """HTTPS catalog_notes only. gs:// pointers stay find-time / BACKGROUND_SEEDS."""
    from cancer_output_atlas.sources.pointer import load_all_catalog_notes

    out: list[OutputRecord] = []
    for rec in load_all_catalog_notes():
        if any(i.scheme == "gs_uri" or str(i.value).startswith("gs://") for i in rec.identifiers):
            continue
        if rec.output_id.startswith("dataset:gs_uri:") or (rec.landing_url or "").startswith("gs://"):
            continue
        out.append(rec)
    return out


def load_huggingface_cards() -> list[OutputRecord]:
    """Public HF dataset/model cards (landing + card text). Never lists parquet/h5ad files."""
    from cancer_output_atlas.sources.huggingface import records_from_fixture as hf_from_fixture

    out: list[OutputRecord] = []
    # Richer extra-harvest cards first so they win output_id / huggingface identity.
    extra = fixtures_dir() / "hf_virtual_cell.json"
    if extra.is_file():
        try:
            out.extend(hf_from_fixture(json.loads(extra.read_text(encoding="utf-8")), "fixture"))
        except (OSError, json.JSONDecodeError, KeyError):
            pass
    path = fixtures_dir() / "huggingface_virtual_cell_cards.json"
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        cards = payload.get("cards") if isinstance(payload, dict) else payload
        for raw in cards or []:
            if not isinstance(raw, dict):
                continue
            dsid = str(raw.get("id") or "").strip()
            url = str(raw.get("landing_url") or "").strip()
            if not dsid or not url.startswith("https://huggingface.co/"):
                continue
            if raw.get("private") or raw.get("disabled"):
                continue
            desc = str(raw.get("description") or "").strip()
            tags = [str(t) for t in (raw.get("tags") or [])]
            summary = desc
            if tags:
                summary = (summary + "\n" if summary else "") + "Keywords: " + ", ".join(tags)
            out.append(
                OutputRecord(
                    output_id=f"dataset:url:{url}",
                    kind="dataset",
                    title=dsid,
                    summary=summary,
                    identifiers=[Identifier("url", url, "huggingface_card")],
                    landing_url=url,
                    source_status="fixture",
                    license_hint=str(raw.get("license") or "Hugging Face public dataset card; matrices not downloaded"),
                    evidence_urls=[url],
                    reuse={
                        "public_metadata": True,
                        "matrices_downloaded": False,
                        "download": "forbidden",
                        "huggingface_card": True,
                    },
                    topics=list(tags),
                )
            )
    return out


def load_extra_portals() -> list[OutputRecord]:
    """CELLxGENE / HCA / scPerturb public-metadata fixtures from the extra harvest."""
    from cancer_output_atlas.sources.cellxgene import records_from_fixture as cxg_from_fixture
    from cancer_output_atlas.sources.hca import records_from_fixture as hca_from_fixture
    from cancer_output_atlas.sources.scperturb import records_from_fixture as scp_from_fixture

    out: list[OutputRecord] = []
    loaders = (
        ("cellxgene_cancer.json", cxg_from_fixture),
        ("hca_cancer.json", hca_from_fixture),
        ("scperturb.json", scp_from_fixture),
    )
    for name, loader in loaders:
        path = fixtures_dir() / name
        if not path.is_file():
            continue
        try:
            out.extend(loader(json.loads(path.read_text(encoding="utf-8")), "fixture"))
        except (OSError, json.JSONDecodeError, KeyError):
            continue
    return out


def load_quinn_extra_records() -> list[OutputRecord]:
    """Read Quinn-owned fixtures that are already OutputRecord JSON. Never writes them.

    GEO-shaped quinn_*.json dicts are merged in ``_load_geo_summaries`` and
    skipped here so the same GSE is not minted twice.
    """
    out: list[OutputRecord] = []
    for path in sorted(fixtures_dir().glob("quinn_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if path.name == "quinn_geo_summaries.json" or (
            isinstance(payload, dict) and any(str(k).upper().startswith("GSE") for k in payload)
            and "nodes" not in payload
            and "publications" not in payload
        ):
            continue
        if isinstance(payload, dict) and payload.get("publications"):
            from cancer_output_atlas.sources.europepmc import records_from_fixture as epmc_from_fixture
            try:
                out.extend(epmc_from_fixture({"results": payload["publications"]}, "fixture"))
            except (FetchError, KeyError, TypeError):
                pass
        if isinstance(payload, dict) and "nodes" in payload:
            out.extend(records_from_json(payload))
            continue
        if isinstance(payload, list) and payload and isinstance(payload[0], dict) and payload[0].get("output_id"):
            out.extend(records_from_json(payload))
            continue
        if isinstance(payload, dict) and payload.get("studies"):
            from cancer_output_atlas.sources.cbioportal import records_from_studies_fixture
            try:
                out.extend(records_from_studies_fixture(payload, "fixture"))
            except (FetchError, KeyError, TypeError):
                continue
    return out


def load_afh_harvest_records() -> list[OutputRecord]:
    """AFH harvest fixtures already shaped as OutputRecord JSON. Never writes them.

    Same pattern as ``load_quinn_extra_records`` / listing_harvest / catalog_notes:
    optional ``fixtures/afh_harvest_*.json`` with a ``nodes`` list. Missing is fine.
    """
    out: list[OutputRecord] = []
    for path in sorted(fixtures_dir().glob("afh_harvest_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and "nodes" in payload:
            out.extend(records_from_json(payload))
        elif (
            isinstance(payload, list)
            and payload
            and isinstance(payload[0], dict)
            and payload[0].get("output_id")
        ):
            out.extend(records_from_json(payload))
    return out


def load_listing_sites() -> list[OutputRecord]:
    """Optional harvest file from another worker. Missing is fine."""
    candidates = [
        fixtures_dir() / "listing-sites-verified.json",
        Path("out/listing-sites-verified.json"),
        Path(__file__).resolve().parents[2] / "out" / "listing-sites-verified.json",
    ]
    path = next((c for c in candidates if c.is_file()), None)
    if path is None:
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    raw_items = payload
    if isinstance(payload, dict):
        raw_items = payload.get("sites") or payload.get("urls") or payload.get("nodes") or []
    out: list[OutputRecord] = []
    for raw in raw_items or []:
        if isinstance(raw, str):
            url, title = raw, raw
        elif isinstance(raw, dict):
            url = str(raw.get("landing_url") or raw.get("url") or raw.get("value") or "").strip()
            title = str(raw.get("title") or raw.get("name") or url)
            summary = str(raw.get("summary") or raw.get("blurb") or "Verified listing-site landing (pointer)")
            access = str(raw.get("access") or "").strip().lower()
            apply_yourself = access in {"user_applies", "controlled", "apply", "apply_yourself"}
        else:
            continue
        if not url.startswith(("http://", "https://")):
            continue
        if isinstance(raw, str):
            summary = "Verified listing-site landing (pointer)"
            apply_yourself = False
        out.append(
            OutputRecord(
                output_id=f"dataset:url:{url}",
                kind=str(raw.get("kind") or "dataset") if isinstance(raw, dict) else "dataset",
                title=title,
                summary=summary,
                identifiers=[Identifier("url", url, "listing_sites_verified")],
                landing_url=url,
                source_status="pointer",
                license_hint="listing-site harvest; pointer only",
                evidence_urls=[url],
                reuse={
                    "public_metadata": True,
                    "matrices_downloaded": False,
                    "download": "forbidden",
                    "listing_site": True,
                    "apply_yourself": apply_yourself,
                },
            )
        )
    return out


def _identity_keys(rec: OutputRecord) -> set[str]:
    return {i.key() for i in rec.identifiers if i.scheme in IDENTITY_SCHEMES}


def dedup_records(records: list[OutputRecord]) -> list[OutputRecord]:
    seen_ids: set[str] = set()
    seen_keys: set[str] = set()
    out: list[OutputRecord] = []
    for rec in records:
        if rec.output_id in seen_ids:
            continue
        keys = _identity_keys(rec)
        if keys & seen_keys:
            continue
        seen_ids.add(rec.output_id)
        seen_keys |= keys
        out.append(rec)
    return out


def ingest(
    seeds: list[Seed] | None = None,
    *,
    use_offline: bool | None = None,
    include_catalogs: bool | None = None,
    include_pointer_cards: bool | None = None,
) -> tuple[list[OutputRecord], list[dict[str, str]]]:
    use_offline = offline() if use_offline is None else use_offline
    records: list[OutputRecord] = []
    skipped: list[dict[str, str]] = []

    default_run = seeds is None
    seed_list = list(SEEDS) if seeds is None else list(seeds)
    if include_catalogs is None:
        include_catalogs = default_run
    if include_pointer_cards is None:
        include_pointer_cards = default_run

    if include_catalogs:
        if use_offline:
            cat, skip = _load_catalogs_offline()
        else:
            cat, skip = _load_catalogs_live()
        records.extend(cat)
        skipped.extend(skip)

    for seed in seed_list:
        try:
            rec = fetch_seed(seed, use_offline=use_offline)
            records.append(rec)
        except (FetchError, SafetyError, OSError, KeyError, json.JSONDecodeError) as exc:
            skipped.append({"seed": seed_key(seed), "reason": f"skipped: {exc}"})

    if include_pointer_cards:
        for seed in POINTER_CARDS:
            try:
                rec = fetch_seed(seed, use_offline=use_offline)
                records.append(rec)
            except (FetchError, SafetyError, OSError, KeyError, json.JSONDecodeError) as exc:
                skipped.append({"seed": seed_key(seed), "reason": f"skipped: {exc}"})

    if include_catalogs or include_pointer_cards:
        try:
            records.extend(load_listing_sites())
        except (OSError, json.JSONDecodeError, KeyError) as exc:
            skipped.append({"seed": "catalog:listing_sites", "reason": f"skipped: {exc}"})
        try:
            records.extend(load_default_catalog_notes())
        except (OSError, json.JSONDecodeError, KeyError, FetchError) as exc:
            skipped.append({"seed": "catalog:catalog_notes", "reason": f"skipped: {exc}"})
        try:
            records.extend(load_huggingface_cards())
        except (OSError, json.JSONDecodeError, KeyError) as exc:
            skipped.append({"seed": "catalog:huggingface_cards", "reason": f"skipped: {exc}"})
        try:
            records.extend(load_extra_portals())
        except (OSError, json.JSONDecodeError, KeyError) as exc:
            skipped.append({"seed": "catalog:extra_portals", "reason": f"skipped: {exc}"})
        try:
            records.extend(load_quinn_extra_records())
        except (OSError, json.JSONDecodeError, KeyError) as exc:
            skipped.append({"seed": "catalog:quinn", "reason": f"skipped: {exc}"})
        try:
            records.extend(load_afh_harvest_records())
        except (OSError, json.JSONDecodeError, KeyError) as exc:
            skipped.append({"seed": "catalog:afh_harvest", "reason": f"skipped: {exc}"})

    return dedup_records(records), skipped


def run_atlas(
    goal: str = GOAL_DEFAULT,
    *,
    use_offline: bool | None = None,
    seeds: list[Seed] | None = None,
    include_catalogs: bool | None = None,
    check_live: bool | None = None,
) -> tuple[AtlasGraph, list[dict[str, Any]]]:
    records, skipped = ingest(
        seeds, use_offline=use_offline, include_catalogs=include_catalogs
    )
    for rec in records:
        classify(rec, goal=goal)
        persist_contract(rec)
    links = link_outputs(records)
    apply_reuse_degree(records, links)
    now = datetime.now(timezone.utc).strftime(ISO)
    graph = AtlasGraph(
        goal=goal,
        generated_at=now,
        nodes=records,
        links=links,
        skipped=skipped,
        policy=dict(POLICY),
    )
    offline_run = offline() if use_offline is None else use_offline
    if check_live is None:
        check_live = not offline_run
    if check_live:
        from cancer_output_atlas.check_links import check_graph_links

        graph, _ = check_graph_links(graph, skip_if_offline=False)
    ranked = rank_records(graph.nodes, goal=goal, query_first=False)
    return graph, ranked


def write_artifacts(graph: AtlasGraph, ranked: list[dict[str, Any]], out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    table_rows = [classification_row(n) for n in graph.nodes if n.source_status != "skipped"]
    paths = {
        "classification_table": out_dir / "classification_table.json",
        "classification_csv": out_dir / "classification_table.csv",
        "link_graph": out_dir / "link_graph.json",
        "digest": out_dir / "digest.md",
        "ranked": out_dir / "ranked.json",
    }
    paths["classification_table"].write_text(json.dumps(table_rows, indent=2) + "\n", encoding="utf-8")
    paths["link_graph"].write_text(json.dumps(graph.to_json(), indent=2) + "\n", encoding="utf-8")
    paths["ranked"].write_text(json.dumps(ranked, indent=2) + "\n", encoding="utf-8")
    paths["digest"].write_text(render_digest(graph, ranked), encoding="utf-8")
    if table_rows:
        fieldnames = list(table_rows[0].keys())
        with paths["classification_csv"].open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in table_rows:
                flat = dict(row)
                for key in ("labels", "biosystems", "ids"):
                    if isinstance(flat.get(key), list):
                        flat[key] = "|".join(str(x) for x in flat[key])
                writer.writerow(flat)
    else:
        paths["classification_csv"].write_text("output_id,kind,title\n", encoding="utf-8")
    from cancer_output_atlas.oncotree import write_cancer_types_table

    extra = write_cancer_types_table(graph.nodes, out_dir)
    paths.update({f"cancer_types_{k}": v for k, v in extra.items()})
    return paths

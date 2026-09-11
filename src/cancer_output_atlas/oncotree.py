"""Official OncoTree lookup. Tag only from source fields; never invent codes."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from cancer_output_atlas.schema import DiseaseType, OutputRecord

ONCOTREE_FILE = "oncotree.json"
SOURCE_MAP_FILE = "oncotree_source_map.json"

# GEO esummary keys that would be a structured cancer type if a catalog ever
# ships them. Title / summary are intentionally absent.
GEO_STRUCTURED_TYPE_KEYS = (
    "cancertype",
    "cancer_type",
    "cancerType",
    "cancerTypeId",
    "oncotree",
    "oncotree_code",
    "oncotreeCode",
)


def _fixtures_dir() -> Path:
    from cancer_output_atlas.sources.catalog import fixtures_dir

    return fixtures_dir()


def _norm(value: str) -> str:
    return " ".join((value or "").split()).casefold()


@lru_cache(maxsize=1)
def load_oncotree() -> dict[str, dict]:
    """code → official {code, name, parent, mainType, tissue, level}."""
    raw = json.loads((_fixtures_dir() / ONCOTREE_FILE).read_text(encoding="utf-8"))
    items = raw.get("tumorTypes") if isinstance(raw, dict) else raw
    out: dict[str, dict] = {}
    for item in items or []:
        code = str(item.get("code") or "").strip()
        if not code:
            continue
        out[code] = {
            "code": code,
            "name": str(item.get("name") or code),
            "parent": item.get("parent"),
            "mainType": item.get("mainType"),
            "tissue": item.get("tissue"),
            "level": item.get("level"),
        }
    return out


@lru_cache(maxsize=1)
def load_source_map() -> dict[str, str]:
    """normalized source_value → official OncoTree code."""
    raw = json.loads((_fixtures_dir() / SOURCE_MAP_FILE).read_text(encoding="utf-8"))
    tree = load_oncotree()
    out: dict[str, str] = {}
    for pair in raw.get("pairs") or []:
        value = str(pair.get("source_value") or "").strip()
        code = str(pair.get("code") or "").strip()
        if not value or code not in tree:
            continue
        out[_norm(value)] = code
    return out


def lookup_code(source_value: str) -> str | None:
    """Exact table lookup only. Unknown / invented values → None."""
    key = _norm(source_value)
    if not key:
        return None
    return load_source_map().get(key)


def types_from_source(pairs: Iterable[tuple[str | None, str]]) -> list[DiseaseType]:
    """Map (source_value, source_field) pairs. Dedupe by official code."""
    tree = load_oncotree()
    seen: set[str] = set()
    out: list[DiseaseType] = []
    for raw, field in pairs:
        if raw is None:
            continue
        value = str(raw).strip()
        if not value:
            continue
        code = lookup_code(value)
        if code is None or code in seen or code not in tree:
            continue
        seen.add(code)
        out.append(DiseaseType(ontology="oncotree", code=code, name=tree[code]["name"], source=field))
    return out


def geo_structured_types(payload: dict) -> list[DiseaseType]:
    """GEO: only a structured type key. Never title leftover tokens."""
    pairs: list[tuple[str | None, str]] = []
    for key in GEO_STRUCTURED_TYPE_KEYS:
        if key in payload and payload[key] not in (None, ""):
            pairs.append((str(payload[key]), f"geo.{key}"))
    return types_from_source(pairs)


def cancer_type_rows(records: list[OutputRecord]) -> list[dict]:
    """Human table: official code, name, parent, node count."""
    tree = load_oncotree()
    counts: dict[str, int] = {}
    for rec in records:
        codes = {d.code for d in rec.disease_types if d.ontology == "oncotree" and d.code}
        for code in codes:
            counts[code] = counts.get(code, 0) + 1
    rows = []
    for code, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        info = tree.get(code) or {}
        rows.append(
            {
                "code": code,
                "name": info.get("name") or code,
                "parent": info.get("parent"),
                "node_count": n,
            }
        )
    return rows


def write_cancer_types_table(records: list[OutputRecord], out_dir: Path) -> dict[str, Path]:
    import csv

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = cancer_type_rows(records)
    csv_path = out_dir / "cancer_types.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["code", "name", "parent", "node_count"])
        writer.writeheader()
        writer.writerows(rows)
    paths = {"csv": csv_path}
    xlsx_path = out_dir / "cancer_types.xlsx"
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font

        wb = Workbook()
        ws = wb.active
        ws.title = "cancer_types"
        ws.append(["oncotree_code", "name", "parent", "node_count"])
        for cell in ws[1]:
            cell.font = Font(bold=True)
        for row in rows:
            ws.append([row["code"], row["name"], row["parent"], row["node_count"]])
        ws.column_dimensions["A"].width = 16
        ws.column_dimensions["B"].width = 44
        ws.column_dimensions["C"].width = 18
        wb.save(xlsx_path)
        paths["xlsx"] = xlsx_path
    except ImportError:
        pass
    return paths


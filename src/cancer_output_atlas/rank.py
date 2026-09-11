"""Rank classified outputs by a reuse goal.

Matching parses the meaning of the goal (topic / type / filler), not leftover
2-grams. A plausible-but-wrong hit is worse than an empty result.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from cancer_output_atlas.classify import GOAL_DEFAULT
from cancer_output_atlas.ods import ods_category
from cancer_output_atlas.schema import OutputRecord, classification_row

_LATIN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_CJK_RUN = re.compile(r"[\u4e00-\u9fff]+")
_NON_ALNUM = re.compile(r"[^a-z0-9\u4e00-\u9fff]+")
_WS = re.compile(r"\s+")
_ACCESSION = re.compile(
    r"^(?:gse|gsm|gpl|nct|pmc|pmid|phs|srp|srx|srr|prjna|e-mtab|e-geod)\d+$",
    re.I,
)

STOPWORDS = frozenset(
    {
        "find",
        "public",
        "resources",
        "resource",
        "i",
        "can",
        "the",
        "a",
        "an",
        "of",
        "and",
        "or",
        "to",
        "for",
        "with",
        "in",
        "on",
        "reuse",
        "use",
        "used",
        "using",
        "找",
        "的",
        "和",
        "与",
        "资源",
        "公开",
        "可以",
        "我",
        "要",
        "做",
        "能用",
        "相关",
        "no",
        "such",
        "yes",
        "this",
        "that",
        "not",
        "any",
        "all",
        "from",
        "by",
        "as",
        "at",
        "it",
        "we",
        "you",
        "they",
        "is",
        "are",
        "was",
        "be",
        "been",
        "its",
        "if",
        "then",
        "into",
        "via",
        "per",
        "vs",
        "versus",
        "based",
        "types",
        "type",
        "output",
        "outputs",
        "results",
        "result",
    }
)

FILLER = STOPWORDS | frozenset(
    {
        "related",
        "relating",
        "regarding",
        "about",
        "looking",
        "please",
        "want",
        "need",
        "show",
        "get",
        "reusable",
        "datasets",
        "stuff",
        "things",
        "有关",
        "一下",
        "帮我",
    }
)

GENERIC_ALONE = frozenset(
    {
        "cell",
        "cells",
        "cancer",
        "tumor",
        "study",
        "data",
        "dataset",
        "human",
        "analysis",
        "gene",
        "expression",
        "细胞",
        "癌症",
        "肿瘤",
        "数据",
        "研究",
        "基因",
        "表达",
    }
)

# Meta / catalog crumbs that match half the graph if allowed as sole topics.
# Prefer abstain over "disease" / "doi" soup hits.
WEAK_ALONE = frozenset(
    {
        "disease",
        "diseases",
        "doi",
        "dois",
        "pmid",
        "pmcid",
        "pmc",
        "fake",
        "demo",
        "test",
        "tests",
        "sample",
        "samples",
        "paper",
        "papers",
        "article",
        "articles",
        "publication",
        "publications",
        "abstract",
        "title",
        "link",
        "links",
        "url",
        "http",
        "https",
        "www",
        "org",
        "com",
        "omics",
        "meta",
        "unknown",
        "null",
        "none",
        "hart",  # Hart canary org crumb in fake DOIs
        "bound",
        "open",
        "non",
        "rate",
        "car",
        "clinical",
        "arthritis",
        "diabetes",
        "asthma",
        "alzheimer",
        "metformin",
        "etc",
        "passwd",
    }
)

_DOI_RE = re.compile(r"(?i)\b(10\.\d{4,9}/[^\s]+)")

TYPE_MAP: dict[str, str] = {
    "dataset": "data",
    "datasets": "data",
    "data": "data",
    "geo": "data",
    "表达谱": "data",
    "数据": "data",
    "数据集": "data",
    "trial": "trial_result",
    "trials": "trial_result",
    "nct": "trial_result",
    "试验": "trial_result",
    "试验结果": "trial_result",
    "software": "software",
    "repo": "software",
    "repos": "software",
    "软件": "software",
    "tool": "tool",
    "tools": "tool",
    "portal": "tool",
    "工具": "tool",
    "workflow": "method",
    "workflows": "method",
    "pipeline": "method",
    "pipelines": "method",
    "method": "method",
    "methods": "method",
    "protocol": "method",
    "protocols": "method",
    "方法": "method",
    "model": "model",
    "models": "model",
    "模型": "model",
    "biospecimen": "biospecimen",
    "biospecimens": "biospecimen",
    "样本": "biospecimen",
}


# Assay / modality phrases kept as a required facet (AND with disease/drug).
_ASSAY_PHRASE_RES = (
    re.compile(r"(?i)\b(single[\s-]?cell\s+rna[\s-]?seq|scrna[\s-]?seq|scRNA-seq)\b"),
    re.compile(r"(?i)\b(bulk\s+rna[\s-]?seq)\b"),
    re.compile(r"(?i)\b(rna[\s-]?seq|rnaseq|rna\s+sequencing)\b"),
    re.compile(r"(?i)\b(perturb[\s-]?seq|crop[\s-]?seq)\b"),
    re.compile(r"(?i)\b(whole[\s-]?exome|wes|wgs|chip[\s-]?seq|atac[\s-]?seq)\b"),
)

# Imaging-only crumbs: drop when an RNA/seq assay facet is required.
_IMAGING_ONLY_RE = re.compile(
    r"(?i)\b(tcia|dicom|dbt|digital\s+breast\s+tomosynthesis|mammograph|radiolog|"
    r"imaging\s+archive|pixel\s+data|mri|ct\s+scan|pet[\s-]?ct)\b"
)

# Drug / therapy facet families: within a family, any synonym satisfies the facet.
_DRUG_FACET_GROUPS: tuple[frozenset[str], ...] = (
    frozenset({"pembrolizumab", "keytruda", "mk-3475", "mk3475"}),
    frozenset({"nivolumab", "opdivo"}),
    frozenset({"immunotherapy", "immuno", "checkpoint", "pd-1", "pd1", "pd-l1", "pdl1"}),
)

# Each disease facet: triggers must appear in the RAW goal; match synonyms gate records.
# Keep triggers narrow so synonym expansion / substring aliases cannot invent extra AND facets.
_DISEASE_FACET_SPECS: tuple[tuple[str, frozenset[str], frozenset[str]], ...] = (
    (
        "breast cancer",
        frozenset({"breast", "breastcancer", "breast cancer"}),
        frozenset({"breast", "breastcancer", "breast cancer"}),
    ),
    (
        "lung cancer",
        frozenset({
            "lung cancer",
            "lungcancer",
            "lung adenocarcinoma",
            "lungadenocarcinoma",
            "lung squamous",
            "lung squamous cell",
            "sclc",
            "small cell lung cancer",
        }),
        frozenset({
            "lung cancer",
            "lungcancer",
            "lung adenocarcinoma",
            "lungadenocarcinoma",
            "lung squamous",
            "lung squamous cell",
            "sclc",
            "small cell lung",
            "small cell lung cancer",
        }),
    ),
    (
        "NSCLC",
        frozenset({
            "nsclc",
            "non small cell lung",
            "non-small cell lung",
            "nonsmallcelllung",
            "luad",
            "lusc",
        }),
        frozenset({
            "nsclc",
            "non small cell lung",
            "non-small cell lung",
            "nonsmallcelllung",
            "luad",
            "lusc",
            "lung adenocarcinoma",
            "lungadenocarcinoma",
            "lung squamous",
            "lung squamous cell",
        }),
    ),
    (
        "melanoma",
        frozenset({"melanoma"}),
        frozenset({"melanoma"}),
    ),
    (
        "ovarian cancer",
        frozenset({"ovarian", "ovarian cancer", "ovariancancer"}),
        frozenset({"ovarian", "ovarian cancer", "ovariancancer"}),
    ),
    (
        "prostate cancer",
        frozenset({"prostate", "prostate cancer", "prostatecancer"}),
        frozenset({"prostate", "prostate cancer", "prostatecancer"}),
    ),
    (
        "colorectal cancer",
        frozenset({"colorectal", "crc", "colorectal cancer", "colorectalcancer"}),
        frozenset({"colorectal", "crc", "colorectal cancer", "colorectalcancer"}),
    ),
    (
        "pancreatic cancer",
        frozenset({"pancreatic", "pancreatic cancer", "pancreaticcancer"}),
        frozenset({"pancreatic", "pancreatic cancer", "pancreaticcancer"}),
    ),
)

# Back-compat alias for any leftover imports/tests.
_DISEASE_FACET_GROUPS: tuple[frozenset[str], ...] = tuple(spec[2] for spec in _DISEASE_FACET_SPECS)

DISTINCTIVE = frozenset(
    {
        "pembrolizumab",
        "keytruda",
        "nsclc",
        "crispri",
        "crispra",
        "virtualcell",
        "immunotherapy",
        "immuno",
        "keynote",
    }
)

_PHRASE_EXPAND: dict[str, tuple[str, ...]] = {
    "virtual cell": ("虚拟细胞", "virtualcell", "virtual-cell"),
    "virtualcell": ("virtual cell", "虚拟细胞"),
    "virtual-cell": ("virtual cell", "虚拟细胞", "virtualcell"),
    "虚拟细胞": ("virtual cell", "virtualcell", "virtual-cell"),
    "virtual cell challenge": ("虚拟细胞", "virtual cell"),
    "nsclc": (
        "non-small cell lung",
        "non small cell lung",
        "non-small-cell lung",
        "lung adenocarcinoma",
    ),
    "pembrolizumab": ("keytruda",),
    "keytruda": ("pembrolizumab",),
    "crispri in cancer": ("crispri",),
    "crispri-in-cancer": ("crispri",),
    "immuno oncology": ("immuno-oncology", "immunotherapy"),
    "immuno-oncology": ("immuno oncology", "immunotherapy"),
}

_TOKEN_EXPAND: dict[str, tuple[str, ...]] = {
    "virtual": ("虚拟",),
    "虚拟": ("virtual",),
    "pembrolizumab": ("keytruda",),
    "keytruda": ("pembrolizumab",),
    "nsclc": ("nsclc",),
}

_KNOWN_TOPIC_PHRASES = tuple(
    sorted(
        {
            "virtual cell challenge",
            "virtual cell",
            "virtual-cell",
            "virtualcell",
            "虚拟细胞",
            "non-small cell lung cancer",
            "non-small-cell lung cancer",
            "non small cell lung cancer",
            "non-small cell lung",
            "non-small-cell lung",
            "non small cell lung",
            "lung adenocarcinoma",
            "immuno-oncology",
            "immuno oncology",
            "immune checkpoint",
            "crispri-in-cancer",
            "crispri in cancer",
            "pembrolizumab",
            "keytruda",
            "nsclc",
            *_PHRASE_EXPAND.keys(),
        },
        key=lambda s: (-len(s), s),
    )
)

_CJK_FILLER_SORTED = tuple(
    sorted(
        (s for s in (FILLER | frozenset(TYPE_MAP)) if _CJK_RUN.fullmatch(s)),
        key=len,
        reverse=True,
    )
)

JUNK_PHRASES = frozenset(
    {
        "cell related",
        "related dataset",
        "cell dataset",
        "related data",
        "cell data",
    }
)


@dataclass(frozen=True)
class GoalQuery:
    raw: str
    phrases: tuple[str, ...]
    content_tokens: tuple[str, ...]
    non_generic: frozenset[str]
    generic_paired: frozenset[str]
    core_non_generic: frozenset[str]
    type_filter: tuple[str, ...]
    must_not: tuple[str, ...]
    topic_text: str
    empty: bool
    source: str
    # AND facets: each inner tuple is OR-synonyms; every group must match the record.
    required_facets: tuple[tuple[str, ...], ...] = ()


def _norm(text: str) -> str:
    s = (text or "").lower().replace("\u2014", " ").replace("\u2013", " ")
    s = re.sub(r"[-_/.,;:()\[\]|+]+", " ", s)
    return _WS.sub(" ", s).strip()


def _compact(text: str) -> str:
    return _NON_ALNUM.sub("", (text or "").lower())


def _strip_cjk_filler(text: str) -> str:
    out = text
    for stop in _CJK_FILLER_SORTED:
        out = out.replace(stop, " ")
    return out


def _latin_tokens(text: str) -> list[str]:
    return _LATIN.findall(text.lower())


def _is_filler_or_type(tok: str) -> bool:
    return tok in FILLER or tok in TYPE_MAP


def _looks_gibberish(tok: str) -> bool:
    """Canary / keyboard-smash unigrams (e.g. zzzxqwvutsrzyxqwvutsr)."""
    if not tok or _CJK_RUN.fullmatch(tok) or _ACCESSION.match(tok):
        return False
    letters = [c for c in tok.lower() if c.isalpha()]
    if len(letters) < 10:
        return False
    vowels = sum(1 for c in letters if c in "aeiou")
    ratio = vowels / len(letters)
    if ratio < 0.22:
        return True
    if len(letters) >= 14 and ratio < 0.28:
        return True
    return False


def _is_weak_alone(tok: str) -> bool:
    if not tok:
        return True
    if tok in GENERIC_ALONE or tok in WEAK_ALONE:
        return True
    if tok.isdigit():
        return True
    if _looks_gibberish(tok):
        return True
    return False


def _is_distinctive(tok: str) -> bool:
    if not tok or _is_filler_or_type(tok) or tok in GENERIC_ALONE or tok in WEAK_ALONE:
        return False
    if tok.isdigit():
        return False
    if _looks_gibberish(tok):
        return False
    if tok in DISTINCTIVE:
        return True
    if _ACCESSION.match(tok):
        return True
    if _CJK_RUN.fullmatch(tok) and len(tok) >= 2:
        return True
    return len(tok) >= 3


def _phrase_usable(phrase: str) -> bool:
    """Drop junk / weak-only / gibberish-paired phrases before ranking."""
    raw = (phrase or "").strip()
    if raw and (_DOI_RE.search(raw) or raw.lower().startswith("10.")):
        return True
    nph = _norm(phrase)
    if not nph or nph in JUNK_PHRASES:
        return False
    toks = nph.split()
    if not toks:
        return False
    if len(toks) == 1:
        return _is_distinctive(toks[0])
    if any(_looks_gibberish(t) for t in toks):
        return False
    if all(_is_weak_alone(t) or _is_filler_or_type(t) for t in toks):
        return False
    return any(_is_distinctive(t) for t in toks)



def _goal_tokens(text: str) -> list[str]:
    pieces: list[str] = []
    for m in re.finditer(r"[a-z0-9]+(?:-[a-z0-9]+)*|[\u4e00-\u9fff]+", text):
        piece = m.group(0)
        if _is_filler_or_type(piece):
            continue
        if "-" in piece:
            parts = [p for p in piece.split("-") if p and not _is_filler_or_type(p)]
            pieces.append(piece)
            pieces.extend(parts)
        else:
            pieces.append(piece)
    out: list[str] = []
    seen: set[str] = set()
    for p in pieces:
        if p in seen or _is_filler_or_type(p):
            continue
        if len(p) < 3 and p not in {"io"} and not _CJK_RUN.fullmatch(p):
            continue
        if _CJK_RUN.fullmatch(p) and len(p) < 2:
            continue
        seen.add(p)
        out.append(p)
    return out


def _latin_boundary_ok(text: str, start: int, end: int) -> bool:
    if start > 0 and text[start - 1].isalnum():
        return False
    if end < len(text) and text[end].isalnum():
        return False
    return True


def _extract_known_phrases(normed: str) -> tuple[list[str], str]:
    if not normed:
        return [], ""
    used = [False] * len(normed)
    hits: list[str] = []
    for phrase in _KNOWN_TOPIC_PHRASES:
        nph = _norm(phrase)
        if not nph:
            continue
        start = 0
        while True:
            idx = normed.find(nph, start)
            if idx < 0:
                break
            end = idx + len(nph)
            if any(c.isascii() and c.isalnum() for c in nph):
                if not _latin_boundary_ok(normed, idx, end):
                    start = idx + 1
                    continue
            if any(used[idx:end]):
                start = idx + 1
                continue
            for i in range(idx, end):
                used[i] = True
            hits.append(nph)
            start = end
    leftover = "".join(ch if not used[i] else " " for i, ch in enumerate(normed))
    leftover = _WS.sub(" ", leftover).strip()
    return hits, leftover


def _type_filter_from_text(normed: str, raw_lower: str) -> tuple[str, ...]:
    families: list[str] = []
    seen: set[str] = set()
    tokens = _latin_tokens(normed)
    for tok in tokens:
        fam = TYPE_MAP.get(tok)
        if fam and fam not in seen:
            seen.add(fam)
            families.append(fam)
    blob = raw_lower + " " + normed
    for word, fam in TYPE_MAP.items():
        if _CJK_RUN.fullmatch(word) and word in blob and fam not in seen:
            seen.add(fam)
            families.append(fam)
    if len(families) == 1:
        return (families[0],)
    return ()


def _expand_phrases(phrases: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()

    def add(p: str) -> None:
        p = _norm(p)
        if not p or p in seen or p in JUNK_PHRASES:
            return
        if " " not in p and p in GENERIC_ALONE:
            return
        if " " not in p and p in TYPE_MAP:
            return
        if " " not in p and p in FILLER:
            return
        if any(t in FILLER or t in TYPE_MAP for t in p.split()) and p not in _KNOWN_TOPIC_PHRASES:
            # allow known "virtual cell"; reject "cell related"
            if p not in { _norm(k) for k in _KNOWN_TOPIC_PHRASES }:
                return
        if not _phrase_usable(p) and p not in { _norm(k) for k in _KNOWN_TOPIC_PHRASES }:
            return
        seen.add(p)
        out.append(p)
        compact = _compact(p)
        if compact and compact != p and compact not in seen and len(compact) >= 6:
            seen.add(compact)
            out.append(compact)

    for p in phrases:
        add(p)
        for extra in _PHRASE_EXPAND.get(p, ()):
            add(extra)
        for extra in _PHRASE_EXPAND.get(_norm(p), ()):
            add(extra)
    return out


def _build_query(
    raw: str,
    phrases: list[str],
    leftover_tokens: list[str],
    type_filter: tuple[str, ...],
    must_not: tuple[str, ...],
    source: str,
) -> GoalQuery:
    phrases = _expand_phrases(phrases)
    content: list[str] = []
    seen_c: set[str] = set()
    for tok in leftover_tokens:
        for item in (tok, *_TOKEN_EXPAND.get(tok, ())):
            if item in FILLER or item in TYPE_MAP or item in seen_c:
                continue
            seen_c.add(item)
            content.append(item)
    for ph in phrases:
        for tok in _latin_tokens(ph) + _CJK_RUN.findall(ph):
            if tok in FILLER or tok in TYPE_MAP or tok in seen_c:
                continue
            seen_c.add(tok)
            content.append(tok)
    expanded: list[str] = []
    seen_t: set[str] = set()
    for tok in content:
        for item in (tok, *_TOKEN_EXPAND.get(tok, ())):
            if item in FILLER or item in seen_t:
                continue
            seen_t.add(item)
            expanded.append(item)
    non_generic = frozenset(t for t in expanded if t not in GENERIC_ALONE)
    generic_paired = frozenset(
        t
        for ph in phrases
        for t in _latin_tokens(ph) + _CJK_RUN.findall(ph)
        if t in GENERIC_ALONE
    )
    core_non_generic = frozenset(t for t in leftover_tokens if t not in GENERIC_ALONE)
    known_norms = {_norm(k) for k in _KNOWN_TOPIC_PHRASES}
    phrases = [p for p in phrases if _phrase_usable(p) or _norm(p) in known_norms]
    strong = any((" " in p) or (_CJK_RUN.fullmatch(p) and len(p) >= 2) for p in phrases)
    distinctive = any(_is_distinctive(t) for t in leftover_tokens + list(phrases))
    empty = not phrases and not distinctive and not strong
    if leftover_tokens == [] and not phrases:
        empty = True
    if phrases and not distinctive and not strong:
        # only weak/gibberish crumbs survived → abstain
        empty = True
        phrases = []
    topic_text = " ".join(phrases) if phrases else " ".join(t for t in leftover_tokens if _is_distinctive(t))
    return GoalQuery(
        raw=raw,
        phrases=tuple(phrases),
        content_tokens=tuple(expanded),
        non_generic=non_generic,
        generic_paired=generic_paired,
        core_non_generic=core_non_generic,
        type_filter=type_filter,
        must_not=must_not,
        topic_text=topic_text,
        empty=empty,
        source=source,
    )


def _doi_from_goal(raw: str) -> str | None:
    """Return a DOI when the goal is primarily that identifier (exact-pin path)."""
    m = _DOI_RE.search(raw or "")
    if not m:
        return None
    doi = m.group(1).rstrip(".,;:)]")
    rest = (raw[: m.start()] + " " + raw[m.end() :]).strip()
    rest_n = _norm(_strip_cjk_filler(rest.lower()))
    rest_toks = [
        t
        for t in _latin_tokens(rest_n) + _CJK_RUN.findall(rest_n)
        if t not in FILLER and t not in TYPE_MAP and t not in GENERIC_ALONE and t not in WEAK_ALONE
    ]
    # Pure DOI / DOI + filler → pin-only. Extra real topics keep normal parse.
    if rest_toks and any(_is_distinctive(t) for t in rest_toks):
        return None
    return doi



def _extract_assay_phrases(raw: str) -> list[str]:
    """Keep hyphenated assay phrases before _norm splits them apart.

    Prefer the most specific assay span: ``single-cell RNA-seq`` must not also
    register a bare ``rna-seq`` hit from the same substring (that OR would let
    bulk RNA-seq records pass an scRNA AND gate).
    """
    found: list[str] = []
    seen: set[str] = set()
    occupied: list[tuple[int, int]] = []
    for cre in _ASSAY_PHRASE_RES:
        for m in cre.finditer(raw or ""):
            span = (m.start(1), m.end(1))
            # Skip if this match sits inside a longer assay span already taken
            if any(span[0] >= a and span[1] <= b for a, b in occupied):
                continue
            ph = _norm(m.group(1))
            if not ph or ph in seen:
                continue
            # Drop any previously kept shorter spans fully covered by this one
            drop: list[str] = []
            for prev in list(found):
                # covered by specificity handled below; keep list clean of nested
                pass
            seen.add(ph)
            found.append(ph)
            occupied.append(span)
    return _narrow_assay_phrases(found)


_SCRNA_NORM = frozenset(
    {
        "single cell rna seq",
        "scrna seq",
        "scrnaseq",
        "sc rna seq",
    }
)
_BULK_RNA_NORM = frozenset({"bulk rna seq", "bulkrnaseq"})
_RNA_SEQ_NORM = frozenset({"rna seq", "rnaseq", "rna sequencing"})


def _narrow_assay_phrases(found: list[str]) -> list[str]:
    """Collapse assay hits to one specificity tier for AND matching."""
    norms = [_norm(x) for x in found]
    if any(n in _SCRNA_NORM or n.startswith("single cell rna") for n in norms):
        # Strict scRNA gate — do NOT OR in bare rna-seq (bulk would pass).
        return [
            "single cell rna seq",
            "scrna seq",
            "scrnaseq",
            "single-cell rna-seq",
            "scRNA-seq",
        ]
    if any(n in _BULK_RNA_NORM for n in norms):
        return ["bulk rna seq", "bulkrnaseq", "bulk RNA-seq"]
    if any(n in _RNA_SEQ_NORM for n in norms):
        # Broad RNA-seq: child scRNA may satisfy; chip-seq / atac must not.
        return [
            "rna seq",
            "rnaseq",
            "rna sequencing",
            "single cell rna seq",
            "scrna seq",
            "scrnaseq",
            "bulk rna seq",
        ]
    return found



def _phrase_tokens_in_blob(phrase: str, blob: str) -> bool:
    """True if phrase tokens appear as a contiguous span in blob tokens.

    Rejects ``small cell lung`` inside ``non small cell lung``.
    """
    ptoks = [t for t in _norm(phrase).split() if t]
    btoks = [t for t in _norm(blob).split() if t]
    if not ptoks or len(ptoks) > len(btoks):
        return False
    m = len(ptoks)
    for i in range(len(btoks) - m + 1):
        if btoks[i : i + m] != ptoks:
            continue
        if ptoks and ptoks[0] == "small" and i > 0 and btoks[i - 1] == "non":
            continue
        return True
    return False


def _alias_in_goal(alias: str, bag: set[str], blob: str) -> bool:
    """True if alias is intentionally present in the RAW goal (no short-code substrings)."""
    ng = _norm(alias)
    cg = _compact(alias)
    if not ng:
        return False
    if ng in bag or (cg and cg in bag):
        return True
    if " " in ng and _phrase_tokens_in_blob(ng, blob):
        return True
    # Long single tokens only (avoid 3–5 letter codes as substrings of nsclc etc.).
    if " " not in ng and len(ng) >= 10 and ng in blob.split():
        return True
    return False


def _facet_groups_from_goal(raw: str, phrases: list[str] | None = None, tokens: list[str] | None = None) -> tuple[tuple[str, ...], ...]:
    """Build AND facets from the RAW goal only (ignore synonym expansion).

    ``phrases`` / ``tokens`` are accepted for call-site back-compat but unused for
    disease/drug triggers — expanded NSCLC→lung adenocarcinoma must not invent facets.
    """
    del phrases, tokens  # explicit: do not use expanded synonyms for facet triggers
    blob = _norm(raw or "")
    bag = set(_latin_tokens(blob))
    # Keep multi-word spans that literally appear in the raw goal.
    for n in range(2, 6):
        toks = blob.split()
        for i in range(0, max(0, len(toks) - n + 1)):
            ph = " ".join(toks[i : i + n])
            bag.add(ph)
            bag.add(_compact(ph))
    facets: list[tuple[str, ...]] = []
    fired_labels: list[str] = []

    for label, triggers, match_syns in _DISEASE_FACET_SPECS:
        if any(_alias_in_goal(g, bag, blob) for g in triggers):
            facets.append(tuple(sorted(match_syns, key=lambda s: (-len(s), s))))
            fired_labels.append(label)

    # NSCLC is more specific than broad lung cancer — drop the broad chip if both fire.
    if "NSCLC" in fired_labels and "lung cancer" in fired_labels:
        keep: list[tuple[str, ...]] = []
        labels_keep: list[str] = []
        for fac, lab in zip(facets, fired_labels):
            if lab == "lung cancer":
                continue
            keep.append(fac)
            labels_keep.append(lab)
        facets, fired_labels = keep, labels_keep

    # Explicit "<site> cancer" in raw when not already covered by a fired disease facet.
    covered_sites = set()
    for lab in fired_labels:
        covered_sites.add(_norm(lab).replace(" cancer", "").strip())
        covered_sites.add(_norm(lab))
    for n in range(2, 5):
        toks = blob.split()
        for i in range(0, max(0, len(toks) - n + 1)):
            nph = " ".join(toks[i : i + n])
            if not nph.endswith(" cancer") or len(nph) <= 7:
                continue
            site = nph[: -len(" cancer")].strip()
            if not site or site in GENERIC_ALONE or site in {"the", "a", "human", "non", "small", "cell"}:
                continue
            if site in covered_sites or nph in covered_sites:
                continue
            if any(site in {_norm(x) for x in fac} or nph in {_norm(x) for x in fac} for fac in facets):
                continue
            facets.append((nph, _compact(nph), site))
            covered_sites.add(site)
            covered_sites.add(nph)

    for group in _DRUG_FACET_GROUPS:
        if any(_alias_in_goal(g, bag, blob) for g in group):
            facets.append(tuple(sorted(group, key=lambda s: (-len(s), s))))

    assays = _extract_assay_phrases(raw)
    if assays:
        facets.append(tuple(assays))

    out: list[tuple[str, ...]] = []
    seen_f: set[tuple[str, ...]] = set()
    for f in facets:
        key = tuple(sorted({_norm(x) for x in f}))
        if key in seen_f:
            continue
        seen_f.add(key)
        out.append(f)
    return tuple(out)


def _record_blob(rec: OutputRecord) -> str:
    fields = _field_blobs(rec)
    assay = getattr(rec, "assay", None) or ""
    return _norm(" ".join([fields["title"], fields["summary"], fields["ids"], fields["landing"], assay]))


def _facet_matches_record(
    facet: tuple[str, ...],
    blob: str,
    compact: str,
    *,
    title_blob: str = "",
    title_compact: str = "",
    prefer_title_for_short: bool = False,
) -> bool:
    """Match a facet against record text. Short disease unigrams (breast) prefer title."""
    for syn in facet:
        ns = _norm(syn)
        cs = _compact(syn)
        if not ns:
            continue
        # Multiword / compact disease forms: allow full blob
        if " " in ns or len(cs) >= 10:
            if ns in blob or (cs and cs in compact):
                return True
            continue
        # Drug-like tokens (>=6) can match full blob
        if len(ns) >= 6 and not prefer_title_for_short:
            if f" {ns} " in f" {blob} " or ns in blob.split() or (cs and cs in compact):
                return True
            continue
        # Short disease unigrams: title (or title-equivalent) only — avoid summary name-drops
        tb = title_blob or blob
        tc = title_compact or compact
        if f" {ns} " in f" {tb} " or tb.startswith(ns + " ") or tb.endswith(" " + ns) or tb == ns:
            return True
        if ns in tb.split():
            return True
        if len(cs) >= 6 and cs in tc:
            return True
    return False


def _is_disease_facet(facet: tuple[str, ...]) -> bool:
    joined = " ".join(_norm(s) for s in facet)
    keys = ("breast", "nsclc", "lung", "luad", "lusc", "melanoma", "ovarian", "prostate")
    return any(k in joined for k in keys)


def _is_assay_facet(facet: tuple[str, ...]) -> bool:
    joined = " ".join(_norm(s) for s in facet)
    return any(k in joined for k in ("rna", "seq", "scrna", "perturb", "crop", "wes", "wgs", "atac", "chip"))


def _required_facets_pass(rec: OutputRecord, query: GoalQuery) -> bool:
    facets = getattr(query, "required_facets", ()) or ()
    if not facets:
        return True
    blob = _record_blob(rec)
    compact = _compact(blob)
    title_blob = _norm(rec.title or "")
    title_compact = _compact(title_blob)
    assay_facet = any(_is_assay_facet(fac) for fac in facets)
    if assay_facet and _IMAGING_ONLY_RE.search(blob):
        if not re.search(
            r"(?i)\b(rna[\s-]?seq|rnaseq|scrna|transcriptom|expression profiling by high throughput sequencing)\b",
            blob,
        ):
            return False
    for fac in facets:
        prefer_title = _is_disease_facet(fac) and any(len(_norm(s)) <= 6 and " " not in _norm(s) for s in fac)
        # Disease facets: require breast cancer / nsclc phrase OR title unigram — not summary-only "breast"
        if _is_disease_facet(fac):
            # Disease must appear in the TITLE when an assay facet is also required
            # (avoid summary name-drops like "renal and breast cancers").
            scope_blob, scope_compact = blob, compact
            if assay_facet:
                scope_blob, scope_compact = title_blob, title_compact
            multi = tuple(s for s in fac if " " in _norm(s) or len(_compact(s)) >= 10)
            short = tuple(s for s in fac if s not in multi)
            ok = False
            if multi and _facet_matches_record(multi, scope_blob, scope_compact):
                ok = True
            elif short and _facet_matches_record(
                short,
                scope_blob,
                scope_compact,
                title_blob=title_blob,
                title_compact=title_compact,
                prefer_title_for_short=True,
            ):
                ok = True
            if not ok:
                return False
            continue
        if _is_assay_facet(fac):
            fac_n = {_norm(s) for s in fac}
            wants_scrna = any(n in _SCRNA_NORM or n.startswith("single cell") for n in fac_n)
            has_broad_rna = any(n in _RNA_SEQ_NORM or n in _BULK_RNA_NORM for n in fac_n)
            if wants_scrna and not has_broad_rna:
                if not re.search(
                    r"(?i)\b(single[\s-]?cell|scRNA|scrna[\s-]?seq|sc[\s-]?rna)\b",
                    blob,
                ):
                    return False
            elif has_broad_rna or wants_scrna:
                # Reject chromatin-only assays posing as RNA-seq
                if re.search(r"(?i)\b(chip[\s-]?seq|atac[\s-]?seq)\b", blob) and not re.search(
                    r"(?i)\b(rna[\s-]?seq|rnaseq|scrna|transcriptom|rna sequencing)\b",
                    blob,
                ):
                    return False
        if not _facet_matches_record(fac, blob, compact, title_blob=title_blob, title_compact=title_compact):
            return False
    return True


def parse_goal_lexical(goal: str) -> GoalQuery:
    raw = (goal or "").strip()
    doi = _doi_from_goal(raw)
    if doi:
        # Exact DOI only — never fall back to token soup (doi/hart/0001).
        # Bypass _build_query/_expand_phrases so "/" "." stay on the pin string.
        d = doi.lower()
        return GoalQuery(
            raw=raw,
            phrases=(d,),
            content_tokens=(d,),
            non_generic=frozenset({d}),
            generic_paired=frozenset(),
            core_non_generic=frozenset({d}),
            type_filter=(),
            must_not=(),
            topic_text=d,
            empty=False,
            source="lexical-doi",
        )
    lowered = raw.lower()
    normed = _norm(_strip_cjk_filler(lowered))
    type_filter = _type_filter_from_text(normed, lowered)
    known, leftover_text = _extract_known_phrases(normed)
    # Re-tokenize leftover without dropping generics so "small cell" can form.
    raw_left: list[str] = []
    seen_l: set[str] = set()
    for m in re.finditer(r"[a-z0-9]+(?:-[a-z0-9]+)*|[一-鿿]+", leftover_text or ""):
        piece = m.group(0)
        if _is_filler_or_type(piece) or piece in seen_l:
            continue
        if len(piece) < 3 and piece not in {"io"} and not _CJK_RUN.fullmatch(piece):
            continue
        seen_l.add(piece)
        raw_left.append(piece)
    extra_phrases: list[str] = []
    for a, b in zip(raw_left, raw_left[1:]):
        if a in GENERIC_ALONE and b in GENERIC_ALONE:
            continue
        if _is_weak_alone(a) and _is_weak_alone(b):
            continue
        if _looks_gibberish(a) or _looks_gibberish(b):
            continue
        if _is_weak_alone(a) or _is_weak_alone(b):
            # allow "breast cancer" (cancer generic) but not "zzz disease"
            if not (_is_distinctive(a) or _is_distinctive(b)):
                continue
            if a in WEAK_ALONE or b in WEAK_ALONE or a.isdigit() or b.isdigit():
                continue
        extra_phrases.append(f"{a} {b}")
    topic_unigrams = [t for t in raw_left if _is_distinctive(t)]
    phrases = known + extra_phrases + topic_unigrams
    leftover_for_content = [t for t in raw_left if t not in FILLER and not _is_weak_alone(t)]
    # Drop generic+assay junk bigrams ("cancer rna") — assay is a separate AND facet.
    _assay_bits = {"rna", "seq", "rnaseq", "scrna", "sequencing", "single", "cell", "cells"}
    phrases = [
        ph
        for ph in phrases
        if not (
            " " in ph
            and any(a in _norm(ph).split() for a in _assay_bits)
            and any(g in _norm(ph).split() for g in GENERIC_ALONE)
        )
    ]
    # Preserve assay phrases that _norm would split (rna-seq → rna + seq).
    assays = _extract_assay_phrases(raw)
    for a in assays:
        if a not in phrases:
            phrases.append(a)
        for tok in a.split():
            if tok not in leftover_for_content and tok not in FILLER:
                leftover_for_content.append(tok)
    q = _build_query(raw, phrases, leftover_for_content, type_filter, (), "lexical")
    facets = _facet_groups_from_goal(raw, list(q.phrases), list(q.content_tokens))
    if facets:
        q = GoalQuery(
            raw=q.raw,
            phrases=q.phrases,
            content_tokens=q.content_tokens,
            non_generic=q.non_generic,
            generic_paired=q.generic_paired,
            core_non_generic=q.core_non_generic,
            type_filter=q.type_filter,
            must_not=q.must_not,
            topic_text=q.topic_text,
            empty=q.empty,
            source=q.source,
            required_facets=facets,
        )
    return q



def _slots_to_query(goal: str, slots: dict[str, Any], lexical: GoalQuery) -> GoalQuery:
    from cancer_output_atlas.goal_slots import sanitize_slots

    clean = sanitize_slots(slots, goal)
    if not clean:
        return lexical
    phrases = [p for p in clean["topic_phrases"] if p not in JUNK_PHRASES]
    # Drop leftover-style junk bigrams that include filler/type tokens.
    kept: list[str] = []
    for p in phrases:
        toks = _norm(p).split()
        if any(t in FILLER or t in TYPE_MAP for t in toks):
            continue
        if p in GENERIC_ALONE or p in WEAK_ALONE:
            continue
        if not _phrase_usable(p):
            continue
        kept.append(p)
    if not kept:
        kept = list(lexical.phrases)
    else:
        # Keep lexical expansions of the same topics (screenshot-safe).
        for p in lexical.phrases:
            if p not in kept:
                kept.append(p)
    types = tuple(clean["ods_types"]) or lexical.type_filter
    must_not = tuple(clean["must_not"])
    leftover = [t for p in kept for t in _latin_tokens(p) + _CJK_RUN.findall(p)]
    source = "gemini"
    try:
        from cancer_output_atlas.model_config import provider

        source = provider() if provider() != "none" else "lexical"
    except Exception:
        source = lexical.source
    q = _build_query(goal.strip(), kept, leftover, types, must_not, source)
    if q.empty:
        return lexical
    return q


@lru_cache(maxsize=64)
def parse_goal(goal: str) -> GoalQuery:
    lexical = parse_goal_lexical(goal)
    q = lexical
    try:
        from cancer_output_atlas.goal_slots import try_llm_slots
        from cancer_output_atlas.model_config import model_ready

        if model_ready():
            slots = try_llm_slots(goal)
            if slots:
                q = _slots_to_query(goal, slots, lexical)
    except Exception:
        pass
    # Always attach AND facets from the raw goal (disease ∧ drug ∧ assay).
    facets = _facet_groups_from_goal(q.raw, list(q.phrases), list(q.content_tokens))
    if facets and facets != getattr(q, "required_facets", ()):
        q = GoalQuery(
            raw=q.raw,
            phrases=q.phrases,
            content_tokens=q.content_tokens,
            non_generic=q.non_generic,
            generic_paired=q.generic_paired,
            core_non_generic=q.core_non_generic,
            type_filter=q.type_filter,
            must_not=q.must_not,
            topic_text=q.topic_text,
            empty=q.empty,
            source=q.source,
            required_facets=facets,
        )
    return q


def _field_blobs(rec: OutputRecord) -> dict[str, str]:
    ids = " ".join(rec.identifier_values() + [i.key() for i in rec.identifiers])
    landing = " ".join(p for p in (rec.landing_url, *rec.evidence_urls) if p)
    return {
        "title": rec.title or "",
        "summary": rec.summary or "",
        "ids": ids,
        "landing": landing,
    }


def _node_tokens(*texts: str) -> set[str]:
    found: set[str] = set()
    for text in texts:
        cleaned = _strip_cjk_filler(_norm(text))
        found.update(_latin_tokens(cleaned))
        for run in _CJK_RUN.findall(cleaned):
            found.add(run)
            if len(run) >= 4:
                found.add(run[:2])
                found.add(run[-2:])
    return found


def _phrase_in(phrase: str, norm_text: str, compact_text: str) -> bool:
    if not phrase:
        return False
    nph = _norm(phrase)
    if nph and nph in norm_text:
        return True
    cph = _compact(phrase)
    if cph and len(cph) >= 6 and cph in compact_text:
        return True
    return False


_FIELD_ZH = {
    "title": "标题",
    "summary": "摘要",
    "ids": "标识",
    "landing": "链接",
}


def _why_text(field: str, snippet: str) -> str:
    label = _FIELD_ZH.get(field, "文本")
    snip = " ".join(snippet.split())
    if len(snip) > 48:
        snip = snip[:45] + "…"
    return f"{label}含 {snip}"


def _doc_tokens(rec: OutputRecord) -> list[str]:
    fields = _field_blobs(rec)
    blob = _norm(" ".join(fields[k] for k in ("title", "summary", "ids")))
    return _latin_tokens(blob) + _CJK_RUN.findall(blob)


class _BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.N = len(docs)
        self.df: Counter[str] = Counter()
        self.doc_len = [len(d) for d in docs]
        self.avgdl = (sum(self.doc_len) / self.N) if self.N else 0.0
        self.tf = [Counter(d) for d in docs]
        for d in docs:
            for t in set(d):
                self.df[t] += 1

    def score(self, query_tokens: list[str], idx: int) -> float:
        dl = self.doc_len[idx] or 1
        tf = self.tf[idx]
        s = 0.0
        avgdl = self.avgdl or 1.0
        for t in query_tokens:
            f = tf.get(t, 0)
            if not f:
                continue
            n = self.df.get(t, 0)
            idf = math.log(1.0 + (self.N - n + 0.5) / (n + 0.5))
            denom = f + self.k1 * (1 - self.b + self.b * dl / avgdl)
            s += idf * (f * (self.k1 + 1)) / denom
        return s


@dataclass
class _Match:
    phrase_field: str
    phrase_hit: str
    ods: str
    pan_cancer: bool = False


# Software / tool / method / model / biospecimen are reusable across cancers.
_PAN_CANCER_ODS = frozenset({"software", "tool", "method", "model", "biospecimen"})
_PAN_CANCER_WHY = "Cross-cancer catalog resource (not filtered by cancer type)."

# Explicit cancer-type lexicon. Drugs (pembrolizumab / Keytruda) alone are not
# a cancer type; NSCLC+pembro is, because of nsclc.
_CANCER_TYPE_TOKENS = frozenset(
    {
        "nsclc",
        "luad",
        "lusc",
        "lung",
        "breast",
        "melanoma",
    }
)
_CANCER_TYPE_PHRASES = (
    "non-small cell",
    "non small cell",
    "non-small-cell",
    "lung cancer",
    "lung adenocarcinoma",
    "nonsmallcell",
    "lungcancer",
    "lungadenocarcinoma",
)


def _ods_of(rec: OutputRecord) -> str:
    return ods_category(
        kind=rec.kind,
        title=rec.title,
        summary=rec.summary,
        ids=[i.key() for i in rec.identifiers],
        landing_url=rec.landing_url,
    )


def _is_cancer_type_goal(query: GoalQuery) -> bool:
    """True when the goal names a cancer type (NSCLC, lung, breast, melanoma, …)."""
    if query.empty:
        return False
    blob = " ".join(
        [
            query.raw or "",
            query.topic_text or "",
            " ".join(query.phrases),
            " ".join(query.content_tokens),
        ]
    )
    compact = _compact(blob)
    normed = _norm(blob)
    for ph in _CANCER_TYPE_PHRASES:
        nph = _norm(ph)
        cph = _compact(ph)
        if nph and nph in normed:
            return True
        if cph and len(cph) >= 6 and cph in compact:
            return True
    tokens = set(query.content_tokens) | set(_latin_tokens(normed))
    if tokens & _CANCER_TYPE_TOKENS:
        return True
    for tok in ("nsclc", "luad", "lusc", "melanoma"):
        if tok in compact:
            return True
    return False


def _pan_cancer_score(rec: OutputRecord) -> float:
    """Keep attached catalog nodes below topic-matched data/trials (~0.12–0.20)."""
    prior = _reuse_prior(rec)
    return round(min(0.20, 0.12 + 0.08 * prior), 3)


def _topic_gate(rec: OutputRecord, query: GoalQuery) -> _Match | None:
    if rec.source_status == "skipped":
        return None
    if query.empty:
        return None
    fields = _field_blobs(rec)
    norms = {k: _norm(v) for k, v in fields.items()}
    compacts = {k: _compact(v) for k, v in fields.items()}
    tokens = _node_tokens(*fields.values())
    phrase_field = ""
    phrase_hit = ""
    for field_name in ("title", "summary", "ids", "landing"):
        for ph in query.phrases:
            if ph in JUNK_PHRASES:
                continue
            if _phrase_in(ph, norms[field_name], compacts[field_name]):
                phrase_field = field_name
                phrase_hit = ph
                break
        if phrase_hit:
            break
    multiword = [
        p
        for p in query.phrases
        if (" " in p) or (_CJK_RUN.fullmatch(p) and len(p) >= 2)
    ]
    distinctive_q = [
        t
        for t in query.phrases
        if " " not in t and _is_distinctive(t) and t not in GENERIC_ALONE
    ]
    if multiword:
        # A 2+ word topic (virtual cell) must match as a phrase. Do not
        # fall back to leftover unigrams (virtual → virtual reality).
        if not phrase_hit:
            return None
        mw_norm = {_norm(p) for p in multiword}
        mw_compact = {_compact(p) for p in multiword if len(_compact(p)) >= 6}
        hit_ok = (
            phrase_hit in multiword
            or _norm(phrase_hit) in mw_norm
            or (_compact(phrase_hit) in mw_compact)
        )
        if not hit_ok:
            if phrase_hit not in query.phrases:
                return None
            if " " not in phrase_hit and _is_weak_alone(phrase_hit):
                return None
            # Unigram expansion of the same topic (virtualcell / 虚拟细胞) is ok
            # only when distinctive.
            if not _is_distinctive(phrase_hit) and not _CJK_RUN.fullmatch(phrase_hit or ""):
                return None
    elif distinctive_q:
        distinctive_q = [t for t in distinctive_q if not _is_weak_alone(t)]
        if not distinctive_q:
            return None
        if not phrase_hit:
            hit_tok = next((t for t in distinctive_q if t in tokens or t in norms["title"] or t in norms["summary"]), "")
            if not hit_tok:
                return None
            title_toks = set(_latin_tokens(norms["title"])) | set(_CJK_RUN.findall(norms["title"]))
            phrase_field = "title" if hit_tok in title_toks or hit_tok in norms["title"] else "summary"
            phrase_hit = hit_tok
        elif _is_weak_alone(phrase_hit):
            return None
    else:
        return None
    if any(_norm(m) and _norm(m) in " ".join(norms.values()) for m in query.must_not):
        return None
    ods = _ods_of(rec)
    if query.type_filter and ods not in query.type_filter:
        return None
    if not _required_facets_pass(rec, query):
        return None
    return _Match(phrase_field=phrase_field, phrase_hit=phrase_hit, ods=ods)


# ClinicalTrials.gov BasicSearch field weights (API search areas):
# NCTId 1.0, BriefTitle 0.89, OfficialTitle 0.85, BriefSummary 0.60.
# https://clinicaltrials.gov/data-api/about-api/search-areas
_FIELD_W = {
    "title": 0.89,
    "ids": 1.0,
    "summary": 0.60,
    "landing": 0.35,
}
_W_SUM = sum(_FIELD_W.values())



def _facet_display_label(facet: tuple[str, ...]) -> str:
    """One human chip per AND facet (canonical labels for UI)."""
    norms = {_norm(s) for s in facet}
    # Canonical disease / drug / assay chips
    if norms & {"breast", "breast cancer", "breastcancer"}:
        return "breast cancer"
    # NSCLC chip: prefer when nsclc/luad/lusc triggers are in the facet match set
    # and the facet was the NSCLC spec (has nsclc code among match syns with luad).
    if norms & {"nsclc", "non small cell lung", "non-small cell lung", "nonsmallcelllung", "luad", "lusc"} and (
        "nsclc" in norms or "luad" in norms or "lusc" in norms or "non small cell lung" in norms or "non-small cell lung" in norms
    ):
        return "NSCLC"
    if norms & {"lung cancer", "lungcancer", "lung adenocarcinoma", "lungadenocarcinoma", "lung squamous", "sclc", "small cell lung", "small cell lung cancer"}:
        return "lung cancer"
    if norms & {"pembrolizumab", "keytruda", "mk-3475", "mk3475", "mk 3475"}:
        return "pembrolizumab / Keytruda"
    if norms & {"immunotherapy", "immuno", "checkpoint", "pd-1", "pd1", "pd-l1", "pdl1", "pd l1"}:
        return "immunotherapy"
    # Broad RNA-seq facets may list scRNA as a *match child*; chip must stay rna-seq
    # unless the facet is scRNA-only (no bare rna-seq / bulk synonyms).
    if any(n in _SCRNA_NORM or n.startswith("single cell") for n in norms) and not any(
        n in _RNA_SEQ_NORM or n in _BULK_RNA_NORM for n in norms
    ):
        return "scRNA-seq"
    if any(n in _BULK_RNA_NORM for n in norms) and not any(n in _RNA_SEQ_NORM | _SCRNA_NORM for n in norms):
        return "bulk RNA-seq"
    if any(n in _RNA_SEQ_NORM or n in _BULK_RNA_NORM or n in _SCRNA_NORM for n in norms):
        return "rna-seq"
    assay_map = {
        "rna seq": "rna-seq",
        "rnaseq": "rna-seq",
        "scrna seq": "scRNA-seq",
        "single cell rna seq": "scRNA-seq",
        "bulk rna seq": "bulk RNA-seq",
    }
    ordered = sorted(facet, key=lambda s: (0 if " " in _norm(s) else 1, -len(s), s))
    if not ordered:
        return ""
    label = _norm(ordered[0])
    return assay_map.get(label, label)


def _public_overlap(
    phrases: list[str],
    *,
    query: GoalQuery | None = None,
    rec: OutputRecord | None = None,
    limit: int = 4,
) -> list[str]:
    """Matches chips: one label per matched AND facet (breast cancer · rna-seq)."""
    out: list[str] = []
    seen_c: set[str] = set()

    def add(label: str) -> None:
        n = _norm(label.replace("-", " "))
        c = _compact(n)
        if not n or c in seen_c:
            return
        if " " not in n and "-" not in label and any(
            n in _norm(x.replace("-", " ")).split() for x in out
        ):
            return
        seen_c.add(c)
        out.append(label)

    if query is not None and rec is not None and getattr(query, "required_facets", ()):
        blob = _record_blob(rec)
        compact = _compact(blob)
        title_blob = _norm(rec.title or "")
        title_compact = _compact(title_blob)
        assay_facet = any(_is_assay_facet(fac) for fac in query.required_facets)
        for fac in query.required_facets:
            if _is_disease_facet(fac) and assay_facet:
                scope_blob, scope_compact = title_blob, title_compact
            else:
                scope_blob, scope_compact = blob, compact
            multi = tuple(s for s in fac if " " in _norm(s) or len(_compact(s)) >= 10)
            short = tuple(s for s in fac if s not in multi)
            ok = False
            if multi and _facet_matches_record(multi, scope_blob, scope_compact):
                ok = True
            elif short and _facet_matches_record(
                short,
                scope_blob,
                scope_compact,
                title_blob=title_blob,
                title_compact=title_compact,
                prefer_title_for_short=True,
            ):
                ok = True
            elif (not _is_disease_facet(fac)) and _facet_matches_record(fac, blob, compact):
                ok = True
            if ok:
                add(_facet_display_label(fac))
        if out:
            return out[:limit]

    ordered = sorted(
        (ph for ph in phrases if ph and ph not in JUNK_PHRASES),
        key=lambda s: (0 if " " in _norm(s) else 1, -len(s), s),
    )
    for ph in ordered:
        add(ph if (" " in ph or "-" in ph) else _norm(ph))
        if len(out) >= limit:
            break
    return out


def _active_phrases(query: GoalQuery) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for p in query.phrases:
        if p in JUNK_PHRASES:
            continue
        key = _norm(p)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def _hits_by_field(rec: OutputRecord, phrases: list[str]) -> dict[str, list[str]]:
    fields = _field_blobs(rec)
    norms = {k: _norm(v) for k, v in fields.items()}
    compacts = {k: _compact(v) for k, v in fields.items()}
    found: dict[str, list[str]] = {k: [] for k in fields}
    for field_name in fields:
        for ph in phrases:
            if _phrase_in(ph, norms[field_name], compacts[field_name]):
                found[field_name].append(ph)
    return found


def _coverage(hits: dict[str, list[str]], phrases: list[str]) -> tuple[float, int, int]:
    n = len(phrases)
    if not n:
        return 0.0, 0, 0
    title_n = len(set(hits.get("title") or []))
    any_set: set[str] = set()
    for vals in hits.values():
        any_set.update(vals)
    any_n = len(any_set)
    # Title hits count more, like CT.gov BriefTitle vs BriefSummary.
    cov = 0.70 * (title_n / n) + 0.30 * (any_n / n)
    return cov, title_n, any_n


def _reuse_prior(rec: OutputRecord) -> float:
    reuse = rec.reuse or {}
    flag = reuse.get("has_results")
    if flag is True or str(flag).lower() in {"true", "1", "yes"}:
        return 1.0
    if flag is False or str(flag).lower() in {"false", "0", "no"}:
        return 0.0
    return 0.35


def _id_pin(rec: OutputRecord, phrases: list[str], goal: str = "") -> bool:
    """Exact accession / NCT / GEO / DOI pin (Invenio identifiers^12, Papers-with-Code title/id pin)."""
    vals = [(i.value or "").lower() for i in rec.identifiers]
    vals.append((rec.output_id or "").lower())
    joined = "".join(vals).replace(" ", "")
    compact = joined.replace("-", "").replace("_", "").replace(":", "").replace("/", "")
    cands = list(phrases) + _latin_tokens(_norm(goal))
    doi_goal = _doi_from_goal(goal) if goal else None
    if doi_goal:
        cands = [doi_goal] + cands
    for p in cands:
        raw_p = (p or "").strip().lower()
        n = _norm(raw_p).replace(" ", "").replace("-", "").replace("/", "")
        if len(n) < 6:
            continue
        is_doi = bool(_DOI_RE.search(raw_p)) or raw_p.startswith("10.")
        is_acc = bool(
            _ACCESSION.match(n) or n.startswith("nct") or n.startswith("gse") or n.startswith("gsm")
        )
        if not (is_doi or is_acc):
            continue
        if n in compact or raw_p.replace(" ", "") in joined:
            return True
    return False


def _minmax(vals: list[float]) -> list[float]:
    if not vals:
        return []
    lo, hi = min(vals), max(vals)
    if hi <= lo:
        return [0.0] * len(vals)
    return [(v - lo) / (hi - lo) for v in vals]


def score_record_against_goal(rec: OutputRecord, query: GoalQuery) -> dict[str, Any] | None:
    """Topic-gate one record. None = drop (no topic match).

    On a cancer-type goal, software/tool/method/model/biospecimen skip the
    topic gate (still honor type_filter and skipped status).
    """
    if rec.source_status == "skipped":
        return None
    if query.empty:
        return None
    ods = _ods_of(rec)
    if query.type_filter and ods not in query.type_filter:
        return None
    if _is_cancer_type_goal(query) and ods in _PAN_CANCER_ODS:
        return {
            "rank_score": _pan_cancer_score(rec),
            "goal_overlap": [],
            "why": _PAN_CANCER_WHY,
            "profile_fit": 0.0 if rec.classification is None else float(rec.classification.profile_fit or 0.0),
            "_ods": ods,
            "_field": "",
            "_title_phrases": 0,
            "_pan_cancer": True,
        }
    hit = _topic_gate(rec, query)
    if hit is None:
        return None
    phrases = _active_phrases(query)
    hits = _hits_by_field(rec, phrases)
    cov, title_n, _any_n = _coverage(hits, phrases)
    field_w = _FIELD_W.get(hit.phrase_field, 0.35)
    prior = _reuse_prior(rec)
    # Single-doc path has no BM25 corpus; field weight + coverage still order hits.
    score = 0.55 * field_w + 0.40 * cov + 0.05 * prior
    overlap: list[str] = []
    for ph in (hits.get("title") or []) + (hits.get("summary") or []) + (hits.get("ids") or []):
        if ph not in overlap:
            overlap.append(ph)
    return {
        "rank_score": round(min(1.0, score), 3),
        "goal_overlap": _public_overlap(overlap, query=query, rec=rec) or ([hit.phrase_hit] if hit.phrase_hit else []),
        "why": _why_text(hit.phrase_field or "title", hit.phrase_hit or query.topic_text),
        "profile_fit": 0.0 if rec.classification is None else float(rec.classification.profile_fit or 0.0),
        "_ods": hit.ods,
        "_field": hit.phrase_field,
        "_title_phrases": title_n,
    }


def _rank_profile_fit(
    records: list[OutputRecord],
    goal: str,
    top: int | None,
) -> list[dict[str, Any]]:
    goal_toks = set(_latin_tokens(goal))
    rows: list[dict[str, Any]] = []
    for rec in records:
        if rec.source_status == "skipped":
            continue
        row = classification_row(rec)
        blob = " ".join(
            [
                rec.title,
                rec.summary,
                rec.perturbation or "",
                rec.assay or "",
                " ".join(rec.classification.labels if rec.classification else []),
            ]
        )
        overlap = set(_latin_tokens(blob)) & goal_toks
        fit = 0.0 if rec.classification is None else rec.classification.profile_fit
        bonus = min(0.15, 0.03 * len(overlap))
        score = round(min(1.0, fit + bonus), 3)
        row["rank_score"] = score
        row["goal_overlap"] = sorted(overlap)
        row["why"] = f"profile_fit={fit}; goal-token overlap={sorted(overlap) or 'none'}"
        rows.append(row)
    rows.sort(key=lambda r: (-r["rank_score"], r["output_id"]))
    if top is not None:
        rows = rows[:top]
    for i, row in enumerate(rows, start=1):
        row["rank"] = i
    return rows


def rank_records(
    records: list[OutputRecord],
    goal: str = GOAL_DEFAULT,
    top: int | None = None,
    *,
    query_first: bool = True,
    query: GoalQuery | None = None,
) -> list[dict[str, Any]]:
    """Rank *records* against *goal*.

    query_first=True (find path): require a topic-slot match in title/summary/ids/
    landing. No topic match → drop. Among gated hits, score is weighted-field
    BM25 (CT.gov BasicSearch weights) plus how many query phrases hit, plus a
    small hasResults prior. Ties: more title phrases, then hasResults, then id.
    profile_fit is not used as a score.

    query_first=False: keep the ingest-time profile_fit table used by classify.
    """
    if not query_first:
        return _rank_profile_fit(records, goal, top)

    q = query or parse_goal(goal)
    if q.empty:
        return []

    cancer_goal = _is_cancer_type_goal(q)
    gated: list[tuple[OutputRecord, _Match]] = []
    attached: list[tuple[OutputRecord, _Match]] = []
    for rec in records:
        if rec.source_status == "skipped":
            continue
        ods = _ods_of(rec)
        if q.type_filter and ods not in q.type_filter:
            continue
        # Pan-cancer catalog tools (software/tool/method/model/biospecimen) still attach
        # on cancer-type goals. Data/trials stay on the AND topic gate below.
        if cancer_goal and ods in _PAN_CANCER_ODS:
            attached.append(
                (rec, _Match(phrase_field="", phrase_hit="", ods=ods, pan_cancer=True))
            )
            continue
        hit = _topic_gate(rec, q)
        if hit is None:
            continue
        gated.append((rec, hit))
    if not gated and not attached:
        return []

    phrases = _active_phrases(q)
    q_tokens = _latin_tokens(_norm(q.topic_text)) + _CJK_RUN.findall(_norm(q.topic_text))
    for p in phrases:
        q_tokens.extend(_latin_tokens(_norm(p)))

    # Per-field BM25 on the already gated set (Solr/TREC-PM style). Always on,
    # not only the lexical parser — Gemini still needs a real score among hits.
    field_norm: dict[str, list[float]] = {k: [0.0] * max(len(gated), 1) for k in _FIELD_W}
    if len(gated) > 1 and q_tokens:
        for field_name in _FIELD_W:
            docs = []
            for rec, _hit in gated:
                blob = _norm(_field_blobs(rec)[field_name])
                docs.append(_latin_tokens(blob) + _CJK_RUN.findall(blob))
            bm25 = _BM25(docs)
            raw = [bm25.score(q_tokens, i) for i in range(len(gated))]
            field_norm[field_name] = _minmax(raw)

    rows: list[dict[str, Any]] = []
    for i, (rec, hit) in enumerate(gated):
        hits = _hits_by_field(rec, phrases)
        cov, title_n, any_n = _coverage(hits, phrases)
        weighted = 0.0
        for field_name, w in _FIELD_W.items():
            weighted += w * field_norm[field_name][i]
        weighted = weighted / _W_SUM if _W_SUM else 0.0
        prior = _reuse_prior(rec)
        pinned = _id_pin(rec, phrases, q.raw)
        # 0.50 BM25 + 0.40 phrase coverage + 0.10 reuse (hasResults).
        base = 0.50 * weighted + 0.40 * cov + 0.10 * prior
        if pinned:
            base = 0.99 + 0.01 * base
        else:
            # Always above pan-cancer attach (~0.12–0.20) so first_look
            # stays topic-matched data/trials, including the 1-doc BM25=0 path.
            base = max(base, 0.21)
        overlap: list[str] = []
        for ph in (hits.get("title") or []) + (hits.get("summary") or []) + (hits.get("ids") or []):
            if ph not in overlap:
                overlap.append(ph)
        row = classification_row(rec)
        row["rank_score"] = round(min(1.0, base), 3)
        row["goal_overlap"] = _public_overlap(overlap, query=q, rec=rec) or ([hit.phrase_hit] if hit.phrase_hit else list(q.phrases[:2]))
        row["why"] = _why_text(hit.phrase_field or "title", hit.phrase_hit or q.topic_text)
        row["_title_phrases"] = title_n
        row["_phrase_hits"] = any_n
        row["_has_results"] = prior
        row["_id_pin"] = pinned
        rows.append(row)
    for rec, hit in attached:
        prior = _reuse_prior(rec)
        row = classification_row(rec)
        row["rank_score"] = _pan_cancer_score(rec)
        row["goal_overlap"] = []
        row["why"] = _PAN_CANCER_WHY
        row["_title_phrases"] = 0
        row["_phrase_hits"] = 0
        row["_has_results"] = prior
        row["_id_pin"] = False
        row["_pan_cancer"] = True
        rows.append(row)
    rows.sort(
        key=lambda r: (
            -r["rank_score"],
            -int(bool(r.get("_id_pin"))),
            -int(r.get("_title_phrases") or 0),
            -float(r.get("_has_results") or 0),
            r["output_id"],
        )
    )
    if top is not None:
        rows = rows[:top]
    for i, row in enumerate(rows, start=1):
        row["rank"] = i
    return rows


def format_rank_table(rows: list[dict[str, Any]]) -> str:
    lines = ["rank\tscore\tkind\tids\ttitle"]
    for row in rows:
        ids = ",".join(row.get("ids") or [])
        lines.append(
            f"{row['rank']}\t{row['rank_score']}\t{row['kind']}\t{ids}\t{row['title']}"
        )
    return "\n".join(lines) + "\n"

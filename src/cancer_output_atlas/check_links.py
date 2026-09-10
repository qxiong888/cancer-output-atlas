"""Daily landing-URL liveness check. Drops gone pages; keeps login walls."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable
from urllib.parse import urlparse

from cancer_output_atlas.safety import GS_RE, is_omics_filename, offline
from cancer_output_atlas.schema import AtlasGraph, Link, OutputRecord

ISO = "%Y-%m-%dT%H:%M:%SZ"


@dataclass(frozen=True)
class ProbeResult:
    url: str
    alive: bool
    status: int | None
    reason: str


def _now() -> str:
    return datetime.now(timezone.utc).strftime(ISO)


def _http_probe(url: str, *, timeout: float = 12.0, attempts: int = 2) -> ProbeResult:
    """HEAD, then small GET if HEAD is 405/403. Never download a body."""
    import http.client
    import socket
    import ssl

    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return ProbeResult(url, True, None, "skip: not-http")
    if is_omics_filename(parsed.path or ""):
        return ProbeResult(url, True, None, "skip: omics-filename-not-fetched")
    if GS_RE.match(url):
        return ProbeResult(url, True, None, "skip: gs-pointer")

    host = parsed.hostname
    if not host:
        return ProbeResult(url, False, None, "dead: no-host")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"

    last_status: int | None = None
    last_reason = "dead: unreachable"
    for attempt in range(attempts):
        try:
            if parsed.scheme == "https":
                conn = http.client.HTTPSConnection(
                    host, port=port, timeout=timeout, context=ssl.create_default_context()
                )
            else:
                conn = http.client.HTTPConnection(host, port=port, timeout=timeout)
            try:
                conn.request(
                    "HEAD",
                    path,
                    headers={"User-Agent": "CancerOutputAtlas/0.1 (link-liveness HEAD)"},
                )
                resp = conn.getresponse()
                status = resp.status
                resp.read(64)
            finally:
                conn.close()
        except (socket.timeout, TimeoutError):
            last_reason = "dead: timeout"
            last_status = None
            continue
        except socket.gaierror:
            return ProbeResult(url, False, None, "dead: dns")
        except OSError as exc:
            last_reason = f"dead: {type(exc).__name__}"
            last_status = None
            continue

        last_status = status
        if status in {405, 403}:
            get_result = _small_get(parsed, path, timeout=timeout)
            if get_result is not None:
                return get_result
            # HEAD 403 and GET also failed to classify — host answered 403.
            if status == 403:
                return ProbeResult(url, True, 403, "alive: login-wall")
        if 200 <= status < 400:
            return ProbeResult(url, True, status, "alive")
        if status in {401, 403}:
            return ProbeResult(url, True, status, "alive: login-wall")
        if status in {404, 410}:
            return ProbeResult(url, False, status, f"dead: http {status}")
        if status >= 500:
            last_reason = f"dead: http {status}"
            continue
        # Other 4xx (except 401/403/404/410): treat as gone.
        return ProbeResult(url, False, status, f"dead: http {status}")

    return ProbeResult(url, False, last_status, last_reason)


def _small_get(parsed, path: str, timeout: float) -> ProbeResult | None:
    import http.client
    import socket
    import ssl

    url = parsed.geturl()
    host = parsed.hostname
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        if parsed.scheme == "https":
            conn = http.client.HTTPSConnection(
                host, port=port, timeout=timeout, context=ssl.create_default_context()
            )
        else:
            conn = http.client.HTTPConnection(host, port=port, timeout=timeout)
        try:
            conn.request(
                "GET",
                path,
                headers={
                    "User-Agent": "CancerOutputAtlas/0.1 (link-liveness GET)",
                    "Range": "bytes=0-511",
                },
            )
            resp = conn.getresponse()
            status = resp.status
            resp.read(512)
        finally:
            conn.close()
    except (socket.timeout, TimeoutError, OSError):
        return None
    if 200 <= status < 400:
        return ProbeResult(url, True, status, "alive")
    if status in {401, 403}:
        return ProbeResult(url, True, status, "alive: login-wall")
    if status in {404, 410} or status >= 500:
        return ProbeResult(url, False, status, f"dead: http {status}")
    return ProbeResult(url, False, status, f"dead: http {status}")


def probe_url(url: str) -> ProbeResult:
    return _http_probe(url)


def _urls_for_node(rec: OutputRecord) -> list[str]:
    urls: list[str] = []
    if rec.landing_url:
        urls.append(rec.landing_url)
    for u in rec.evidence_urls or []:
        if u and u not in urls:
            urls.append(u)
    return urls


def check_graph_links(
    graph: AtlasGraph,
    *,
    checker: Callable[[str], ProbeResult] | None = None,
    dropped_path: Path | None = None,
    checked_at: str | None = None,
    skip_if_offline: bool = True,
) -> tuple[AtlasGraph, list[dict]]:
    """Remove nodes whose landing page is gone. Login walls stay as pointers.

    `checker` is injectable for offline tests. Live HTTP is skipped when
    COA_OFFLINE=1 unless a checker is provided.
    """
    if checker is None:
        if skip_if_offline and offline():
            return graph, []
        checker = probe_url

    when = checked_at or _now()
    cache: dict[str, ProbeResult] = {}
    dropped_nodes: list[OutputRecord] = []
    dropped_rows: list[dict] = []

    def probe(url: str) -> ProbeResult:
        if url not in cache:
            cache[url] = checker(url)
        return cache[url]

    keep: list[OutputRecord] = []
    for rec in graph.nodes:
        urls = _urls_for_node(rec)
        http_urls = [
            u
            for u in urls
            if urlparse(u).scheme in {"http", "https"} and not is_omics_filename(urlparse(u).path or "")
        ]
        if not http_urls:
            keep.append(rec)
            continue
        landing = rec.landing_url if rec.landing_url in http_urls else http_urls[0]
        result = probe(landing)
        if result.alive:
            if not (result.reason or "").startswith("skip:"):
                rec.link_ok = True
                rec.checked_at = when
            keep.append(rec)
            continue
        dropped_nodes.append(rec)
        dropped_rows.append(
            {
                "output_id": rec.output_id,
                "url": landing,
                "status": result.status,
                "checked_at": when,
                "reason": result.reason,
            }
        )

    drop_ids = {r.output_id for r in dropped_nodes}
    new_links = [
        ln
        for ln in graph.links
        if ln.source not in drop_ids and (not ln.target_is_node or ln.target not in drop_ids)
    ]
    graph.nodes = keep
    graph.links = new_links

    if dropped_path is not None and dropped_rows:
        dropped_path.parent.mkdir(parents=True, exist_ok=True)
        with dropped_path.open("a", encoding="utf-8") as fh:
            for row in dropped_rows:
                fh.write(json.dumps(row) + "\n")
    return graph, dropped_rows


def graph_from_json(payload: dict) -> AtlasGraph:
    from cancer_output_atlas.pipeline import records_from_json
    from cancer_output_atlas.schema import POLICY, Link

    nodes = records_from_json(payload)
    links = [
        Link(
            source=raw["source"],
            rel=raw["rel"],
            target=raw["target"],
            evidence=raw.get("evidence") or "",
            target_is_node=bool(raw.get("target_is_node", True)),
        )
        for raw in payload.get("links") or []
    ]
    return AtlasGraph(
        goal=payload.get("goal") or "",
        generated_at=payload.get("generated_at") or _now(),
        nodes=nodes,
        links=links,
        skipped=list(payload.get("skipped") or []),
        policy=dict(payload.get("policy") or POLICY),
    )

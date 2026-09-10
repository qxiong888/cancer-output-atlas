"""Demo page + find API. Rank a baked graph only; no live ingest.

When strands-agents is installed, GET /api/find constructs official
strands.Agent with a find tool wrapping find_by_goal (no Bedrock AgentCore).
Ranking is always find_by_goal. Public product name: Cancer Output Atlas.
"""

from __future__ import annotations

import json
import os
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from cancer_output_atlas.find import find_by_goal, load_graph_records, write_goal_find_json
from cancer_output_atlas.runtime import STRANDS_AVAILABLE, STRANDS_VERSION

GOAL_A = (
    "Find public NSCLC pembrolizumab / Keytruda immunotherapy resources I can reuse"
)

NOT_FOUND_HTML = (
    "<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"utf-8\">"
    "<title>Not found</title></head><body>"
    "<p>Page not found.</p>"
    "<p><a href=\"/\">Cancer Output Atlas</a></p>"
    "</body></html>"
)


def write_demo_html(out_dir: Path, fallback: dict | None = None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    blob = "null"
    if fallback is not None:
        blob = json.dumps(fallback, ensure_ascii=False)
    tmpl = Path(__file__).with_name("demo_page.html")
    html = tmpl.read_text(encoding="utf-8")
    html = html.replace("__FALLBACK_A__", blob)
    path = out_dir / "demo.html"
    path.write_text(html, encoding="utf-8")
    return path


def _health_payload() -> dict:
    return {
        "ok": True,
        "service": os.environ.get("K_SERVICE") or "cancer-output-atlas",
        "revision": os.environ.get("K_REVISION") or "local",
    }


def _find_with_agent(records, goal: str, top: int, graph_path: Path | None) -> dict:
    graph_s = str(graph_path) if graph_path else None
    if not STRANDS_AVAILABLE:
        payload = find_by_goal(records, goal, top=top, graph_path=graph_s)
        payload["agent"] = "none"
        payload["strands_available"] = False
        payload["strands_version"] = None
        payload["tools_used"] = []
        payload["agent_trace"] = []
        return payload

    from cancer_output_atlas.agent_tools import make_find_tool
    from cancer_output_atlas.runtime import build_find_agent

    cache: dict = {"trace": [], "tools_used": []}
    find_tool = make_find_tool(records, graph_path=graph_s, cache=cache, top_default=top)
    try:
        agent = build_find_agent(tools=[find_tool])
        agent(goal or "")
    except Exception:
        pass
    payload = cache.get("payload")
    if payload is None:
        payload = find_by_goal(records, goal, top=top, graph_path=graph_s)
    payload["agent"] = "strands"
    payload["strands_available"] = True
    payload["strands_version"] = STRANDS_VERSION
    payload["tools_used"] = list(cache.get("tools_used") or [])
    payload["agent_trace"] = list(cache.get("trace") or [])
    return payload


class _Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, out_dir: Path, graph_path: Path, records=None, **kwargs):
        self._out_dir = out_dir
        self._graph_path = graph_path
        self._records = records or []
        self._demo_html = (out_dir / "demo.html").read_bytes() if (out_dir / "demo.html").is_file() else b""
        super().__init__(*args, directory=str(out_dir), **kwargs)

    def log_message(self, fmt: str, *args) -> None:
        print(f"serve: {args[0] if args else fmt}")

    def do_GET(self) -> None:  # noqa: N802
        self._dispatch(write_body=True)

    def do_HEAD(self) -> None:  # noqa: N802
        self._dispatch(write_body=False)

    def _dispatch(self, *, write_body: bool) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/health":
            return self._send_bytes(
                200,
                "application/json; charset=utf-8",
                json.dumps(_health_payload(), ensure_ascii=False).encode("utf-8"),
                write_body=write_body,
            )
        if path in {"/", "/index.html", "/demo.html"}:
            body = self._demo_html
            if not body:
                demo = self._out_dir / "demo.html"
                body = demo.read_bytes() if demo.is_file() else b""
            if not body:
                return self._send_product_404(write_body=write_body)
            return self._send_bytes(200, "text/html; charset=utf-8", body, write_body=write_body)
        if path == "/api/find":
            return self._api_find(parse_qs(parsed.query), write_body=write_body)
        return self._send_product_404(write_body=write_body)

    def _send_bytes(self, code: int, content_type: str, body: bytes, *, write_body: bool) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if write_body:
            self.wfile.write(body)

    def _send_json(self, code: int, payload: dict, *, write_body: bool = True) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send_bytes(code, "application/json; charset=utf-8", body, write_body=write_body)

    def _send_product_404(self, *, write_body: bool) -> None:
        self._send_bytes(
            404,
            "text/html; charset=utf-8",
            NOT_FOUND_HTML.encode("utf-8"),
            write_body=write_body,
        )

    def send_error(self, code, message=None, explain=None):
        if code == 404:
            return self._send_product_404(write_body=self.command != "HEAD")
        return super().send_error(code, message, explain)

    def _api_find(self, qs: dict[str, list[str]], *, write_body: bool) -> None:
        raw = unquote((qs.get("goal") or [""])[0] or "")
        goal = raw.strip()
        try:
            top = int((qs.get("top") or ["40"])[0])
        except ValueError:
            top = 40
        top = max(1, min(top, 80))
        records = getattr(self, "_records", None)
        if not records:
            if not self._graph_path.is_file():
                err = {"error": "link_graph.json missing", "ok": False}
                return self._send_json(404, err, write_body=write_body)
            records = load_graph_records(self._graph_path)
        try:
            payload = _find_with_agent(records, goal, top, self._graph_path)
        except Exception as exc:
            return self._send_json(500, {"error": f"find failed: {exc}"}, write_body=write_body)
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send_bytes(200, "application/json; charset=utf-8", body, write_body=write_body)
        if write_body:
            try:
                write_goal_find_json(payload, self._out_dir / "goal_find.json")
            except OSError:
                pass


def serve(out_dir: Path, *, graph_path: Path | None = None, host: str = "0.0.0.0", port: int = 8080) -> None:
    out_dir = out_dir.resolve()
    graph_path = (graph_path or (out_dir / "link_graph.json")).resolve()
    port = int(os.environ.get("PORT", port))
    fallback = None
    a_json = out_dir / "goal_find_A.json"
    if a_json.is_file():
        try:
            fallback = json.loads(a_json.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            fallback = None
    write_demo_html(out_dir, fallback)
    records = load_graph_records(graph_path) if graph_path.is_file() else []
    handler = partial(_Handler, out_dir=out_dir, graph_path=graph_path, records=records)
    httpd = ThreadingHTTPServer((host, port), handler)
    print(f"demo: http://{host}:{port}/")
    print(f"file: {out_dir / 'demo.html'}")
    print(f"graph: {graph_path} nodes={len(records)}")
    httpd.serve_forever()

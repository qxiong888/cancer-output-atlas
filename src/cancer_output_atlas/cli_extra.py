
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def cmd_find(args: argparse.Namespace) -> int:
    from cancer_output_atlas.find import (
        find_from_graph,
        format_find_text,
        write_goal_find_json,
        write_goal_find_xlsx,
    )

    graph = Path(args.graph)
    if not graph.is_file():
        print("find does not re-ingest; graph missing: " + str(graph), file=sys.stderr)
        print("first run: python -m cancer_output_atlas run --offline --out out", file=sys.stderr)
        return 2
    payload = find_from_graph(graph, args.goal, top=args.top)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    xlsx = Path(args.xlsx) if args.xlsx else out / "goal_find.xlsx"
    jpath = Path(args.json_path) if args.json_path else out / "goal_find.json"
    write_goal_find_json(payload, jpath)
    try:
        write_goal_find_xlsx(payload, xlsx)
        print("wrote " + str(xlsx))
    except RuntimeError as exc:
        print("xlsx skipped: " + str(exc), file=sys.stderr)
    print("wrote " + str(jpath))
    print(format_find_text(payload), end="")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from cancer_output_atlas.serve import serve

    serve(
        Path(args.out),
        graph_path=Path(args.graph) if args.graph else None,
        host=args.host,
        port=args.port,
    )
    return 0


def add_find_and_serve(sub: argparse._SubParsersAction) -> None:
    findp = sub.add_parser("find", help="rank an existing graph by goal, grouped by ODS")
    findp.add_argument("--goal", required=True)
    findp.add_argument("--graph", default="out/link_graph.json")
    findp.add_argument("--top", type=int, default=8)
    findp.add_argument("--by-ods", action="store_true", default=True)
    findp.add_argument("--out", default="out")
    findp.add_argument("--xlsx", default=None)
    findp.add_argument("--json", dest="json_path", default=None)
    findp.set_defaults(func=cmd_find)

    srv = sub.add_parser("serve", help="demo page + find API (baked graph only)")
    srv.add_argument("--out", default="out")
    srv.add_argument("--graph", default=None)
    srv.add_argument("--host", default="0.0.0.0")
    srv.add_argument("--port", type=int, default=8080)
    srv.set_defaults(func=cmd_serve)

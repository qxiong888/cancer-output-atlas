"""Command-line entry: run, rank, agent."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from cancer_output_atlas.classify import GOAL_DEFAULT
from cancer_output_atlas.cli_extra import add_find_and_serve
from cancer_output_atlas.pipeline import records_from_json, run_atlas, write_artifacts
from cancer_output_atlas.rank import format_rank_table, rank_records
from cancer_output_atlas.runtime import STRANDS_AVAILABLE, STRANDS_IMPORT_ERROR, STRANDS_VERSION


def _cmd_run(args: argparse.Namespace) -> int:
    if args.offline:
        os.environ["COA_OFFLINE"] = "1"
    check_live = bool(getattr(args, "force_check_links", False))
    if getattr(args, "check_links", False) and not args.offline:
        check_live = True
    if args.offline and not getattr(args, "force_check_links", False):
        check_live = False
    graph, ranked = run_atlas(
        goal=args.goal,
        use_offline=args.offline or None,
        check_live=check_live,
    )
    out = Path(args.out)
    paths = write_artifacts(graph, ranked, out)
    print(f"wrote {paths['classification_table']}")
    print(f"wrote {paths['link_graph']}")
    print(f"wrote {paths['digest']}")
    print(f"wrote {paths['ranked']}")
    print(format_rank_table(ranked[: args.top]), end="")
    return 0


def _cmd_rank(args: argparse.Namespace) -> int:
    path = Path(args.table or args.graph)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list) and payload and "output_id" in payload[0] and "identifiers" not in payload[0]:
        # ranked or classification table — re-print by profile_fit
        rows = sorted(payload, key=lambda r: (-float(r.get("profile_fit") or r.get("rank_score") or 0), r.get("output_id") or ""))
        for i, row in enumerate(rows[: args.top], start=1):
            row["rank"] = i
            row.setdefault("rank_score", row.get("profile_fit") or 0)
            row.setdefault("ids", [])
            row.setdefault("kind", "")
            row.setdefault("title", "")
        print(format_rank_table(rows[: args.top]), end="")
        return 0
    records = records_from_json(payload)
    ranked = rank_records(records, goal=args.goal, top=args.top)
    print(format_rank_table(ranked), end="")
    return 0


def _cmd_agent(args: argparse.Namespace) -> int:
    from cancer_output_atlas.agent_tools import SYSTEM_PROMPT, tool_list
    from cancer_output_atlas.runtime import build_agent

    print(
        f"strands_available={STRANDS_AVAILABLE} version={STRANDS_VERSION} "
        f"import_error={STRANDS_IMPORT_ERROR}"
    )
    if args.offline:
        os.environ["COA_OFFLINE"] = "1"
    graph, ranked = run_atlas(goal=args.goal, use_offline=args.offline or None)
    paths = write_artifacts(graph, ranked, Path(args.out))
    print(f"pipeline artifacts in {args.out}")
    if args.no_llm:
        print("skipped Agent() construction (--no-llm). tools ran as functions.")
        print(f"digest: {paths['digest']}")
        return 0
    model = args.model or os.environ.get("STRANDS_MODEL") or None
    try:
        agent = build_agent(tools=tool_list(), system_prompt=SYSTEM_PROMPT, model=model)
    except Exception as exc:
        print(f"Agent() not constructed: {exc}", file=sys.stderr)
        return 2
    prompt = (
        f"Summarize this atlas run for goal {args.goal!r}. "
        f"Use only these observed IDs and do not invent any. "
        f"Ranked JSON: {json.dumps(ranked[: args.top])}"
    )
    result = agent(prompt)
    print(result)
    return 0


def _cmd_info(_: argparse.Namespace) -> int:
    print(f"strands_available={STRANDS_AVAILABLE}")
    print(f"strands_version={STRANDS_VERSION}")
    print(f"strands_import_error={STRANDS_IMPORT_ERROR}")
    print(f"default_goal={GOAL_DEFAULT}")
    return 0


def _cmd_check_links(args: argparse.Namespace) -> int:
    from cancer_output_atlas.check_links import check_graph_links, graph_from_json
    from cancer_output_atlas.rank import rank_records

    graph_path = Path(args.graph)
    payload = json.loads(graph_path.read_text(encoding="utf-8"))
    graph = graph_from_json(payload)
    out = Path(args.out)
    dropped_path = out / "dropped_links.jsonl"
    graph, dropped = check_graph_links(
        graph,
        dropped_path=dropped_path,
        skip_if_offline=False if args.force else True,
    )
    ranked = rank_records(graph.nodes, goal=graph.goal, query_first=False)
    paths = write_artifacts(graph, ranked, out)
    print(f"checked landings; dropped {len(dropped)} nodes")
    print(f"wrote {paths['link_graph']}")
    if dropped:
        print(f"appended {dropped_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="coa",
        description="Cancer Output Atlas — classify and link public research outputs.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="unattended fetch/classify/link (no LLM)")
    run.add_argument("--goal", default=GOAL_DEFAULT)
    run.add_argument("--out", default="out")
    run.add_argument("--offline", action="store_true", help="use fixtures; no network")
    run.add_argument("--top", type=int, default=10)
    run.add_argument(
        "--check-links",
        action="store_true",
        help="probe landing URLs after ingest (skipped offline unless --force-check-links)",
    )
    run.add_argument("--force-check-links", action="store_true")
    run.set_defaults(func=_cmd_run)

    rank = sub.add_parser("rank", help="rank an existing table or graph by goal")
    rank.add_argument("--table", default=None, help="classification_table.json")
    rank.add_argument("--graph", default=None, help="link_graph.json")
    rank.add_argument("--goal", default=GOAL_DEFAULT)
    rank.add_argument("--top", type=int, default=10)
    rank.set_defaults(func=_cmd_rank)

    agent = sub.add_parser("agent", help="run pipeline; optionally construct strands.Agent")
    agent.add_argument("--goal", default=GOAL_DEFAULT)
    agent.add_argument("--out", default="out")
    agent.add_argument("--offline", action="store_true")
    agent.add_argument("--no-llm", action="store_true", help="do not construct Agent()")
    agent.add_argument("--model", default=None, help="optional official Strands model id")
    agent.add_argument("--top", type=int, default=8)
    agent.set_defaults(func=_cmd_agent)

    info = sub.add_parser("info", help="show runtime / strands status")
    info.set_defaults(func=_cmd_info)

    chk = sub.add_parser("check-links", help="drop stale landing URLs from a graph")
    chk.add_argument("--graph", default="out/link_graph.json")
    chk.add_argument("--out", default="out")
    chk.add_argument(
        "--force",
        action="store_true",
        help="probe even when COA_OFFLINE=1 (still no payload download)",
    )
    chk.set_defaults(func=_cmd_check_links)
    add_find_and_serve(sub)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "rank" and not args.table and not args.graph:
        print("rank requires --table or --graph", file=sys.stderr)
        return 2
    return args.func(args)


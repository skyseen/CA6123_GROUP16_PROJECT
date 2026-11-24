"""Run an HDB research session from the terminal."""
import argparse
import json
from pathlib import Path
import re
import sys
from uuid import uuid4

from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.sqlite import SqliteStore
from langgraph.types import Command

from .agents import GeminiAgents
from .graph import RECURSION_LIMIT, Settings, build_graph, prepare_turn
from .guards import inspect_input, redact
from .observability import enable_phoenix, phoenix_enabled
from .onemap import OneMap
from .policy import index_rules
from .state import BuyerProfile


def decision_for(gate: dict, *, auto: bool) -> dict:
    kind = gate["type"]
    if auto:
        return {"action": {"refresh_data": "skip", "clarify": "skip",
                           "confirm_plan": "approve", "approve_output": "reject"}[kind]}
    print(f"\n{kind.replace('_', ' ').title()}", file=sys.stderr)
    for key, value in gate.items():
        if key not in {"type", "options"} and value:
            print(f"{key.replace('_', ' ')} = {value}", file=sys.stderr)
    options = gate["options"]
    while True:
        action = input(f"[{' / '.join(options)}] ").strip().lower()
        if action not in options:
            print("Choose one of the displayed options.", file=sys.stderr)
            continue
        payload = {"action": action}
        if kind == "clarify" and action == "answer":
            payload["text"], events = inspect_input(input("Details: "))
            payload["input_events"] = events
        if kind == "confirm_plan":
            if action == "edit":
                try:
                    edits = json.loads(input('Filter changes as JSON, e.g. {"max_price": 650000}: '))
                    if not isinstance(edits, dict) or not edits.keys() <= {
                        "months_back", "max_price", "flat_type", "min_est_lease_years",
                        "max_commute_min", "destination", "avoid_floor", "towns"
                    }:
                        raise ValueError("Unknown filter")
                    BuyerProfile.model_validate(edits)
                except ValueError:
                    print("Enter a JSON object with valid filter field names and values.", file=sys.stderr)
                    continue
                payload["changes"] = {key: redact(value)[0] if isinstance(value, str) else value
                                      for key, value in edits.items()}
            if action in ("approve", "edit"):
                payload["remember"] = input("Remember the town, floor and commute destination? [y/N] ").strip().lower() == "y"
        if kind == "approve_output" and action == "edit":
            payload["summary"] = redact(input("New summary (figures must match the report): "))[0]
        return payload


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("brief", nargs="?")
    parser.add_argument("--offline", action="store_true", help="Use the local parser, data, rules and route cache")
    parser.add_argument("--auto", action="store_true", help="Demo mode. Confirm the plan and decline refresh or export.")
    parser.add_argument("--mcp", action="store_true", help="Use local MCP transport for routes and approved exports")
    parser.add_argument("--save", action="store_true", help="Request an approved local JSON export")
    parser.add_argument("--trace", action="store_true", help="Export traces to a local Arize Phoenix server")
    parser.add_argument("--thread", default=None, help="Resume or continue this checkpoint thread")
    parser.add_argument("--user", default="local", help="Local preference namespace identifier")
    parser.add_argument("--db", type=Path, default=Path("data/hdb.db"))
    parser.add_argument("--session-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    for key, value in (("--user", args.user), ("--thread", args.thread)):
        if value and (not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", value) or redact(value)[1]):
            parser.error(f"{key} must be a short local label without personal identifiers")
    if args.save and args.auto:
        parser.error("--save requires interactive approval")
    thread = args.thread or uuid4().hex
    if phoenix_enabled(args.trace):
        url = enable_phoenix()
        print(f"Tracing to {url}" if url else "Phoenix packages are not installed; tracing is off.", file=sys.stderr)
    try:
        model = None if args.offline else GeminiAgents.from_env()
    except ValueError as exc:
        parser.error(str(exc))
    args.session_dir.mkdir(parents=True, exist_ok=True)
    settings = Settings(args.db, model=model, vectorstore=index_rules(args.session_dir / "chroma"),
                        onemap=None if args.offline else OneMap.from_env(), online=not args.offline, mcp=args.mcp,
                        log=Path("logs/run.jsonl"), export=args.session_dir / "shortlists")
    # Namespacing checkpoints by user keeps two local preference profiles isolated.
    config = {"configurable": {"thread_id": f"{args.user}/{thread}", "user_id": args.user},
              "recursion_limit": RECURSION_LIMIT}
    print(f"Thread {thread}", file=sys.stderr)
    with SqliteSaver.from_conn_string(str(args.session_dir / "ckpt.db")) as saver, \
            SqliteStore.from_conn_string(str(args.session_dir / "memory.db")) as store:
        store.setup()
        graph = build_graph(settings, saver, store)
        snapshot = graph.get_state(config)
        pending = [item for task in snapshot.tasks for item in task.interrupts]
        try:
            if pending:
                result = graph.invoke(Command(resume=decision_for(pending[0].value, auto=args.auto)), config)
            else:
                brief = args.brief if args.brief is not None else input("Research brief: ")
                if not brief.strip():
                    parser.error("Enter a research brief")
                result = graph.invoke(prepare_turn(brief, save=args.save), config)
            while result.get("__interrupt__"):
                decision = decision_for(result["__interrupt__"][0].value, auto=args.auto)
                result = graph.invoke(Command(resume=decision), config)
        except (EOFError, KeyboardInterrupt):
            print(f"\nSession paused. Resume with --thread {thread} and the same --user.", file=sys.stderr)
            return 130
    print(result["canonical_report"], end="")
    if result.get("saved_path"):
        print(f"Saved {result['saved_path']}", file=sys.stderr)
    if "model_fallback" in result["guard_events"]:
        print("A model call failed and the local fallback completed its step.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

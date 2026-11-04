"""Replay a Hermes session turn through KiroACPClient and measure tool-call health.

Uses real kiro-cli (each run costs Kiro credits, about 0.3 per run with Sonnet).

    ~/.hermes/hermes-agent/venv/bin/python scripts/replay_session.py --list
    ~/.hermes/hermes-agent/venv/bin/python scripts/replay_session.py SESSION_ID LAST_MESSAGE_ID --runs 10

LAST_MESSAGE_ID is the last message kept as history (pick the message just before the
assistant reply that was cut off). Per run it reports raw <tool_call> blocks, how many parsed
as-is, and how many stayed unrecoverable after this plugin's repair. Unparseable raw blocks are
printed so the new failure shape can be added to repair_tool_calls and the tests.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

HERMES_HOME = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))
HERMES_REPO = Path(os.environ.get("HERMES_REPO", HERMES_HOME / "hermes-agent"))
PLUGIN_DIR = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(HERMES_REPO), str(PLUGIN_DIR)]

import kiro_acp_client as kiro  # noqa: E402

TOOLS = [
    {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": props, "required": list(props)}}}
    for name, desc, props in [
        ("terminal", "Run a shell command", {"command": {"type": "string"}}),
        ("read_file", "Read a file", {"path": {"type": "string"}}),
        ("search_files", "Regex search in files", {"pattern": {"type": "string"}, "path": {"type": "string"}}),
    ]
]


def load_messages(db: sqlite3.Connection, session_id: str, last_id: int) -> list[dict]:
    cols = [row[1] for row in db.execute("pragma table_info(messages)")]
    messages = [{"role": "system", "content": "You are Hermes, a helpful agent. Use the provided tools."}]
    for row in db.execute("select * from messages where session_id=? and id<=? order by id", (session_id, last_id)):
        rec = dict(zip(cols, row))
        msg = {"role": rec["role"], "content": rec["content"] or ""}
        if rec["role"] == "tool":
            msg["tool_call_id"] = rec.get("tool_call_id")
        messages.append(msg)
    return messages


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session_id", nargs="?")
    ap.add_argument("last_message_id", nargs="?", type=int)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--model", default="claude-sonnet-5.5")
    ap.add_argument("--list", action="store_true", help="list recent sessions and their kiro-acp replies")
    args = ap.parse_args()
    db = sqlite3.connect(HERMES_HOME / "state.db")

    if args.list or not args.session_id:
        for sid, source in db.execute("select id, source from sessions order by started_at desc limit 10"):
            print(f"{sid}  ({source})")
            for mid, content in db.execute(
                    "select id, substr(replace(coalesce(content,''), char(10), ' '), 1, 90) from messages "
                    "where session_id=? and role='assistant' order by id desc limit 3", (sid,)):
                print(f"    {mid}: {content}")
        return 0

    stats = {"blocks": 0, "raw_valid": 0, "unrecoverable": 0}
    original = kiro.repair_tool_calls

    def spy(text: str):
        result = original(text)
        for _name, body in kiro._TOOL_CALL_RE.findall(text):
            stats["blocks"] += 1
            try:
                json.loads(body.strip())
                stats["raw_valid"] += 1
            except ValueError as exc:
                print(f"RAW-INVALID ({exc}): ...{body.strip()[-120:]!r}")
        stats["unrecoverable"] += result[1]
        return result

    kiro.repair_tool_calls = spy
    messages = load_messages(db, args.session_id, args.last_message_id)
    client = kiro.KiroACPClient()
    with_calls = 0
    for i in range(args.runs):
        reply = client.chat.completions.create(model=args.model, messages=messages, tools=TOOLS, timeout=300)
        calls = reply.choices[0].message.tool_calls or []
        with_calls += bool(calls)
        print(f"run {i}: {len(calls)} tool call(s), finish={reply.choices[0].finish_reason}")
    print(f"runs={args.runs} turns_with_tool_call={with_calls} {stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

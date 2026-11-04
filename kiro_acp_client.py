"""Kiro ACP client — thin adaptation of Hermes' in-tree Copilot ACP shim.

The upstream ``agent/copilot_acp_client.py`` is a generic JSON-RPC/ACP-over-stdio
bridge parameterized by command + args; only its env-var resolution and error
copy are Copilot-specific. This subclass points those at ``kiro-cli acp``
(Kiro CLI from AWS, kiro.dev) and keeps everything else — session lifecycle,
``session/update`` streaming, permission handling, the file-safety bridge —
inherited, so fixes upstream flow through.

kiro-cli specifics verified against v2.22.0:
- handshake answers ``initialize`` (protocolVersion 1, loadSession, image prompts)
- spawn flags: ``--agent`` / ``--model`` / ``--effort`` / ``-a`` (trust tools)
- auth rides the subprocess (~/.kiro login) — no env keys to manage
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
import re
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent import acp_client as _acp
from agent.acp_openai_bridge import TOOL_CALL_CONTRACT
from agent.copilot_acp_client import ACP_MARKER_BASE_URL as _COPILAT_MARKER, CopilotACPClient

logger = logging.getLogger(__name__)
_turn_context = contextvars.ContextVar("kiro_console_turn", default=None)

KIRO_MARKER_BASE_URL = "acp://kiro"

# Second cause of cut-off turns: ``kiro_default`` gives the model Kiro's own tools (read, shell, ...).
# The model sometimes calls those natively instead of writing a <tool_call> block; Hermes answers
# Kiro's permission request with "cancelled", Kiro ends the turn with stopReason "refusal", and only
# the narration before the call reaches Hermes (measured: 1 of 4 replays). A dedicated agent with no
# tools and no MCP servers leaves the <tool_call> text as the model's only way to act.
KIRO_AGENT_NAME = "hermes-bridge"
DEFAULT_KIRO_ARGS = ("acp", "--agent", KIRO_AGENT_NAME)
_KIRO_AGENT_CONFIG = {
    "name": KIRO_AGENT_NAME,
    "description": "Text-only agent for Hermes: no Kiro tools; Hermes runs tools from <tool_call> blocks.",
    "tools": [],
    "includeMcpJson": False,
    "resources": [],
}

# Root cause of cut-off turns: Hermes' bridge contract asks for a three-level object
# ``{"id","type","function":{"name","arguments":"<JSON string>"}}``. The command inside the
# arguments string is escaped twice, and the block ends in a run of closers; Kiro models drop a
# quote or a brace there (measured: 7 of 10 calls malformed even with inline object arguments).
# The bridge drops an unparseable block, so the turn ends on the narration before it.
# Fix: Kiro gets a flat format with no nesting and single escaping,
#   <tool_call name="terminal">{"command": "ls"}</tool_call>
# and this module converts it to the bridge's canonical block before Hermes parses it.
# Lost trailing braces / quotes are still repaired, and one retry runs if a block stays broken.
_KIRO_TOOL_CONTRACT = (
    "Available tools (JSON schema list below). To call a tool, emit "
    '<tool_call name="TOOL_NAME">ARGUMENTS</tool_call> where ARGUMENTS is one JSON object holding the '
    "tool's parameters. Emit one block per call; several blocks are allowed. "
    'Example: <tool_call name="terminal">{"command": "ls -la"}</tool_call>'
)
_KIRO_PREAMBLE_LINE = (
    "IMPORTANT: If you take an action with a tool, you MUST output it as a "
    '<tool_call name="TOOL_NAME">{...arguments...}</tool_call> block as described below.'
)
# A block ends at its closer, at the next block (a forgotten closer), or at the end of the reply.
_TOOL_CALL_RE = re.compile(
    r'<tool_call(?:\s+name\s*=\s*"([^"]*)")?\s*>(.*?)(?:</tool_call>|(?=<tool_call[\s>])|\Z)', re.DOTALL)
_ESCAPE_RE = re.compile(r"\\(.)", re.DOTALL)
# Claude sometimes trails its native tool-call markup (``</parameter></invoke>``) after the JSON.
_CLOSING_TAG_RE = re.compile(r"</[\w:.-]+>")
_FIELD_RE = {key: re.compile(rf'"{key}"\s*:\s*"([^"\\]+)"') for key in ("id", "name")}
_ARGUMENTS_RE = re.compile(r'"arguments"\s*:\s*')
_DECODER = json.JSONDecoder()
_RETRY_NOTE = (
    "Your previous reply contained a <tool_call> block whose arguments were not valid JSON, so it could "
    'not be run. Reply again from the same point, using <tool_call name="TOOL_NAME">{...}</tool_call> '
    "with one complete JSON object inside."
)


def kiro_tool_prompt(prompt_text: str) -> str:
    """Swap Hermes' nested tool-call contract (and its preamble line) for the flat Kiro format."""
    if TOOL_CALL_CONTRACT not in prompt_text:
        return prompt_text  # no tools in this prompt, or upstream reworded the contract
    prompt_text = prompt_text.replace(TOOL_CALL_CONTRACT, _KIRO_TOOL_CONTRACT, 1)
    for line in _acp._PROMPT_PREAMBLE:
        if "<tool_call>" in line:
            prompt_text = prompt_text.replace(line, _KIRO_PREAMBLE_LINE, 1)
    return prompt_text


def _fix_escapes(text: str) -> str:
    """Double backslashes that start no valid JSON escape (``\\|`` from shell regexes)."""
    return _ESCAPE_RE.sub(lambda m: m.group(0) if m.group(1) in '"\\/bfnrtu' else "\\\\" + m.group(1), text)


def _object_and_rest(text: str) -> tuple[dict[str, Any], str] | None:
    """JSON object at the start of ``text`` plus the text after it, minus stray closers and leaked
    closing tags; tolerates up to two lost ``}``."""
    text = text.strip()
    for fixed in (text, _fix_escapes(text)):
        for candidate in (fixed, fixed + "}", fixed + "}}"):
            try:
                obj, end = _DECODER.raw_decode(candidate)
            except ValueError:
                continue
            if isinstance(obj, dict):
                return obj, _CLOSING_TAG_RE.sub("", candidate[end:]).strip(' \t\r\n}"')
    return None


def _leading_object(text: str) -> dict[str, Any] | None:
    """JSON object that makes up all of ``text`` (see ``_object_and_rest``)."""
    found = _object_and_rest(text)
    return found[0] if found is not None and not found[1] else None


def _salvage_arguments(rest: str) -> dict[str, Any] | None:
    """Arguments object from the text after ``"arguments":``, string-encoded or inline."""
    if rest.startswith("{"):
        return _leading_object(rest)
    if not rest.startswith('"'):
        return None
    body = rest[1:]
    # Cut a few trailing characters at most: the damage is a lost quote or brace at the end.
    for cut in range(len(body), max(len(body) - 8, 0), -1):
        for candidate in (body[:cut], _fix_escapes(body[:cut])):
            try:
                decoded = json.loads(f'"{candidate}"')
            except ValueError:
                continue
            if (args := _leading_object(decoded)) is not None:
                return args
    return None


def _block(name: str, args: dict[str, Any], call_id: str | None = None) -> str:
    payload: dict[str, Any] = {
        "type": "function", "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}
    if call_id:
        payload = {"id": call_id, **payload}
    return f"<tool_call>{json.dumps(payload, ensure_ascii=False)}</tool_call>"


def _canonical_block(body: str) -> str | None:
    """Repaired bridge-format block for ``body``: None when it already parses, "" when unrecoverable.

    Only the block's own JSON is repaired; a parseable block passes through even when its
    ``arguments`` string is bad JSON."""
    try:
        obj = json.loads(body)
    except ValueError:
        obj = None
    if isinstance(obj, dict) and isinstance(fn := obj.get("function"), dict):
        name = fn.get("name")
        if isinstance(name, str) and name.strip():
            return None
    name = _FIELD_RE["name"].search(body)
    marker = _ARGUMENTS_RE.search(body)
    if name is None or marker is None:
        return ""
    args = _salvage_arguments(body[marker.end():].strip())
    if args is None:
        return ""
    call_id = _FIELD_RE["id"].search(body)
    return _block(name.group(1), args, call_id.group(1) if call_id else None)


def repair_tool_calls(text: str) -> tuple[str, int]:
    """Convert Kiro's ``<tool_call name=...>`` blocks to the bridge format and repair malformed
    bridge-format blocks → ``(text, unrecoverable_count)``.

    Parseable closed bridge-format blocks are left byte-for-byte; an unclosed trailing one gets its closer."""
    broken = 0

    def _replace(match: re.Match[str]) -> str:
        nonlocal broken
        original, name, body = match.group(0), (match.group(1) or "").strip(), match.group(2).strip()
        if not body.startswith("{"):  # a prose mention of the tag, not a call
            return original
        if name:
            # Prose the model wrote after a call it forgot to close stays as text after the block.
            found = _object_and_rest(body)
            block = "" if found is None else _block(name, found[0]) + (f"\n{found[1]}" if found[1] else "")
        else:
            block = _canonical_block(body)
        if block == "":
            broken += 1
            return original
        if block is None:
            return original if original.endswith("</tool_call>") else original + "</tool_call>"
        return block

    return _TOOL_CALL_RE.sub(_replace, text), broken


def resolve_kiro_command() -> str:
    """Binary override order: HERMES_KIRO_ACP_COMMAND → KIRO_CLI_PATH → kiro-cli."""
    return (
        os.getenv("HERMES_KIRO_ACP_COMMAND", "").strip()
        or os.getenv("KIRO_CLI_PATH", "").strip()
        or "kiro-cli"
    )


def resolve_kiro_args() -> list[str]:
    """Argv override (shlex-split), else ``DEFAULT_KIRO_ARGS`` (the tool-less agent). Extra flags like
    ``--agent NAME`` / ``--effort high`` / ``-a`` can be passed wholesale."""
    return shlex.split(os.getenv("HERMES_KIRO_ACP_ARGS", "").strip()) or list(DEFAULT_KIRO_ARGS)


def ensure_kiro_agent(args: list[str], home: str | None = None) -> None:
    """Write the tool-less agent config in the selected HOME when needed."""
    if not any(a == "--agent" and b == KIRO_AGENT_NAME for a, b in zip(args, args[1:])) \
            and f"--agent={KIRO_AGENT_NAME}" not in args:
        return
    path = (Path(home) if home else Path.home()) / ".kiro" / "agents" / f"{KIRO_AGENT_NAME}.json"
    text = json.dumps(_KIRO_AGENT_CONFIG, indent=2) + "\n"
    try:
        if path.is_file() and path.read_text(encoding="utf-8") == text:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    except OSError as exc:  # kiro-cli then reports the unknown agent itself
        logger.warning("Could not write Kiro agent config %s: %s", path, exc)


class KiroACPClient(CopilotACPClient):
    """OpenAI-client-compatible facade driving ``kiro-cli acp`` over stdio."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        acp_command: str | None = None,
        acp_args: list[str] | None = None,
        command: str | None = None,
        args: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            api_key=api_key,
            base_url=base_url or KIRO_MARKER_BASE_URL,
            acp_command=acp_command or command or resolve_kiro_command(),
            acp_args=list(acp_args or args or resolve_kiro_args()),
            **kwargs,
        )
        self._seat_warning_logged = False

    def _active_seat(self):
        root = Path(os.environ.get("KIRO_CONSOLE_HOME") or Path.home() / ".kiro-console")
        try:
            data = json.loads((root / "active-seat.json").read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("expected an object")
            home = data.get("home")
            if not isinstance(home, str) or not home or not Path(home).is_dir():
                return root, "default", None
            seat = data.get("seat")
            return root, seat if isinstance(seat, str) and seat else "default", home
        except FileNotFoundError:
            return root, "default", None
        except (OSError, ValueError, UnicodeError) as exc:
            if not self._seat_warning_logged:
                logger.warning("Could not read kiro-console active seat; using default: %s", exc)
                self._seat_warning_logged = True
            return root, "default", None

    def _create_chat_completion(self, *, model=None, messages=None, **kwargs):
        # The provider name is not a model id; treat it as "use the session
        # default" instead of attempting a bogus model selection (which the
        # inherited code would log a Copilot-flavored warning for).
        if str(model or "").strip().lower() in ("", "kiro-acp", "kiro", "kiro-cli", "kiro-acp-agent"):
            model = None
        return super()._create_chat_completion(model=model, messages=messages, **kwargs)

    def _run_prompt(self, prompt_text: str, *, timeout_seconds: float, model: str | None = None) -> tuple[str, str]:
        token = _turn_context.set({"model": model or "", "seat": "default", "root": None})
        try:
            prompt_text = kiro_tool_prompt(prompt_text)
            text, reasoning = super()._run_prompt(prompt_text, timeout_seconds=timeout_seconds, model=model)
            repaired, broken = repair_tool_calls(text)
            if not broken:
                return repaired, reasoning
            logger.warning("Kiro ACP returned %d unparseable <tool_call> block(s); retrying once.", broken)
            retry_text, retry_reasoning = super()._run_prompt(
                f"{prompt_text}\n\n{_RETRY_NOTE}", timeout_seconds=timeout_seconds, model=model)
            retry_repaired, retry_broken = repair_tool_calls(retry_text)
            if retry_broken:
                logger.warning("Kiro ACP retry still returned %d unparseable <tool_call> block(s).", retry_broken)
            return retry_repaired, retry_reasoning
        finally:
            _turn_context.reset(token)

    def _handle_server_message(self, msg, **kwargs):
        if msg.get("method") == "_kiro.dev/metadata":
            try:
                params = msg.get("params")
                if isinstance(params, dict) and isinstance(params.get("meteringUsage"), list):
                    credits = next((item["value"] for item in params["meteringUsage"]
                                    if isinstance(item, dict) and item.get("unit") == "credit"
                                    and isinstance(item.get("value"), (int, float))
                                    and not isinstance(item["value"], bool)), None)
                    duration = params.get("turnDurationMs")
                    context = _turn_context.get() or {}
                    root = context.get("root")
                    if credits is not None and isinstance(duration, (int, float)) and root and root.is_dir():
                        record = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
                                  "seat": context.get("seat", "default"), "model": context.get("model", ""),
                                  "credits": credits, "turn_ms": duration}
                        fd = os.open(root / "metering.jsonl", os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
                        with os.fdopen(fd, "a", encoding="utf-8") as out:
                            out.write(json.dumps(record, separators=(",", ":")) + "\n")
            except Exception:
                logger.debug("Could not append kiro-console metering", exc_info=True)
            return True
        return super()._handle_server_message(msg, **kwargs)

    def _spawn(self):  # noqa: ANN202 - matches parent's Popen[str] return
        root, seat, home = self._active_seat()
        context = _turn_context.get()
        if context is not None:
            _turn_context.set({**context, "root": root, "seat": seat})
        ensure_kiro_agent(self._acp_args, home)
        try:
            if not home:
                return super()._spawn()
            # Mirror ACPClient._spawn only for an isolated seat: the parent builds its
            # environment internally, after Hermes may have overridden HOME.
            if _acp._acp_supported(self._acp_command, self._acp_args) is False:
                preview = " ".join(self._acp_args[:3]) if self._acp_args else "(none)"
                raise RuntimeError(
                    f"ACP transport not supported by '{self._acp_command}': `{preview}` is rejected as an unknown option. "
                    "This usually means the CLI is an older release (e.g. Claude Code v2.x) or a different tool than expected. "
                    + self.unsupported_transport_hint)
            from hermes_cli._subprocess_compat import windows_hide_flags
            env = _acp._build_subprocess_env()
            env["HOME"] = home
            try:
                proc = subprocess.Popen(
                    [self._acp_command] + self._acp_args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace", bufsize=1,
                    cwd=self._acp_cwd, env=env, creationflags=windows_hide_flags())
            except FileNotFoundError as exc:
                raise RuntimeError(
                    f"Could not start {self.label} command '{self._acp_command}'. {self._missing_command_hint()}") from exc
            if proc.stdin is None or proc.stdout is None:
                proc.kill()
                raise RuntimeError(f"{self.label} process did not expose stdin/stdout pipes.")
            with self._active_process_lock:
                self._active_processes.add(proc)
                self.is_closed = False
            return proc
        except RuntimeError as exc:
            # Translate the Copilot-flavored install hint into a Kiro one.
            if "Could not start Copilot ACP command" in str(exc):
                raise RuntimeError(
                    f"Could not start Kiro ACP command '{self._acp_command}'. Install Kiro CLI "
                    "(kiro.dev/docs/cli) and log in, or set HERMES_KIRO_ACP_COMMAND/KIRO_CLI_PATH."
                ) from exc
            raise

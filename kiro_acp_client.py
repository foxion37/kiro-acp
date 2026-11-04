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

import json
import logging
import os
import re
import shlex
from typing import Any

from agent.copilot_acp_client import ACP_MARKER_BASE_URL as _COPILAT_MARKER, CopilotACPClient

logger = logging.getLogger(__name__)

KIRO_MARKER_BASE_URL = "acp://kiro"

# Hermes' shared bridge drops a <tool_call> block whose JSON does not parse, leaving only
# the narration before it ("...확인하겠습니다.") as a finished turn. Kiro models mostly break
# the doubly-escaped ``arguments`` string, so repair that before the bridge parses it.
_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)(?:</tool_call>|\Z)", re.DOTALL)
_ESCAPE_RE = re.compile(r"\\(.)", re.DOTALL)
_FIELD_RE = {key: re.compile(rf'"{key}"\s*:\s*"([^"\\]+)"') for key in ("id", "name")}
_ARGUMENTS_RE = re.compile(r'"arguments"\s*:\s*')
_DECODER = json.JSONDecoder()
_RETRY_NOTE = (
    "Your previous reply contained a <tool_call> block that was not valid JSON, so it could not be run. "
    "Reply again from the same point. Each <tool_call> must hold one valid JSON object; "
    "function.arguments is a JSON string, so escape every quote and backslash inside it."
)


def _fix_escapes(text: str) -> str:
    """Double backslashes that start no valid JSON escape (``\\|`` from shell regexes)."""
    return _ESCAPE_RE.sub(lambda m: m.group(0) if m.group(1) in '"\\/bfnrtu' else "\\\\" + m.group(1), text)


def _leading_object(text: str) -> dict[str, Any] | None:
    """JSON object at the start of ``text``, tolerating stray closers after it."""
    for candidate in (text, _fix_escapes(text)):
        try:
            obj, end = _DECODER.raw_decode(candidate.lstrip())
        except ValueError:
            continue
        if isinstance(obj, dict) and not candidate.lstrip()[end:].strip(' \t\r\n}"'):
            return obj
    return None


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


def _canonical_block(body: str) -> str | None:
    """Repaired ``<tool_call>`` block for ``body``: None when it already parses, "" when unrecoverable.

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
    payload: dict[str, Any] = {
        "type": "function", "function": {"name": name.group(1), "arguments": json.dumps(args, ensure_ascii=False)}}
    if call_id is not None:
        payload = {"id": call_id.group(1), **payload}
    return f"<tool_call>{json.dumps(payload, ensure_ascii=False)}</tool_call>"


def repair_tool_calls(text: str) -> tuple[str, int]:
    """Rewrite malformed ``<tool_call>`` blocks as valid JSON → ``(text, unrecoverable_count)``.

    Parseable closed blocks are left byte-for-byte; an unclosed trailing block gets its closer."""
    broken = 0

    def _replace(match: re.Match[str]) -> str:
        nonlocal broken
        original, body = match.group(0), match.group(1).strip()
        if not body.startswith("{"):  # a prose mention of the tag, not a call
            return original
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
    """Argv override (shlex-split), else ``['acp']``. Extra flags like
    ``--agent NAME`` / ``--effort high`` / ``-a`` can be passed wholesale."""
    return shlex.split(os.getenv("HERMES_KIRO_ACP_ARGS", "").strip()) or ["acp"]


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

    def _create_chat_completion(self, *, model=None, messages=None, **kwargs):
        # The provider name is not a model id; treat it as "use the session
        # default" instead of attempting a bogus model selection (which the
        # inherited code would log a Copilot-flavored warning for).
        if str(model or "").strip().lower() in ("", "kiro-acp", "kiro", "kiro-cli", "kiro-acp-agent"):
            model = None
        return super()._create_chat_completion(model=model, messages=messages, **kwargs)

    def _run_prompt(self, prompt_text: str, *, timeout_seconds: float, model: str | None = None) -> tuple[str, str]:
        text, reasoning = super()._run_prompt(prompt_text, timeout_seconds=timeout_seconds, model=model)
        repaired, broken = repair_tool_calls(text)
        if not broken:
            return repaired, reasoning
        # One fresh attempt: a dropped block would end the turn on the narration before it.
        logger.warning("Kiro ACP returned %d unparseable <tool_call> block(s); retrying once.", broken)
        retry_text, retry_reasoning = super()._run_prompt(
            f"{prompt_text}\n\n{_RETRY_NOTE}", timeout_seconds=timeout_seconds, model=model)
        retry_repaired, retry_broken = repair_tool_calls(retry_text)
        if retry_broken:
            logger.warning("Kiro ACP retry still returned %d unparseable <tool_call> block(s).", retry_broken)
        return retry_repaired, retry_reasoning

    def _spawn(self):  # noqa: ANN202 - matches parent's Popen[str] return
        try:
            return super()._spawn()
        except RuntimeError as exc:
            # Translate the Copilot-flavored install hint into a Kiro one.
            if "Could not start Copilot ACP command" in str(exc):
                raise RuntimeError(
                    f"Could not start Kiro ACP command '{self._acp_command}'. Install Kiro CLI "
                    "(kiro.dev/docs/cli) and log in, or set HERMES_KIRO_ACP_COMMAND/KIRO_CLI_PATH."
                ) from exc
            raise

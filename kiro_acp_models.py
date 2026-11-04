"""Fetch Kiro's live model catalog over ACP.

``kiro-cli acp`` reports its models in the ``session/new`` result
(``models.availableModels``). This helper performs the minimal handshake —
initialize → session/new → read the model ids → terminate — without running a
prompt, so picker/model-catalog calls stay cheap (~2-4s, subprocess spawn
dominates).
"""

from __future__ import annotations

import json
import subprocess
import time
from typing import Any

from .kiro_acp_client import ensure_kiro_agent, resolve_kiro_args, resolve_kiro_command

_INITIALIZE_PARAMS = {
    "protocolVersion": 1,
    "clientCapabilities": {"fs": {"readTextFile": True, "writeTextFile": True}},
    "clientInfo": {"name": "hermes-agent", "title": "Hermes Agent", "version": "0.0.0"},
}


def list_kiro_models(timeout_seconds: float = 30.0) -> list[str] | None:
    """Model ids from a short-lived ACP session, or None when unavailable.

    Raises nothing fatal: any failure returns None so callers fall back to
    their own catalog defaults.
    """
    command = resolve_kiro_command()
    args = resolve_kiro_args()
    ensure_kiro_agent(args)
    try:
        proc = subprocess.Popen(
            [command, *args],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
    except (FileNotFoundError, OSError):
        return None
    try:
        def _request(req_id: int, method: str, params: dict[str, Any]) -> dict[str, Any] | None:
            assert proc.stdin is not None and proc.stdout is not None
            proc.stdin.write(json.dumps({"jsonrpc": "2.0", "id": req_id, "method": method, "params": params}) + "\n")
            proc.stdin.flush()
            deadline = time.monotonic() + timeout_seconds
            while time.monotonic() < deadline and proc.poll() is None:
                line = proc.stdout.readline()
                if not line:
                    time.sleep(0.05)
                    continue
                try:
                    msg = json.loads(line)
                except Exception:
                    continue
                if msg.get("id") == req_id:
                    return msg
            return None

        if _request(1, "initialize", _INITIALIZE_PARAMS) is None:
            return None
        session = _request(2, "session/new", {"cwd": "/tmp", "mcpServers": []})
        if not session or "error" in session:
            return None
        models = ((session.get("result") or {}).get("models") or {}).get("availableModels") or []
        ids = [str(m.get("modelId")) for m in models if isinstance(m, dict) and m.get("modelId")]
        return ids or None
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=2)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

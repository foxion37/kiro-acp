"""Kiro ACP provider profile for Hermes Agent.

kiro-cli does not speak OpenAI-over-HTTP: it drives an external ACP subprocess
over stdio, so the profile supplies its own client via
:meth:`ProviderProfile.create_client`. Ships as a standalone plugin per the
upstream vendor-integration policy (see the closed PRs #80362/#98151) — zero
core edits.
"""

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


def resolve_kiro_command() -> str:
    """Binary override order: HERMES_KIRO_ACP_COMMAND → KIRO_CLI_PATH → kiro-cli.

    Delegates to the client module lazily: importing it at module level drags in
    agent.copilot_acp_client and its chain, which breaks discovery under some
    import orders (hermes_cli.config → auth circular window)."""
    from .kiro_acp_client import resolve_kiro_command as _impl

    return _impl()


def resolve_kiro_args() -> list[str]:
    """Argv override (shlex-split), else ['acp']; see resolve_kiro_command."""
    from .kiro_acp_client import resolve_kiro_args as _impl

    return _impl()


class KiroACPProfile(ProviderProfile):
    """Kiro CLI (AWS) via ACP subprocess — external process, no REST endpoint."""

    _MODEL_LIST_CACHE: tuple = ()  # (timestamp, models)

    def create_client(self, **client_kwargs: Any) -> Any:
        """Build the ACP stdio shim rather than an HTTP client."""
        from .kiro_acp_client import KiroACPClient

        command = client_kwargs.get("command") or self.process_command
        args = tuple(client_kwargs.get("args") or self.process_args or ())

        return KiroACPClient(command=command, args=args, **{
            k: v for k, v in client_kwargs.items() if k not in ("command", "args")
        })

    def fetch_models(
        self, *, api_key: str | None = None, base_url: str | None = None, timeout: float = 8.0
    ) -> list[str] | None:
        """Live model catalog from the ACP subprocess (session/new reports
        models.availableModels). Cached for an hour; falls back to None on
        failure so callers use their own defaults."""
        import time as _time

        now = _time.time()
        if self._MODEL_LIST_CACHE and now - self._MODEL_LIST_CACHE[0] < 3600:
            return list(self._MODEL_LIST_CACHE[1]) or None
        try:
            from .kiro_acp_models import list_kiro_models

            models = list_kiro_models(timeout_seconds=min(max(timeout, 20.0), 60.0))
        except Exception:
            models = None
        if models:
            self.__class__._MODEL_LIST_CACHE = (now, tuple(models))
            return list(models)
        return None


kiro_acp = KiroACPProfile(
    name="kiro-acp",
    aliases=("kiro", "kiro-cli", "kiro-acp-agent"),
    display_name="Kiro (AWS)",
    description="Kiro CLI agent driven over ACP stdio",
    api_mode="chat_completions",  # ACP subprocess uses chat_completions routing
    env_vars=(),  # Auth handled by the kiro-cli subprocess itself
    base_url="acp://kiro",  # ACP internal scheme
    auth_type="external_process",
    # How to launch the CLI. Env overrides let users point at a custom binary
    # or pass extra flags (e.g. --agent / --effort) without editing the plugin.
    # The default agent is kiro_acp_client.KIRO_AGENT_NAME (tool-less, written on spawn).
    process_command="kiro-cli",
    process_args=("acp", "--agent", "hermes-bridge"),
    process_command_env_vars=("HERMES_KIRO_ACP_COMMAND", "KIRO_CLI_PATH"),
    process_args_env_var="HERMES_KIRO_ACP_ARGS",
)

register_provider(kiro_acp)

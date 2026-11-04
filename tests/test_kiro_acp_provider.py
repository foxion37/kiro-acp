"""Behavior-contract tests for the kiro-acp provider plugin.

Two layers:
- Profile/client contracts (pure, no network): registration shape, env
  override plumbing, error translation.
- Real-discovery contract: the plugin registers when dropped into an isolated
  ``$HERMES_HOME/plugins/model-providers/kiro-acp/`` — the exact path real
  users install to. No live kiro-cli process is spawned by these tests.
"""

from __future__ import annotations

import importlib
import sys
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_plugin(monkeypatch, hermes_home: Path | None = None):
    """Import the plugin exactly as Hermes' _import_plugin_dir does for user
    plugins (spec_from_file_location + submodule_search_locations), so the
    test exercises the real load path, including relative imports."""
    import importlib.util

    if hermes_home is not None:
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    module_name = "_hermes_user_provider_kiro_acp_test"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(
        module_name, REPO_ROOT / "__init__.py",
        submodule_search_locations=[str(REPO_ROOT)],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


class TestProfileContract:
    def test_registers_kiro_acp_with_external_process_auth(self, monkeypatch):
        import providers
        mod = _load_plugin(monkeypatch)
        # The module-level register_provider call in __init__ targets Hermes'
        # registry; simulate what discovery produces.
        names = [p.name for p in providers.list_providers()]
        mod.register_provider(mod.kiro_acp)
        profile = providers.get_provider_profile("kiro-acp")
        assert profile is not None
        assert profile.auth_type == "external_process"
        assert profile.process_command == "kiro-cli"
        assert profile.process_args == ("acp",)
        assert "kiro" in profile.aliases

    def test_env_overrides_replace_command_and_args(self, monkeypatch):
        mod = _load_plugin(monkeypatch)
        monkeypatch.setenv("HERMES_KIRO_ACP_COMMAND", "/opt/custom/kiro")
        monkeypatch.setenv("HERMES_KIRO_ACP_ARGS", "acp --agent builder --effort high")
        assert mod.resolve_kiro_command() == "/opt/custom/kiro"
        assert mod.resolve_kiro_args() == ["acp", "--agent", "builder", "--effort", "high"]

    def test_defaults_without_env(self, monkeypatch):
        mod = _load_plugin(monkeypatch)
        monkeypatch.delenv("HERMES_KIRO_ACP_COMMAND", raising=False)
        monkeypatch.delenv("KIRO_CLI_PATH", raising=False)
        monkeypatch.delenv("HERMES_KIRO_ACP_ARGS", raising=False)
        assert mod.resolve_kiro_command() == "kiro-cli"
        assert mod.resolve_kiro_args() == ["acp"]

    def test_create_client_returns_kiro_client_with_profile_defaults(self, monkeypatch):
        mod = _load_plugin(monkeypatch)
        client = mod.kiro_acp.create_client()
        assert type(client).__name__ == "KiroACPClient"
        assert client._acp_command == "kiro-cli"
        assert client._acp_args == ["acp"]
        assert client.base_url == "acp://kiro"


# Kiro's real failure: the arguments string lost its closing quote, so the block is not JSON.
_BROKEN_CALL = (
    r"""<tool_call>{"id":"call_1","type":"function","function":{"name":"terminal","arguments":"""
    r""""{\"command\":\"grep -n 'a\\|\\\"b\\\"' f.py | head -5\"}}</tool_call>"""
)
_GOOD_CALL = '<tool_call>{"id":"call_2","type":"function","function":{"name":"terminal","arguments":"{\\"command\\": \\"ls\\"}"}}</tool_call>'


def _client_module(monkeypatch):
    mod = _load_plugin(monkeypatch)
    return importlib.import_module(f"{mod.__name__}.kiro_acp_client")


class TestToolCallRepair:
    def test_broken_arguments_string_becomes_a_runnable_call(self, monkeypatch):
        from agent.acp_openai_bridge import extract_tool_calls_from_text

        client_mod = _client_module(monkeypatch)
        text = "확인합니다.\n\n" + _BROKEN_CALL
        assert extract_tool_calls_from_text(text)[0] == []  # the bridge alone drops it

        repaired, broken = client_mod.repair_tool_calls(text)
        calls, cleaned = extract_tool_calls_from_text(repaired)

        assert broken == 0
        assert cleaned == "확인합니다."
        assert [(c.id, c.function.name) for c in calls] == [("call_1", "terminal")]
        assert json.loads(calls[0].function.arguments) == {"command": "grep -n 'a\\|\"b\"' f.py | head -5"}

    def test_valid_blocks_and_prose_mentions_pass_through(self, monkeypatch):
        client_mod = _client_module(monkeypatch)
        text = "Use a <tool_call> tag.\n" + _GOOD_CALL
        assert client_mod.repair_tool_calls(text) == (text, 0)

    def test_unclosed_valid_block_gets_closed(self, monkeypatch):
        client_mod = _client_module(monkeypatch)
        assert client_mod.repair_tool_calls(_GOOD_CALL[: -len("</tool_call>")]) == (_GOOD_CALL, 0)

    def test_unrecoverable_block_triggers_one_retry(self, monkeypatch):
        client_mod = _client_module(monkeypatch)
        prompts: list[str] = []
        replies = iter(["보겠습니다.\n<tool_call>{\"function\": oops}</tool_call>", _GOOD_CALL])

        def fake_run_prompt(self, prompt_text, *, timeout_seconds, model=None):
            prompts.append(prompt_text)
            return next(replies), ""

        monkeypatch.setattr(client_mod.CopilotACPClient, "_run_prompt", fake_run_prompt)
        client = client_mod.KiroACPClient(command="kiro-cli", args=["acp"])

        assert client._run_prompt("P", timeout_seconds=5) == (_GOOD_CALL, "")
        assert prompts == ["P", f"P\n\n{client_mod._RETRY_NOTE}"]

    def test_prompt_uses_flat_format_and_flat_calls_reach_the_bridge(self, monkeypatch):
        from agent.acp_client import _format_messages_as_prompt
        from agent.acp_openai_bridge import extract_tool_calls_from_text

        client_mod = _client_module(monkeypatch)
        tools = [{"type": "function", "function": {"name": "terminal", "parameters": {"type": "object"}}}]
        prompt = client_mod.kiro_tool_prompt(
            _format_messages_as_prompt([{"role": "user", "content": "hi"}], tools=tools))
        assert "must be a JSON string" not in prompt
        assert "OpenAI function-call shape" not in prompt
        assert '<tool_call name="TOOL_NAME">' in prompt

        # Flat calls with the measured Kiro failures: a lost closing brace, leaked native markup.
        text = ('보겠습니다.\n<tool_call name="terminal">{"command": "grep \'a\\\\|b\' f"}</parameter>\n</invoke></tool_call>\n'
                '<tool_call name="read_file">{"path": "x.py"</tool_call>')
        repaired, broken = client_mod.repair_tool_calls(text)
        calls, cleaned = extract_tool_calls_from_text(repaired)

        assert broken == 0
        assert cleaned == "보겠습니다."
        assert [(c.function.name, json.loads(c.function.arguments)) for c in calls] == [
            ("terminal", {"command": "grep 'a\\|b' f"}), ("read_file", {"path": "x.py"})]

    def test_forgotten_closer_keeps_both_calls_and_the_prose(self, monkeypatch):
        from agent.acp_openai_bridge import extract_tool_calls_from_text

        client_mod = _client_module(monkeypatch)
        text = ('<tool_call name="terminal">{"command": "du -sh x"}</parameter>\n</invoke>\n\n'
                'Correction, again:\n\n<tool_call name="terminal">{"command": "git log -1"}')
        repaired, broken = client_mod.repair_tool_calls(text)
        calls, cleaned = extract_tool_calls_from_text(repaired)

        assert broken == 0
        assert cleaned == "Correction, again:"
        assert [json.loads(c.function.arguments)["command"] for c in calls] == ["du -sh x", "git log -1"]


class TestRealDiscovery:
    def test_plugin_dir_discovered_from_isolated_hermes_home(self, tmp_path, monkeypatch):
        """The install path real users use: drop-in under
        $HERMES_HOME/plugins/model-providers/kiro-acp/ must register."""
        import providers

        home = tmp_path / "hermes-home"
        target = home / "plugins" / "model-providers" / "kiro-acp"
        target.mkdir(parents=True)
        for fname in ("__init__.py", "kiro_acp_client.py", "plugin.yaml"):
            (target / fname).write_bytes((REPO_ROOT / fname).read_bytes())

        # Reset discovery state exactly like a fresh process.
        monkeypatch.setattr(providers, "_discovered", False)
        providers._REGISTRY.pop("kiro-acp", None)
        for alias in ("kiro", "kiro-cli", "kiro-acp-agent"):
            providers._ALIASES.pop(alias, None)

        monkeypatch.setenv("HERMES_HOME", str(home))
        profile = providers.get_provider_profile("kiro-acp")

        assert profile is not None, "kiro-acp must be discovered from $HERMES_HOME drop-in"
        assert profile.auth_type == "external_process"
        assert get_provider_profile_by_alias(providers, "kiro") is profile


def get_provider_profile_by_alias(providers, alias: str):
    return providers.get_provider_profile(alias)

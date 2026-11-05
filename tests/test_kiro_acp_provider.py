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
        assert profile.process_args == ("acp", "--agent", "hermes-bridge")
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
        assert mod.resolve_kiro_args() == ["acp", "--agent", "hermes-bridge"]

    def test_create_client_returns_kiro_client_with_profile_defaults(self, monkeypatch):
        mod = _load_plugin(monkeypatch)
        client = mod.kiro_acp.create_client()
        assert type(client).__name__ == "KiroACPClient"
        assert client._acp_command == "kiro-cli"
        assert client._acp_args == ["acp", "--agent", "hermes-bridge"]
        assert client.base_url == "acp://kiro"

    def test_spawn_writes_tool_less_agent_it_selects(self, monkeypatch, tmp_path):
        # Kiro's own tools let the model act outside <tool_call>; Hermes cancels the permission
        # request and the turn ends on narration. The selected agent must expose no tools.
        mod = _client_module(monkeypatch)
        monkeypatch.setenv("HOME", str(tmp_path))
        path = tmp_path / ".kiro" / "agents" / "hermes-bridge.json"
        path.parent.mkdir(parents=True)
        path.write_text('{"name": "hermes-bridge", "tools": ["*"]}')
        client = mod.KiroACPClient(command="/nonexistent/kiro-cli")
        with pytest.raises(RuntimeError, match="Could not start Kiro ACP command"):
            client._spawn()
        config = json.loads(path.read_text())
        assert config["name"] == "hermes-bridge"
        assert config["tools"] == [] and config["includeMcpJson"] is False

    def test_other_agent_override_leaves_kiro_config_alone(self, monkeypatch, tmp_path):
        mod = _client_module(monkeypatch)
        monkeypatch.setenv("HOME", str(tmp_path))
        mod.ensure_kiro_agent(["acp", "--agent", "builder"])
        assert not (tmp_path / ".kiro").exists()


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


class TestKiroConsole:
    def test_selected_home_wins_over_hermes_environment(self, tmp_path, monkeypatch):
        mod = _client_module(monkeypatch)
        monkeypatch.setenv("KIRO_CONSOLE_HOME", str(tmp_path))
        home = tmp_path / "seat-home"
        home.mkdir()
        (tmp_path / "active-seat.json").write_text(json.dumps({"seat": "seat-a", "home": str(home)}))
        monkeypatch.setattr(mod._acp, "_acp_supported", lambda *args: True)
        monkeypatch.setattr(mod._acp, "_build_subprocess_env", lambda: {"HOME": "/hermes-override"})
        captured = {}
        # The active process collection requires hashable objects.
        class Process:
            stdin = object()
            stdout = object()
        proc = Process()
        def popen(*args, **kwargs):
            captured.update(kwargs)
            return proc
        monkeypatch.setattr(mod.subprocess, "Popen", popen)
        client = mod.KiroACPClient()
        assert client._spawn() is proc
        assert captured["env"]["HOME"] == str(home)
        assert (home / ".kiro" / "agents" / "hermes-bridge.json").is_file()

    def test_hermes_consumer_target_overrides_global_seat(self, tmp_path, monkeypatch):
        mod = _client_module(monkeypatch)
        monkeypatch.setenv("KIRO_CONSOLE_HOME", str(tmp_path))
        monkeypatch.setattr(mod, "ensure_kiro_agent", lambda *args: None)
        global_home, pinned_home = tmp_path / "seat-a", tmp_path / "seat-b"
        global_home.mkdir()
        pinned_home.mkdir()
        path = tmp_path / "active-seat.json"
        doc = {"seat": "seat-a", "home": str(global_home), "consumers": {
            "hermes": {"mode": "pinned", "seat": "seat-b", "home": str(pinned_home)},
            "omp": {"mode": "follow", "seat": "seat-a", "home": None}}}
        path.write_text(json.dumps(doc))
        client = mod.KiroACPClient()
        assert client._active_seat()[1:] == ("seat-b", str(pinned_home))
        # Hermes pinned to the default login runs as today even while the global seat is isolated.
        doc["consumers"]["hermes"] = {"mode": "pinned", "seat": "default", "home": None}
        path.write_text(json.dumps(doc))
        assert client._active_seat()[1:] == ("default", None)

    def test_fallbacks_use_original_spawn(self, tmp_path, monkeypatch):
        mod = _client_module(monkeypatch)
        monkeypatch.setenv("KIRO_CONSOLE_HOME", str(tmp_path))
        monkeypatch.setattr(mod, "ensure_kiro_agent", lambda *args: None)
        marker = object()
        monkeypatch.setattr(mod.CopilotACPClient, "_spawn", lambda self: marker)
        client = mod.KiroACPClient()
        path = tmp_path / "active-seat.json"
        assert client._spawn() is marker
        for content in ('{"seat":"default","home":null}',
                        json.dumps({"seat": "bad", "home": str(tmp_path / "missing")}),
                        "{invalid"):
            path.write_text(content)
            assert client._spawn() is marker

    def test_metering_attaches_spawn_seat_and_model(self, tmp_path, monkeypatch):
        mod = _client_module(monkeypatch)
        monkeypatch.setenv("KIRO_CONSOLE_HOME", str(tmp_path))
        monkeypatch.setattr(mod, "ensure_kiro_agent", lambda *args: None)
        monkeypatch.setattr(mod.CopilotACPClient, "_spawn", lambda self: object())
        message = {"jsonrpc": "2.0", "method": "_kiro.dev/metadata",
                   "params": {"sessionId": "s", "contextUsagePercentage": 3.2,
                              "meteringUsage": [{"value": 0.3266, "unit": "credit",
                                                 "unitPlural": "credits"}],
                              "turnDurationMs": 6383}}
        home = tmp_path / "seat-home"
        home.mkdir()
        (tmp_path / "active-seat.json").write_text(json.dumps({"seat": "seat-a", "home": str(home)}))
        def run_prompt(client, prompt_text, *, timeout_seconds, model):
            # A mocked spawn still records the seat, without starting Kiro.
            monkeypatch.setattr(mod._acp, "_acp_supported", lambda *args: True)
            class Process:
                stdin = object()
                stdout = object()
            monkeypatch.setattr(mod.subprocess, "Popen", lambda *args, **kwargs: Process())
            client._spawn()
            (tmp_path / "active-seat.json").write_text('{"seat":"default","home":null}')
            assert client._handle_server_message(message)
            return "ok", ""
        monkeypatch.setattr(mod.CopilotACPClient, "_run_prompt", run_prompt)
        client = mod.KiroACPClient()
        assert client._run_prompt("hello", timeout_seconds=1, model="claude-sonnet-5.5") == ("ok", "")
        path = tmp_path / "metering.jsonl"
        record = json.loads(path.read_text())
        assert record["ts"].endswith("Z")
        assert {k: record[k] for k in ("seat", "model", "credits", "turn_ms")} == {
            "seat": "seat-a", "model": "claude-sonnet-5.5", "credits": 0.3266, "turn_ms": 6383}
        assert path.stat().st_mode & 0o777 == 0o600

    def test_metering_does_not_create_data_root(self, tmp_path, monkeypatch):
        mod = _client_module(monkeypatch)
        root = tmp_path / "missing"
        monkeypatch.setenv("KIRO_CONSOLE_HOME", str(root))
        monkeypatch.setattr(mod, "ensure_kiro_agent", lambda *args: None)
        monkeypatch.setattr(mod.CopilotACPClient, "_spawn", lambda self: object())
        def run_prompt(client, prompt_text, *, timeout_seconds, model):
            client._spawn()
            client._handle_server_message({"method": "_kiro.dev/metadata", "params": {
                "meteringUsage": [{"value": 1, "unit": "credit"}], "turnDurationMs": 42}})
            return "ok", ""
        monkeypatch.setattr(mod.CopilotACPClient, "_run_prompt", run_prompt)
        mod.KiroACPClient()._run_prompt("hello", timeout_seconds=1, model="test")
        assert not root.exists()

    def test_quota_retries_chain_home_and_meters_answering_seat(self, tmp_path, monkeypatch):
        mod = _client_module(monkeypatch)
        monkeypatch.setenv("KIRO_CONSOLE_HOME", str(tmp_path))
        homes = [tmp_path / name for name in ("primary", "fallback")]
        for home in homes:
            home.mkdir()
        (tmp_path / "active-seat.json").write_text(json.dumps({"consumers": {"hermes": {
            "seat": "primary", "home": str(homes[0]),
            "chain": [{"seat": "fallback", "home": str(homes[1])}]}}}))
        monkeypatch.setattr(mod, "ensure_kiro_agent", lambda *args: None)
        monkeypatch.setattr(mod._acp, "_acp_supported", lambda *args: True)
        monkeypatch.setattr(mod._acp, "_build_subprocess_env", lambda: {})
        spawned = []
        class Process:
            stdin = object()
            stdout = object()
        def popen(*args, **kwargs):
            spawned.append(kwargs["env"]["HOME"])
            return Process()
        monkeypatch.setattr(mod.subprocess, "Popen", popen)
        prompts = []
        def run_prompt(client, prompt_text, *, timeout_seconds, model):
            client._spawn()
            prompts.append(prompt_text)
            if len(prompts) == 1:
                raise RuntimeError("CREDIT quota exceeded")
            client._handle_server_message({"method": "_kiro.dev/metadata", "params": {
                "meteringUsage": [{"value": 0.5, "unit": "credit"}], "turnDurationMs": 20}})
            return "fallback answer", ""
        monkeypatch.setattr(mod.CopilotACPClient, "_run_prompt", run_prompt)
        assert mod.KiroACPClient()._run_prompt("hello", timeout_seconds=1, model="test") == ("fallback answer", "")
        assert spawned == [str(home) for home in homes]
        assert prompts == ["hello", "hello"]
        record = json.loads((tmp_path / "metering.jsonl").read_text())
        assert record["seat"] == "fallback"
        assert record["model"] == "test"

    def test_non_quota_error_does_not_retry(self, tmp_path, monkeypatch):
        mod = _client_module(monkeypatch)
        monkeypatch.setenv("KIRO_CONSOLE_HOME", str(tmp_path))
        home = tmp_path / "seat"
        home.mkdir()
        (tmp_path / "active-seat.json").write_text(json.dumps({"seat": "primary", "home": str(home),
            "chain": [{"seat": "fallback", "home": str(home)}]}))
        monkeypatch.setattr(mod, "ensure_kiro_agent", lambda *args: None)
        monkeypatch.setattr(mod._acp, "_acp_supported", lambda *args: True)
        class Process:
            stdin = object()
            stdout = object()
        monkeypatch.setattr(mod.subprocess, "Popen", lambda *args, **kwargs: Process())
        attempts = []
        error = RuntimeError("protocol failure")
        def run_prompt(client, prompt_text, *, timeout_seconds, model):
            client._spawn()
            attempts.append(prompt_text)
            raise error
        monkeypatch.setattr(mod.CopilotACPClient, "_run_prompt", run_prompt)
        with pytest.raises(RuntimeError) as caught:
            mod.KiroACPClient()._run_prompt("hello", timeout_seconds=1)
        assert caught.value is error
        assert attempts == ["hello"]

    def test_empty_chain_keeps_original_failure(self, tmp_path, monkeypatch):
        mod = _client_module(monkeypatch)
        monkeypatch.setenv("KIRO_CONSOLE_HOME", str(tmp_path))
        (tmp_path / "active-seat.json").write_text(json.dumps({"seat": "default", "home": None, "chain": []}))
        monkeypatch.setattr(mod, "ensure_kiro_agent", lambda *args: None)
        monkeypatch.setattr(mod.CopilotACPClient, "_spawn", lambda self: object())
        attempts = []
        error = TimeoutError("429 throttle")
        def run_prompt(client, prompt_text, *, timeout_seconds, model):
            client._spawn()
            attempts.append(prompt_text)
            raise error
        monkeypatch.setattr(mod.CopilotACPClient, "_run_prompt", run_prompt)
        with pytest.raises(TimeoutError) as caught:
            mod.KiroACPClient()._run_prompt("hello", timeout_seconds=1)
        assert caught.value is error
        assert attempts == ["hello"]

    def test_ordered_chain_and_tool_repair_retry_stay_on_selected_seat(self, tmp_path, monkeypatch):
        mod = _client_module(monkeypatch)
        monkeypatch.setenv("KIRO_CONSOLE_HOME", str(tmp_path))
        homes = [tmp_path / name for name in ("one", "two", "three")]
        for home in homes:
            home.mkdir()
        (tmp_path / "active-seat.json").write_text(json.dumps({
            "seat": "one", "home": str(homes[0]),
            "chain": [{"seat": home.name, "home": str(home)} for home in homes[1:]]}))
        monkeypatch.setattr(mod, "ensure_kiro_agent", lambda *args: None)
        monkeypatch.setattr(mod._acp, "_acp_supported", lambda *args: True)
        monkeypatch.setattr(mod._acp, "_build_subprocess_env", lambda: {})
        spawned, prompts = [], []
        class Process:
            stdin = object()
            stdout = object()
        def popen(*args, **kwargs):
            spawned.append(kwargs["env"]["HOME"])
            return Process()
        monkeypatch.setattr(mod.subprocess, "Popen", popen)
        def run_prompt(client, prompt_text, *, timeout_seconds, model):
            client._spawn()
            prompts.append(prompt_text)
            if len(prompts) <= 2:
                raise OSError("not logged in")
            if len(prompts) == 3:
                return '<tool_call>{"function": oops}</tool_call>', ""
            return _GOOD_CALL, ""
        monkeypatch.setattr(mod.CopilotACPClient, "_run_prompt", run_prompt)
        assert mod.KiroACPClient()._run_prompt("hello", timeout_seconds=1) == (_GOOD_CALL, "")
        assert spawned == [str(homes[0]), str(homes[1]), str(homes[2]), str(homes[2])]
        assert prompts == ["hello", "hello", "hello", f"hello\n\n{mod._RETRY_NOTE}"]

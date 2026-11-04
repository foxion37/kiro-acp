# kiro-acp

Fork of [tobenwarrior/hermes-kiro-provider](https://github.com/tobenwarrior/hermes-kiro-provider)
that fixes Kiro turns ending mid-sentence ("…확인하겠습니다." with no tool call):
Kiro gets a flat `<tool_call name="TOOL">{arguments}</tool_call>` format instead
of Hermes' nested string-arguments contract, malformed blocks are repaired
before Hermes parses them, and Kiro runs as a tool-less `hermes-bridge` agent so
the model cannot call Kiro's own tools (which Hermes cancels, ending the turn).
Agents: start with [AGENTS.md](AGENTS.md).

Use the [Kiro CLI](https://kiro.dev/) (AWS's agentic coding CLI) as a model
provider inside [Hermes Agent](https://github.com/NousResearch/hermes-agent) —
Kiro's agent runs as an ACP subprocess behind Hermes' normal provider
interface, so `hermes -m kiro-acp`, `/model`, profiles, and the gateway all
work like any other backend.

Shipped as a **standalone plugin repo** per the upstream vendor-integration
policy ([hermes-agent#80362](https://github.com/NousResearch/hermes-agent/pull/80362)):
zero edits to hermes-agent core; it rides the documented
`auth_type="external_process"` provider seam
([Model Provider Plugins → External-process (ACP) providers](https://hermes-agent.nousresearch.com/docs/developer-guide/model-provider-plugin)).

## Prerequisites

- hermes-agent installed and working (`hermes doctor` clean)
- [Kiro CLI](https://kiro.dev/docs/cli/) installed and **logged in**
  (`kiro-cli` on PATH; login state in `~/.kiro`)

Verified against Kiro CLI 2.22.0 (`kiro-cli acp`, ACP protocol v1 over stdio).

## Install

**Via the plugin installer** (recommended):

```bash
hermes plugins install foxion37/kiro-acp
```

**Or manual drop-in**:

```bash
git clone https://github.com/foxion37/kiro-acp.git
mkdir -p "${HERMES_HOME:-$HOME/.hermes}/plugins/model-providers"
cp -r kiro-acp "${HERMES_HOME:-$HOME/.hermes}/plugins/model-providers/kiro-acp"
```

Both paths register the `kiro-acp` provider for the next session.

## Use

```bash
hermes -m kiro-acp "what can you do?"
```

Or pick **Kiro (AWS)** in `hermes model` / `/model`. The provider name and
aliases are `kiro-acp`, `kiro`, `kiro-cli`.

Auth is handled by the `kiro-cli` subprocess itself (your existing `~/.kiro`
login) — no API keys to configure.

## Configuration (optional env overrides)

| Env var | Purpose | Default |
|---|---|---|
| `HERMES_KIRO_ACP_COMMAND` | Path to a custom kiro binary | `kiro-cli` |
| `KIRO_CLI_PATH` | Older alias for the same | — |
| `HERMES_KIRO_ACP_ARGS` | Extra argv (shlex-split), e.g. `"acp --agent hermes-bridge --effort high"`. Choosing another agent brings back Kiro's own tools and the cut-off turns they cause | `acp --agent hermes-bridge` |

## How it works

`KiroACPProfile` is a `ProviderProfile` with `auth_type="external_process"`
that spawns `kiro-cli acp` and adapts it to Hermes' OpenAI-style client
interface by extending the in-tree `copilot-acp` ACP shim (`agent.copilot_acp_client`)
— session lifecycle, streaming updates, permission handling, and the
file-safety bridge are inherited, so upstream fixes flow through. Model
listing is delegated to the subprocess; `model="kiro-acp"` means "use the
session default model".

The default argv selects the `hermes-bridge` agent. Before each spawn the plugin
writes `~/.kiro/agents/hermes-bridge.json` (no tools, no MCP servers) when it is
missing or differs; Hermes' tools stay the only way the model can act.

## kiro-console integration

When kiro-console is installed, each spawn reads `active-seat.json` under
`$KIRO_CONSOLE_HOME` (default `~/.kiro-console`). An existing seat HOME is used
for Kiro and its bridge agent configuration; a missing, unreadable, malformed,
null-HOME or nonexistent-directory selection falls back to the normal spawn.
Completed-turn credit metadata is appended to `metering.jsonl` in that data
root (created mode 600); the plugin never creates the data root, and metering
errors do not interrupt a turn.

## Development

```bash
# unit + discovery tests (need a hermes-agent checkout; defaults to ~/.hermes/hermes-agent)
HERMES_REPO=/path/to/hermes-agent python -m pytest tests/ -q
```

Tests cover the profile contract (registration, env overrides, client
construction), real discovery from an isolated `$HERMES_HOME`, and the
tool-call format and repair — no live Kiro calls. `scripts/replay_session.py`
replays a real Hermes session against live Kiro to measure tool-call health.

## License

MIT

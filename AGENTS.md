# kiro-acp — agent runbook

Hermes model-provider plugin that drives `kiro-cli acp`. Fork of
[tobenwarrior/hermes-kiro-provider](https://github.com/tobenwarrior/hermes-kiro-provider)
with a fix for Kiro turns that stop mid-sentence. Plugin name stays
`kiro-acp-provider` (installed directory and `plugins.enabled` entry); provider
id `kiro-acp`, aliases `kiro`, `kiro-cli`.

## Symptom this repo exists for

A Hermes reply through `kiro-acp` ends on a lead-in such as
"…확인하겠습니다." / "Let me check…" with no tool call, and the turn stops.
Logs may show `Empty response … provider=kiro-acp` or
`Kiro ACP returned N unparseable <tool_call> block(s)`.

## Root cause (keep this model in mind)

1. Kiro has no channel for Hermes tools, so the ACP bridge in hermes-agent
   (`agent/acp_openai_bridge.py`, `TOOL_CALL_CONTRACT`) asks the model to write
   `<tool_call>{JSON}</tool_call>` as plain text.
2. Hermes' contract is a three-level object whose `arguments` is a JSON-encoded
   string: shell commands get escaped twice and the block ends in a run of
   closers. Kiro models drop a quote or `}` there (measured: 7 of 10 calls broken).
3. `extract_tool_calls_from_text` silently drops a block that is not valid JSON,
   so only the narration before it survives and the turn ends.

## What the fix does (`kiro_acp_client.py`)

- `kiro_tool_prompt`: replaces Hermes' contract (and its preamble line) with the
  flat format `<tool_call name="TOOL">{arguments}</tool_call>`. If upstream
  rewords `TOOL_CALL_CONTRACT`, the swap silently stops applying: check this first.
- `repair_tool_calls`: converts flat blocks to the bridge format and repairs the
  observed failure shapes: lost trailing `}`/`"`, leaked Claude markup
  (`</parameter></invoke>`), a forgotten `</tool_call>` before the next block or
  before prose, invalid JSON escapes such as `\|`, and the old string-arguments
  form with a lost closing quote.
- `KiroACPClient._run_prompt`: applies both and re-prompts once (`_RETRY_NOTE`)
  only when a block stays unrecoverable.

All of this lives in the plugin, never in hermes-agent core, so `hermes update`
cannot erase it.

## When the symptom comes back

1. **Is the fix installed?** In every plugin copy
   (`~/.hermes/plugins/kiro-acp-provider` and
   `~/.hermes/profiles/*/plugins/kiro-acp-provider`): `git log --oneline -3`
   must contain this repo's commits and `git remote -v` must point at this repo.
2. **Did core change underneath?** Compare `TOOL_CALL_CONTRACT` and the
   `<tool_call>` line of `_PROMPT_PREAMBLE` in
   `~/.hermes/hermes-agent/agent/acp_openai_bridge.py` / `agent/acp_client.py`
   with what `kiro_tool_prompt` expects, and check `ACPClient._run_prompt`'s
   signature. Adapt the plugin, not core.
3. **New failure shape?** Find the cut-off turn and replay it:
   ```bash
   ~/.hermes/hermes-agent/venv/bin/python scripts/replay_session.py --list
   ~/.hermes/hermes-agent/venv/bin/python scripts/replay_session.py SESSION_ID LAST_MESSAGE_ID --runs 10
   ```
   `RAW-INVALID` lines show the raw block endings. Add the new shape to
   `repair_tool_calls` and a regression test with that exact shape.
   Each run spends Kiro credits; 10 to 20 runs are enough to measure.
4. **Verify** (target: `unrecoverable=0`):
   - tests: `HERMES_REPO=~/.hermes/hermes-agent python -m pytest tests -q`
     (the Hermes venv has no pytest; use any Python that has it, and do not
     install packages into the Hermes venv)
   - replay: `scripts/replay_session.py … --runs 10`
   - real path: `hermes chat -Q -q "terminal 도구로 'echo ok'를 실행하고 출력만 알려줘."`
5. **Deploy**: commit and push here, then update every copy and restart:
   ```bash
   for d in ~/.hermes/plugins/kiro-acp-provider ~/.hermes/profiles/*/plugins/kiro-acp-provider; do
     git -C "$d" pull --ff-only
   done
   hermes gateway restart   # run from a shell outside the gateway
   ```

## Taking upstream changes

Remote `upstream` = tobenwarrior/hermes-kiro-provider. Merge it into `main`,
resolve conflicts in favour of keeping `kiro_tool_prompt` / `repair_tool_calls`
/ the `_run_prompt` override, run step 4, then push and deploy. Do not reinstall
from upstream: `hermes plugins install tobenwarrior/...` replaces the fixed copy.

## Rules

- Never print Kiro credentials; auth lives in kiro-cli's own store.
- Keep personal paths, session content and account names out of commits.

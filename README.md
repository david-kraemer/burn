# burn

`burn` is a terminal user interface (TUI) for monitoring token use in Claude
Code and Codex.

It reads the JSONL transcript files these tools write. It does not
instrument either tool.

Claude Code does not save quota data in transcript files. `burn` gets this data
from Anthropic. It uses the OAuth token in the Claude Code login keychain. This
is the account data shown by `/usage`. Use `--no-remote` to read local files
only. In this mode, the five-hour meter estimates weighted tokens. Use
`--limit` to set the allowance for this estimate.

## Install

Install the command with `uv`:

```sh
uv tool install --editable .
```

Use `--editable` while developing `burn`. Omit it to install a snapshot.

## Use

Run `burn` without a subcommand to open the live monitor.

```text
burn
burn tools
burn turns
burn session <session-id-prefix>
burn cost
burn waste
burn verify
```

The views have these purposes:

| Command | Purpose |
| --- | --- |
| `burn` | Show token use by session. |
| `burn tools` | Show context growth by tool. |
| `burn turns` | Show the most expensive prompts. |
| `burn session <id>` | Show one session in detail. |
| `burn cost` | Estimate spend by model and project. |
| `burn waste` | Show cache tokens recreated after idle periods. |
| `burn verify` | Compare observed Claude tokens with Claude totals. |

Common options:

```text
--window <minutes>    History window. The default is 300 minutes.
--source cc|cx        Show Claude Code or Codex only.
--sort <column>       Select the initial sort column.
--top <number>        Limit rows in detail views. The default is 15.
--interval <seconds>  Refresh interval for the live view. The default is 2.
--limit <tokens>      Set the local weighted-token allowance.
--no-remote           Read local transcripts. Skip the quota request.
--once                Print one live-view frame and exit.
--plain               Print ASCII text without colour.
--json                Print one snapshot as JSON. See "Programmatic use".
--follow              With --json, print a document per line every --interval.
```

For example:

```sh
burn --window 60 --source cx
burn tools --window 1440 --top 20
burn --once
```

When standard output is not a terminal, every view prints plain ASCII text
with no colour codes, no block characters, and no trailing blanks, and the
live view prints one frame instead of taking over the screen. `--plain` asks
for the same output on a terminal. Block characters map one for one to ASCII
(`█` to `#`, `·` to `-`, sparkline levels to `_.:~=+*#`), so columns stay
aligned.

Quota reset times show the clock alone when the reset is later today. A
reset on another day shows its date and the time left, for example
`Thu Nov 19 5:59AM EST (in 3d04h)`.

## Live-view controls

| Key | Action |
| --- | --- |
| `↑` `↓`, `j` `k` | Move through the sessions. |
| `Enter` | Open or close the selected session. |
| `t` | Switch the detail pane between tools and turns. |
| `s`, `S` | Change the sort column. |
| `r` | Reverse the sort order. |
| `/` | Filter by session, project, model, or agent. Press `Enter` to accept. |
| `a` | Cycle through both agents, Claude Code, and Codex. |
| `+`, `-` | Increase or decrease the time window. |
| `Space` | Pause or resume sampling. |
| `c` | Clear the filter, selection, and agent filter. |
| `h`, `?` | Show the key reference. |
| `q` | Quit. |

## Data sources

`burn` reads these files:

| Agent | Transcript location | Usage record |
| --- | --- | --- |
| Claude Code | `~/.claude/projects/*/*.jsonl` | `assistant` records and `message.usage` |
| Claude Code subagents | `~/.claude/projects/*/<session>/subagents/**/*.jsonl` | the same |
| Codex | `~/.codex/sessions/Y/M/D/rollout-*.jsonl` | `event_msg` records and `token_count` |

The program normalizes both formats into one call record. It reads new data
on each refresh and tracks the last byte read from each transcript.

Claude Code can write several records for one API call. `burn` merges records
with the same message and request identifiers so that it does not count the
same usage more than once. The last record carries the final usage; earlier
ones in subagent transcripts carry a partial output count.

Subagent records carry the parent's session ID, so their usage joins the
parent session's row. The row's `CTX` and model come from the main thread.

Codex can repeat a `token_count` event without new usage. `burn` counts an
event only when the running total changes.

## Token weight

The monitor reports `WEIGHT`, which expresses usage in input-token equivalents.
It applies these relative weights:

| Token type | Weight |
| --- | ---: |
| Input | 1.0 |
| Cache write | 1.25 |
| Cache read | 0.1 |
| Output | 5.0 |

This measure reflects the tool's relative token prices. It is not a
provider invoice.

`burn tools` reports two related values:

- `ADDED` is context growth associated with a tool result.
- `CARRIED` is the later cost of reading that context again.

## Cost estimates

Claude Code records cost data when a session closes. `burn cost` uses
sessions whose observed tokens reconcile with that data to calculate an
effective rate for each model.

Codex transcript records contain token counts but no cost. `burn` therefore
does not report a Codex dollar amount.

The Claude meter uses reported quota data if available. If remote quota data is
not available, it uses a five-hour estimate. Use `--limit` to set the
weighted-token allowance:

```sh
burn --limit 40e6
```

The value is a five-hour weighted-token allowance. The Codex meter uses the
percentages recorded by Codex.

## Limits

The transcript is not a complete billing record.

Claude Code can bill background work absent from its transcripts. `burn
verify` shows the difference between observed and Claude's recorded totals,
counting subagent transcripts toward the session that started them.

Codex token totals are read from its `token_count` records. The tool does not
infer costs that Codex does not record.

## Programmatic use

`burn --json` prints one JSON document and exits. `burn --json --follow`
prints one compact document per line (JSON Lines) every `--interval` seconds,
flushing after each. It reads transcripts incrementally and exits quietly on
Ctrl-C or when the reader closes the pipe. `--window`, `--source`, `--sort`,
and `--no-remote` apply to both.

```sh
burn --json --window 60 | jq '.sessions[] | {session, weight}'
burn --json --follow --interval 5 --no-remote | jq -c "{at, sessions: (.sessions | length)}"
```

Python programs can import the same data from the `burn` package. Public
functions carry the `burn_` prefix and live at the top level; records keep
their class names (`burn.Snapshot`, `burn.View`, …). Submodules are
implementation detail.

```python
import asyncio
import burn

doc = asyncio.run(burn.burn_latest(window=60))           # one document

async def watch():
    async for doc in burn.burn_documents(window=60, interval=5):   # until you stop
        ...
```

| Function | Returns |
| --- | --- |
| `async burn_latest(window=300, *, source=None, sort="weight", remote=True)` | One document. Waits up to four seconds for quota data. |
| `async burn_documents(window=300, *, interval=2.0, source=None, sort="weight", remote=True)` | An async generator of documents. Only the first waits for quota data. |
| `burn_document(snapshot, view)` | The document for a `Snapshot` and `View`. Pure. |

### Document schema

Fields may be added. Existing fields keep their names and meanings. `schema`
changes only when a field is removed or redefined. Times are Unix epoch
seconds. Percentages run from 0 to 100.

| Field | Type | Meaning |
| --- | --- | --- |
| `schema` | int | Document version, currently 1. |
| `at` | float | Sample time. |
| `window_minutes` | int | History window of `sessions`. |
| `source` | string or null | Agent filter: `"cc"`, `"cx"`, or null for both. |
| `quota` | list | Claude quota windows, shortest first. Empty without remote data. |
| `quota[].name` | string | `"5h"`, `"week"`, or the reported kind. |
| `quota[].group` | string | `"session"`, `"weekly"`, and so on. |
| `quota[].used_percent` | float | Share of the window used. |
| `quota[].resets_at` | float or null | Reset time. |
| `quota[].scope` | string or null | Model the window applies to, if scoped. |
| `quota[].active` | bool | True for the window that governs the account. |
| `quota_at` | float or null | When the quota values were fetched. |
| `quota_pending` | bool | True when the values are older than 15 minutes and are kept only for their shape. |
| `spend` | object or null | Extra-usage credits: `used`, `cap`, in major currency units. |
| `codex` | list | Codex quota windows: `used_percent`, `window_minutes`, `resets_at`. |
| `sessions` | list | One row per session and project, sorted by `--sort`, largest first. |
| `sessions[].source` | string | `"cc"` or `"cx"`. |
| `sessions[].session` | string | Session key: the first 12 characters of the session ID. |
| `sessions[].project` | string | Last component of the working directory. |
| `sessions[].model` | string | Model of the main thread's latest request. |
| `sessions[].ctx` | int | Main thread's current context, in tokens. |
| `sessions[].cache` | float | Percent of cached input read rather than written. |
| `sessions[].think` | int | Reasoning tokens. |
| `sessions[].weight` | float | Weighted tokens in the window, subagents included. |
| `sessions[].rate` | float | Weighted tokens per minute over the last five minutes. |
| `sessions[].cost` | float | Estimated dollars. Zero for Codex. |
| `sessions[].calls` | int | Requests in the window. |
| `sessions[].first`, `sessions[].last` | float | Times of the first and last request in the window. |
| `sessions[].fanout` | bool | The session started subagents. |

A session that ran in several directories has one row per project. Sum
rows by `session` to total a session.

## Requirements

- Python 3.13 or later
- `uv`
- A terminal that supports the live view

The runtime Python dependencies are `rich` and `certifi`.

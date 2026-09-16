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
```

For example:

```sh
burn --window 60 --source cx
burn tools --window 1440 --top 20
burn --once
```

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
| Codex | `~/.codex/sessions/Y/M/D/rollout-*.jsonl` | `event_msg` records and `token_count` |

The program normalizes both formats into one call record. It reads new data
on each refresh and tracks the last byte read from each transcript.

Claude Code can write several records for one API call. `burn` merges records
with the same message and request identifiers so that it does not count the
same usage more than once.

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

Claude Code can bill subagent and background work absent from its main
transcript. `burn verify` shows the difference between observed and Claude's
recorded totals. Sessions using `Task`, `Agent`, or `Workflow` can show a
larger difference.

Codex token totals are read from its `token_count` records. The tool does not
infer costs that Codex does not record.

## Requirements

- Python 3.13 or later
- `uv`
- A terminal that supports the live view

The runtime Python dependencies are `rich` and `certifi`.

# burn

Token burn inspection for Claude Code and Codex, built entirely from the JSONL
transcripts both agents already write. Nothing to instrument.

    burn                  live per-session monitor
    burn tools            which tools are inflating context
    burn turns            the most expensive prompts
    burn session <id>     one session in detail
    burn cost             spend, at rates calibrated from your own billing
    burn waste            cache re-creation you are paying for
    burn verify           audit against Claude's own totals

`--window <minutes>` on any view, `--source cc|cx` to isolate one agent,
`--sort <column>` to pick the starting sort, `--once` for a single frame.
A 24-hour window takes about 0.3s.

## Install

    uv tool install --editable .

Editable, so edits to the working tree take effect with no reinstall. Drop
`--editable` for a snapshot install instead.

The file also carries a PEP 723 header, so `uv run burn.py` works in a fresh
clone with nothing installed at all. `check.py` re-runs the correctness checks
behind the fixes described below.

## The live view is interactive

Like `top`, it is driven from the keyboard rather than from flags.

| key | what it does |
|---|---|
| `↑ ↓` `j k` | move the cursor |
| `↵` | zoom the selected session into a pane below the table |
| `t` | swap that pane between tools and turns |
| `s` `S` | cycle the sort column forwards / backwards |
| `r` | reverse the sort order |
| `/` | filter by session, project, model or agent |
| `a` | cycle agent: both → claude → codex |
| `+` `-` | widen / narrow the time window |
| `space` | pause and resume sampling |
| `c` | clear filter, zoom and agent selection |
| `h` `?` | key reference |
| `q` | quit |

The sorted column is marked in its header, the table scrolls to fit whatever
height the terminal has (with a count of what is off-screen), sessions seen for
the first time flash green, and sessions quiet for two minutes dim out.

Quota meters run across the top in `htop` bracket style. Codex's are real
percentages from its own records; Claude's tracks the five-hour block's clock,
because no allowance is written to disk — pass `--limit` (e.g. `--limit 40e6`)
to turn it into a true gauge against a number you supply.

## Where the numbers come from

| | Claude Code | Codex |
|---|---|---|
| files | `~/.claude/projects/*/*.jsonl` | `~/.codex/sessions/Y/M/D/rollout-*.jsonl` |
| one API call | `type:"assistant"` → `message.usage` | `event_msg` → `token_count` |
| quota | not recorded; reconstructed from gaps | `rate_limits.used_percent`, exact |
| cost | `cost-state`, at session close | not recorded at all |

Both formats are append-only, so each poll seeks to the byte offset it stopped
at. Everything normalises to one `Call` record, which is what every view reads.

Claude writes one record *per content block*, all repeating the same usage.
Keying on `(message id, request id)` merges them without double-counting the
tokens and without losing the `tool_use` blocks, which live in the later
records — the raw-to-merged ratio is about 2.0, so skipping this roughly
doubles every figure.

## WEIGHT, not total

Cache reads cost a tenth of an input token but dominate any raw sum. `WEIGHT`
charges input 1x, cache writes 1.25x, cache reads 0.1x and output 5x, matching
Anthropic's published ratios, so a single number tracks what is being spent.

## ADDED vs CARRIED

`burn tools` attributes context growth to the tool that caused it. Growth
between two consecutive calls, less the assistant's own output, is what tool
results and user text injected; it is split across the tools that ran in
between, in proportion to what each returned.

`CARRIED` then charges each tool for what it *keeps* costing. Tokens are
written to cache once and re-read by every later call, so the same result early
in a long thread costs many times what it costs at the end. The gap is not
small — over a recent day here, 2.2M tokens added became 12.0M weighted once
re-reads were counted.

Two things bound that figure, and without them it runs away:

*Threads.* A session id is not always one linear conversation. Codex runs side
threads under the same id, so its calls arrive interleaved — a real session
here bounces between a 162k prefix and an 88k one. Read as a single
conversation, every switch back up looks like 74k of fresh context, and that
session accumulated 3.6M of "growth" against a context that never exceeded
168k. Calls are therefore assigned to the open thread whose last prefix sits
closest below them.

*Compaction.* A call that undercuts every open thread has had its context
discarded and replaced by a summary, so it starts a new thread and the old
one stops accruing re-reads.

CARRIED is a decomposition of what a session actually spent, so it must never
exceed it. It is checked against that invariant: across 66 local sessions the
largest CARRIED is 0.78 of its session's real weighted total.

## Rates are calibrated, not hardcoded

Claude writes its own dollar figure into `cost-state` when a session closes.
Dividing that by the weighted tokens observed in the same session gives an
effective `$/weighted-Mtok` per model with nothing hardcoded, so it cannot go
stale as models change.

Only sessions whose tokens reconcile are used. Fitting actual per-token prices
by least squares was tried first and fails: `cache_creation` mixes two
ephemeral tiers at different prices, and fan-out sessions are billed for tokens
no transcript contains, which drags the fit to implausible values (negative
input prices). The blended rate is robust where the fit is not.

## What it cannot see

Claude Code writes only main-thread API calls. Subagent and `Workflow` traffic
is billed to the session but appears in no transcript — there are no
`isSidechain` records anywhere in the tree and no usage records outside
`projects/`. Short background calls (title generation, classifiers) are absent
too.

Measured against Claude's own `cost-state` across 89 closed sessions: ordinary
sessions under-report by a median of **4%**; sessions that fan out to subagents
by **22-99%**. `burn verify` shows the current split, and `⑂` marks affected
sessions in the live view.

Codex has no such gap: summing each `token_count` event's `last_token_usage`
reproduces the session's reported `total_token_usage` exactly.

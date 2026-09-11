# Repository history

On 2026-09-11, the Causal Final workspace became the active implementation on
`main` in [Axionis47/causal-inference-agent](https://github.com/Axionis47/causal-inference-agent).

## Preserved earlier implementation

The previous GitHub `main` is preserved on
[`codex/archive-main-before-causal-final-2026-09-11`](https://github.com/Axionis47/causal-inference-agent/tree/codex/archive-main-before-causal-final-2026-09-11).
Its tip at the transition was `b5e1a9e68a2724eb6b9df42d17f6d82140a4fae4`
(`docs: clarify causal copilot overview`). Its complete reachable commit history
is retained for examining earlier designs, failure modes and subsequent lessons.
Existing archive and feature branches are also retained.

## Active implementation

The new `main` retains the Causal Final workspace's 138 existing commits through
`72413c21fac837cead07bcdbe0361b13cf364e1a`, followed by a checkpoint of the
workspace's current source, tests, prompts, registries and documentation. The
previously uncommitted work is recorded together in that checkpoint; no earlier
commit sequence was fabricated for it.

The two implementations have independent Git histories. This transition preserves
each history on its own branch. Continue new work and checkpoint commits on
`main`. Use the [decision ledger](LEDGER.md) and task verification documents to
understand the reasons for changes and known limitations, including the recorded
complexity-budget failures.

For a direct file comparison between the preserved implementation and active work:

```bash
git fetch origin
git diff --stat origin/codex/archive-main-before-causal-final-2026-09-11 origin/main
git diff origin/codex/archive-main-before-causal-final-2026-09-11 origin/main -- path/to/file
```

Generated runs, local credentials, scratch files and local analysis caches remain
outside version control. Curated task budget reports and authored evaluation
findings are retained with the source.

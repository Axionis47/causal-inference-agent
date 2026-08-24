#!/bin/bash
# Stop hook: the Fable session may not end a turn while docs/ has uncommitted
# changes. Forces the checkpoint discipline: docs updated -> ledger updated ->
# committed. Opus sessions are exempt (Fable reviews and gates their work).

input=$(cat)
role="${CLAUDE_ROLE:-fable}"
[ "$role" = "fable" ] || exit 0

# Never loop: if we already blocked once this stop, let it through.
active=$(printf '%s' "$input" | jq -r '.stop_hook_active // false')
[ "$active" = "true" ] && exit 0

repo=$(cd "$(dirname "$0")/../.." && pwd)
cd "$repo" || exit 0
git rev-parse --git-dir >/dev/null 2>&1 || exit 0

if [ -n "$(git status --porcelain -- docs/ 2>/dev/null)" ]; then
  jq -n '{decision:"block",reason:"Checkpoint rule: docs/ has uncommitted changes. Before stopping: (1) make sure docs/LEDGER.md reflects the current task statuses and any decisions made this stretch, (2) commit the docs checkpoint (git add docs && git commit). Then stop."}'
  exit 0
fi
exit 0

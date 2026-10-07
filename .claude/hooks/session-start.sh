#!/bin/bash
# Runs when a Claude Code cloud session starts. On a laptop it does nothing.
#
# Two jobs:
#   1. The Vertex AI key. The environment holds strings, not files, so the key JSON rides in
#      GOOGLE_SERVICE_ACCOUNT_JSON; this writes it under .artifacts/ (git-ignored) and points
#      GOOGLE_APPLICATION_CREDENTIALS at it. VERTEX_PROJECT falls back to the key's project_id.
#   2. The dependencies, so make check works in the session.
set -euo pipefail

if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "$CLAUDE_PROJECT_DIR"

if [ -n "${GOOGLE_SERVICE_ACCOUNT_JSON:-}" ]; then
  mkdir -p .artifacts
  key=".artifacts/gcp-key.json"
  umask 077
  printf '%s' "$GOOGLE_SERVICE_ACCOUNT_JSON" > "$key"
  umask 022
  echo "export GOOGLE_APPLICATION_CREDENTIALS=\"$CLAUDE_PROJECT_DIR/$key\"" >> "$CLAUDE_ENV_FILE"
  if [ -z "${VERTEX_PROJECT:-}" ]; then
    project=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("project_id",""))' "$key")
    [ -n "$project" ] && echo "export VERTEX_PROJECT=\"$project\"" >> "$CLAUDE_ENV_FILE"
  fi
  echo "session-start: wrote the Vertex AI key to $key"
else
  echo "session-start: GOOGLE_SERVICE_ACCOUNT_JSON is not set; tests, lint and types work, a real run does not"
fi

uv sync --frozen
npm --prefix web install --no-audit --no-fund

#!/bin/bash
# PreToolUse write-ownership guard.
# Role comes from CLAUDE_ROLE in the launching terminal's environment:
#   unset / "fable" -> architect: may write only Markdown under docs/
#   "opus"          -> implementer: may write anything in the repo EXCEPT docs/,
#                      CLAUDE.md, and .claude/
# .claude/ and CLAUDE.md are user-owned: no agent role may modify them.

input=$(cat)
role="${CLAUDE_ROLE:-fable}"
tool=$(printf '%s' "$input" | jq -r '.tool_name // empty')
cwd=$(printf '%s' "$input" | jq -r '.cwd // empty')
repo=$(cd "$(dirname "$0")/../.." && pwd)

deny() {
  jq -n --arg r "$1" '{hookSpecificOutput:{hookEventName:"PreToolUse",permissionDecision:"deny",permissionDecisionReason:$r}}'
  exit 0
}

case "$tool" in
  Edit|Write|NotebookEdit)
    fp=$(printf '%s' "$input" | jq -r '.tool_input.file_path // .tool_input.notebook_path // empty')
    [ -z "$fp" ] && exit 0
    case "$fp" in
      /*) abs="$fp" ;;
      *)  abs="${cwd:-$repo}/$fp" ;;
    esac
    case "$abs" in
      *"/../"*|*"/.."|"") deny "write-guard: path traversal is not allowed" ;;
    esac
    case "$abs" in
      "$repo"/*) rel="${abs#"$repo"/}" ;;
      *) exit 0 ;;  # outside the repo (scratchpad, memory) is unrestricted
    esac
    case "$rel" in
      .claude/*|CLAUDE.md)
        deny "write-guard: .claude/ and CLAUDE.md are user-owned; no agent role may modify enforcement config" ;;
    esac
    if [ "$role" = "opus" ]; then
      case "$rel" in
        docs/*) deny "write-guard [OPUS]: docs/ is Fable-owned. PRDs and SYSTEM-CONTRACT.md are frozen. If documentation is wrong or missing, stop and return FABLE REVISION REQUIRED." ;;
        *) exit 0 ;;
      esac
    else
      case "$rel" in
        docs/*.md|docs/*/*.md|docs/*/*/*.md) exit 0 ;;
        *) deny "write-guard [FABLE]: writes are allowed only to Markdown files under docs/. Implementation files are Opus-owned; document the defect instead of patching it." ;;
      esac
    fi
    ;;
  Bash)
    cmd=$(printf '%s' "$input" | jq -r '.tool_input.command // empty')
    [ -z "$cmd" ] && exit 0
    mutates='sed[[:space:]]+-i|(^|[;&|[:space:]])(tee|rm|mv|cp|touch|truncate|patch|dd)[[:space:]]|[^<]>{1,2}[^&]'
    if [ "$role" = "opus" ]; then
      if printf '%s' "$cmd" | grep -Eq "$mutates" \
         && printf '%s' "$cmd" | grep -Eq '(^|[[:space:]"'"'"'=/])docs/|SYSTEM-CONTRACT|PRD-00|(^|[[:space:]"'"'"'=/])CLAUDE\.md|\.claude/'; then
        deny "write-guard [OPUS]: this shell command appears to modify docs/ or enforcement config, which are Fable/user-owned"
      fi
    else
      if printf '%s' "$cmd" | grep -Eq "$mutates" \
         && printf '%s' "$cmd" | grep -Eq '(^|[[:space:]"'"'"'=/])(src|tests|evals|tools|migrations|prompts)/|pyproject\.toml|uv\.lock|package\.json|\.claude/'; then
        deny "write-guard [FABLE]: this shell command appears to modify implementation files, which are Opus-owned"
      fi
    fi
    ;;
esac
exit 0

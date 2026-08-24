# T-001 — Repository scaffolding, dependency lock, budget checker

Status: READY_FOR_OPUS
Owner: Opus (terminal `causal-final-6a`)
Governing documents: `docs/product/SYSTEM-CONTRACT.md` (hash in `docs/LEDGER.md`), this spec.
Contract sections: §1.2 (runtime), §14 (pinned stack), §14.1 (complexity budget),
§14.1.1 (counting rule), §14.1.2 (coding-agent gate), acceptance criteria 23–26.

## Allowed-write files (exact; nothing else)

- `pyproject.toml`
- `uv.lock`
- `.python-version`
- `src/causal/__init__.py`
- `src/causal/{shared,cli,runtime,intake,design,preparation,estimation,presentation}/__init__.py` (all empty)
- `tools/budget_check.py`
- `tools/__init__.py` (empty, only if needed for imports)
- `tests/__init__.py`, `tests/tools/__init__.py` (empty, only if pytest requires them)
- `tests/tools/test_budget_check.py`
- `conftest.py` (only if pytest discovery requires it; otherwise omit)

## Deliverable 1 — environment

1. `.python-version` = `3.12.8`.
2. `pyproject.toml`:
   - `[project]` name `causal`, `requires-python == 3.12.8` compatible spec,
     dependencies = exactly the runtime pins of SYSTEM-CONTRACT §14
     (pydantic, kaggle, polars, psycopg[binary,pool], boto3, langgraph,
     langgraph-checkpoint-postgres, langsmith, google-genai, graphviz, numpy,
     scipy, pandas, scikit-learn, pyfixest, rdrobust, rddensity, altair,
     vl-convert-python — each `==` its exact §14 version).
   - Dev dependency group: `pytest==9.1.1`, `hypothesis==6.165.5`,
     `ruff==0.16.3`, `mypy==2.3.0`.
   - `[tool.ruff]`, `[tool.mypy]` (strict), `[tool.pytest.ini_options]` minimal.
   - Build backend: `uv_build` or `hatchling` — implementer's choice, note it.
3. `uv.lock` produced by `uv lock` with `uv==0.12.0` resolving the full set on
   Python 3.12. **If any exact pin fails to resolve, STOP — do not substitute a
   nearby version — and return `FABLE REVISION REQUIRED` naming the package,
   the pinned version, and the versions the index offers.**
4. Package skeleton: `src/causal/` with the eight subpackages above, all
   `__init__.py` empty (0 logical lines).

## Deliverable 2 — budget checker (`tools/budget_check.py`)

Implements counting rule `python-token-lines.v1` (SYSTEM-CONTRACT §14.1.1),
standard library only (`tokenize`, `ast`, `argparse`, `json`, `subprocess` for
git, `pathlib`). No third-party imports.

Behavior:

1. Python counting: a physical line counts when touched by ≥1 token other than
   ENCODING, NL, NEWLINE, INDENT, DEDENT, COMMENT, ENDMARKER. Every physical
   line spanned by a string/docstring token counts.
2. SQL/JSON/YAML/TOML/prompt-template counting: non-empty, non-comment
   physical lines (`--` for SQL, `#` for YAML/TOML; JSON has no comments).
3. Counts only git-tracked files at a named revision (`git ls-tree` /
   `git show`); working tree allowed via an explicit `--worktree` flag.
4. Scope assignment (§14.1 tables): each of the eight `src/causal/<pkg>/`
   rows; `tests/` → tests scope; `migrations/`, `evals/`, `prompts/` →
   declarative scope; `tools/` → tests scope (verification tooling; Fable
   decision D-003 in `docs/LEDGER.md`). A tracked implementation file matching
   none of these is reported as status `blocked_complexity_budget` with the
   unassigned path.
5. Module/function limits via AST: ≤50 production modules, ≤350 lines/module,
   ≤75 lines/function, per-package and 15,000/8,000/3,000/26,000 ceilings.
6. Output: one JSON `ImplementationBudgetReportV1` on stdout:
   `counting_rule_version`, `implementation_task_id`, `base_revision`,
   `proposed_revision`, per-scope `{baseline, added, deleted, projected, actual}`,
   `module_count`, `largest_module {path, lines}`, `largest_function
   {path, name, lines}`, `changed_files`, `breached_dimensions`, `status`
   (`within_budget` | `warning` | `rethink_required` | `blocked_complexity_budget`),
   `parent_rethink_id` (nullable). `warning` at ≥80% of any ceiling.
7. CLI: `python tools/budget_check.py --task-id ID --base REV
   [--proposed REV | --worktree] [--json]`.

## Line budget for this task

- `tools/budget_check.py` ≤ 300 logical lines (module ceiling 350 applies).
- `tests/tools/test_budget_check.py` ≤ 300 logical lines.
- All `__init__.py` files: 0 logical lines.

## Tests (required)

`tests/tools/test_budget_check.py`, pytest, covering at minimum:
docstring lines all count; comment-only and blank lines do not; multi-line
string spans count fully; YAML/SQL comment exclusion; scope assignment for one
file per scope; function-size measurement via a fixture snippet; the 80%
warning threshold; unassigned-file blocker; report JSON contains every field
above. Use tmp git repos or in-memory strings; no network.

## Forbidden work

Any file outside the allowed-write list; any application/domain logic; README
or docs; CI configuration; Dockerfile/OCI image; extra dependencies; relaxing
any pin; pre-commit config; modifying `docs/`, `CLAUDE.md`, `.claude/`.

## Completion evidence (return via SendMessage to the Fable session)

1. Statement that you verified the SYSTEM-CONTRACT hash from `docs/LEDGER.md`
   and this spec's hash (given in the dispatch message) before writing.
2. `uv lock` success: total resolved package count.
3. `uv run pytest` output summary (all green).
4. `uv run ruff check .` and `uv run mypy tools/` clean.
5. Budget-checker self-report: run it on your own change set; paste the JSON.
6. Changed-file list and the git commit hash (commit only allowed files;
   message starts `T-001:`).

# Skill: Documentation Coverage Tracker

Load when: tracking which files and components are documented.

## What to track per file

For every .py file in backend/src/:
- Has module docstring? (yes/no)
- Has class docstrings? (yes/no per class)
- Referenced in any docs/*.md? (yes/no)
- Has unit tests? (yes/no — check tests/ for matching test file)
- Has agentic eval? (yes/no — check benchmarks/agentic_evals/)

## Output format

```markdown
## Documentation Coverage

### Summary
- Files: X total, Y documented (Z%)
- Classes: X total, Y with docstrings (Z%)
- Test coverage: X files tested out of Y (Z%)

### Per-Directory Coverage

#### backend/src/agents/specialists/
| File | Docstring | In docs/ | Unit tests | Eval |
|------|-----------|----------|------------|------|
| data_profiler.py | yes | agents.md | yes (31) | yes (11) |
| sensitivity_analyst.py | yes | agents.md | no | yes (8) |
| dag_expert.py | yes | agents.md | no | no |

#### backend/src/causal/methods/
| File | Docstring | In docs/ | Unit tests |
|------|-----------|----------|------------|
| ols.py | yes | causal-methods.md | yes |
| iv.py | yes | causal-methods.md | yes |
```

## Update cadence

Re-run coverage tracker:
- After every sprint
- After adding/removing files
- After adding/removing tests

# Skill: Dead Code Scan

Load when: scanning for unused imports, unreferenced functions, duplicate components, orphaned files.

## What to scan

Every .py file in backend/src/ and every .ts/.tsx file in frontend/src/.

## What counts as dead code

### 1. Unused imports
An import statement where the imported name is never referenced in that file.
```python
from src.agents.base import AnalysisState  # but AnalysisState never used below
```
Check: grep for the imported name in the rest of the file.

### 2. Unreferenced functions
A function or method defined but never called anywhere in the entire codebase.
```python
def _old_helper():  # defined in file A, never called in ANY file
```
Check: grep for the function name across ALL files. If zero hits outside its definition, it's dead.
Exception: methods starting with `test_` (pytest discovers them). Methods in `__init__` or `__all__`.

### 3. Duplicate components
Same logic implemented in 2+ places.
Example: two different effect estimator agents doing the same thing.
Check: look for files with very similar names or classes with overlapping WRITES_STATE_FIELDS.

### 4. Orphaned files
A .py file that is never imported by any other file.
Check: grep for the module name across all files. If zero imports, it's orphaned.
Exception: entry points (main.py, run_*.py), test files, __init__.py.

## What to NOT flag

- Test files and fixtures
- __init__.py files (even if they only re-export)
- Entry points: main.py, app.py, run_*.py
- Config files: settings.py, pyproject.toml
- Anything in __all__ exports
- Abstract base class methods (meant to be overridden)
- Registered handlers (discovered via decorator, not direct import)

## Output format

```markdown
## Dead Code Report

### Unused Imports (certain)
| File | Line | Import | Confidence |
|------|------|--------|------------|
| src/agents/foo.py | 12 | from bar import Baz | certain |

### Unreferenced Functions (check manually)
| File | Line | Function | References Found |
|------|------|----------|-----------------|
| src/agents/foo.py | 45 | _old_method | 0 |

### Orphaned Files (certain)
| File | Imported By |
|------|------------|
| src/agents/old_agent.py | nobody |

### Duplicates (check manually)
| File A | File B | What's duplicated |
|--------|--------|------------------|
```

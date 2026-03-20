# Skill: Regression Check

Load when: verifying no existing functionality broke.

## What to check

### Agent registry
```bash
cd backend
python3 -c "from src.agents.registry import create_all_agents; agents = create_all_agents(); print(f'{len(agents)} agents: {sorted(agents.keys())}')"
# Expected: 13 agents
```

### Method registry
```bash
python3 -c "from src.causal.methods.base import get_method_registry; methods = get_method_registry(); print(f'{len(methods)} methods: {sorted(methods.keys())}')"
# Expected: 12 methods
```

### Unit tests
```bash
cd backend && pytest tests/unit -q --tb=short
# Expected: 381 tests passing
```

### Frontend tests
```bash
cd frontend && npx vitest run
```

### Import check
```bash
python3 -c "from src.agents.base.state import AnalysisState; print('State OK')"
python3 -c "from src.api.main import app; print('API OK')"
```

## Regression = any of these

1. Previously passing test now fails
2. Agent count changed unexpectedly
3. Method count changed unexpectedly
4. Import errors in core modules
5. API endpoints return different status codes

## Output format

```
## Regression Check — [date]

Agents: X registered (expected: 13)
Methods: X registered (expected: 12)
Tests: X passing, Y failing (expected: 381 passing)
Imports: all OK / [list failures]

Regressions found: [none | list specific failures]
```

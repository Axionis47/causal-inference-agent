# Skill: CI Workflow

Load when: creating or modifying CI/CD workflows.

## CI checks

```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]
jobs:
  backend:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install -r backend/requirements.txt
      - run: cd backend && pytest tests/unit -q --tb=short

  frontend:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
      - run: cd frontend && npm ci
      - run: cd frontend && npx vitest run

  lint:
    runs-on: ubuntu-latest
    steps:
      - run: cd backend && ruff check src/
      - run: cd frontend && npm run lint
```

## Rules

1. Unit tests must pass on every PR
2. Never skip tests in CI
3. Use -q --tb=short for pytest (not -v)
4. Mock all LLM calls — no API keys needed in CI
5. Frontend tests use vitest
6. Lint check: ruff for Python, eslint for TypeScript

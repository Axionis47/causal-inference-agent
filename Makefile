# One way to run things. `make` alone lists the targets.

.DEFAULT_GOAL := help
UV ?= uv run
WEB := --prefix web

help:  ## list the targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F ':.*## ' '{printf "  %-12s %s\n", $$1, $$2}'

# ------------------------------------------------------------------ checks

lint: ## ruff, the layer contract, eslint and prettier
	$(UV) ruff check causal_agent
	$(UV) ruff format --check causal_agent
	$(UV) lint-imports
	npm run $(WEB) lint

fmt: ## ruff and prettier, in place
	$(UV) ruff check --fix causal_agent
	$(UV) ruff format causal_agent
	npm run $(WEB) format

types: ## mypy on the package, tsc on the web
	$(UV) mypy
	npm run $(WEB) typecheck

test: ## the Python suite
	$(UV) pytest causal_agent -q

test-web: ## the web suite
	npm test $(WEB) -- --run

schema: ## regenerate the OpenAPI schema and the web's types from it
	$(UV) python -m causal_agent.server --openapi > web/openapi.json
	npm run $(WEB) types

check: lint types test test-web schema ## everything CI runs

# ------------------------------------------------------------------ running

dev-api: ## the API on :8000
	$(UV) python -m causal_agent.server

dev-web: ## the web dev server on :5173, proxying /api
	npm run $(WEB) dev

build-web: ## the web bundle the API serves
	npm run $(WEB) build

evals: ## run one family's evals: make evals FAMILY=discontinuity
	$(UV) python -m causal_agent.evals.run $(FAMILY)

.PHONY: help lint fmt types test test-web schema check dev-api dev-web build-web evals

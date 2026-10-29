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

ladder: ## write docs/ladders/<family>.md from each lane's declarations; CI fails on a diff
	LANGSMITH_TRACING=false $(UV) python -m causal_agent.evals.ladder

check: lint types test test-web schema ladder ## everything CI runs

# ------------------------------------------------------------------ running

dev-api: ## the API on :8000
	$(UV) python -m causal_agent.server

dev-web: ## the web dev server on :5173, proxying /api
	npm run $(WEB) dev

studio: ## langgraph dev on :2024 for LangGraph Studio, with the server's pickled threads wiped first
	rm -rf .langgraph_api && $(UV) langgraph dev --no-browser --port 2024

build-web: ## the web bundle the API serves
	npm run $(WEB) build

evals: ## run one family's evals: make evals FAMILY=discontinuity
	$(UV) python -m causal_agent.evals.run $(FAMILY)

explore-image: ## build the required exploration and diagnostic Docker image
	docker build -t causal-desk-explore -f docker/explore.Dockerfile .

container-build: ## build the application and restricted script worker
	docker compose build app sandbox

container-up: ## start the local application container (persistent named volumes)
	docker compose up -d --wait app

container-down: ## stop the application container, keeping its data volumes
	docker compose down

container-smoke: ## build current checkout and test browser/storage in disposable volumes
	$(UV) python docker/smoke.py

container-sandbox: ## verify the enforcing worker through the running application
	docker compose exec -T app python /usr/local/bin/causal-sandbox-probe.py

docfigs: ## redraw docs/diagrams/*.svg from the current declared trajectories
	LANGSMITH_TRACING=false $(UV) python -m causal_agent.evals.docfigs

trace: ## the newest LangSmith trace as a tree in the terminal: make trace ARGS="--last 8" | ARGS="--thoughts"
	$(UV) python -m causal_agent.evals.trace $(ARGS)

.PHONY: help lint fmt types test test-web schema ladder check dev-api dev-web build-web evals explore-image container-build container-up container-down container-smoke container-sandbox docfigs studio trace

ingestion-init: ## initialize the ingestion database and private bucket using configured endpoints
	$(UV) python -m causal_agent.server.ingestion init

ingestion-cleanup: ## reconcile unpublished ingestion objects older than 24 hours
	$(UV) python -m causal_agent.server.ingestion cleanup

.PHONY: ingestion-init ingestion-cleanup


test-ingestion: ## required real ingestion tests in harness-owned Docker storage
	LANGSMITH_TRACING=false $(UV) pytest causal_agent/server/tests/test_ingestion.py causal_agent/server/tests/library --ingestion-integration -q

test-ingestion-mutations: ## require baseline green and named failures for deliberate defects
	LANGSMITH_TRACING=false $(UV) python scripts/ingestion_mutations.py
	$(UV) python scripts/ingestion_web_mutations.py

.PHONY: test-ingestion test-ingestion-mutations

test-library: ## Phase 2 owned library with real isolated storage
	LANGSMITH_TRACING=false $(UV) pytest causal_agent/server/tests/library --ingestion-integration -q

.PHONY: test-library


test-profiling: ## Phase 3 computation and isolated real profile storage
	LANGSMITH_TRACING=false $(UV) pytest causal_agent/profile/tests causal_agent/server/tests/profiling --ingestion-integration -q

test-storage: ## Phases 1-4 using one isolated storage harness
	LANGSMITH_TRACING=false $(UV) pytest causal_agent/profile/tests causal_agent/server/tests/test_ingestion.py causal_agent/server/tests/library causal_agent/server/tests/profiling causal_agent/memory/context/tests causal_agent/server/tests/context --ingestion-integration -q

.PHONY: test-profiling test-storage


test-context: ## Phase 4 semantic extraction and owned, isolated context storage
	LANGSMITH_TRACING=false $(UV) pytest causal_agent/memory/context/tests causal_agent/server/tests/context --ingestion-integration -q

.PHONY: test-context

verify-phase4: ## all Phase 4 checks and one report; WITH_MODEL=1 opts into Gemini/trace acceptance
	$(UV) python scripts/verify_phase4.py $(if $(filter 1,$(WITH_MODEL)),--with-model) $(if $(filter 1,$(ALL_CONTROLS)),--all-controls)

.PHONY: verify-phase4

verify-chat: ## consolidated chat checks; WITH_MODEL=1 also runs Gemini conversations and trace readback
	$(UV) python -m scripts.verify_chat $(if $(filter 1,$(WITH_MODEL)),--with-model,)

.PHONY: verify-chat

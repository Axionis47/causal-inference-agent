.PHONY: setup dev dev-backend dev-frontend test test-backend test-frontend lint typecheck docker-up docker-down docker-build docker-prod docker-full clean help

# Default target
help:
	@echo "Causal Orchestrator — Development Commands"
	@echo ""
	@echo "  make setup          Install all dependencies"
	@echo "  make dev            Start backend + frontend for local dev"
	@echo "  make dev-backend    Start backend only (uvicorn --reload)"
	@echo "  make dev-frontend   Start frontend only (vite dev)"
	@echo "  make test           Run all tests"
	@echo "  make test-backend   Run backend tests only"
	@echo "  make test-frontend  Run frontend tests only"
	@echo "  make lint           Lint backend + frontend"
	@echo "  make typecheck      Type-check backend + frontend"
	@echo "  make docker-up      Start services via Docker Compose"
	@echo "  make docker-down    Stop Docker Compose services"
	@echo "  make docker-build   Build Docker images"
	@echo "  make docker-prod    Start production stack with nginx proxy (port 80)"
	@echo "  make docker-full    Start full stack with Redis"
	@echo "  make clean          Remove build artifacts"
	@echo ""

# ─── Setup ───────────────────────────────────────────────────────────────────
setup:
	@echo "Installing backend dependencies..."
	cd backend && pip install -r requirements.txt
	cd backend && pip install pytest pytest-asyncio pytest-cov pytest-timeout httpx
	@echo "Installing frontend dependencies..."
	cd frontend && npm ci
	@echo ""
	@echo "Done! Copy .env.example to .env and add your API keys:"
	@echo "  cp .env.example .env"

# ─── Development ─────────────────────────────────────────────────────────────
dev-backend:
	cd backend && uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

dev-frontend:
	cd frontend && npm run dev

dev:
	@echo "Starting backend and frontend..."
	@echo "Backend: http://localhost:8000"
	@echo "Frontend: http://localhost:5173"
	@echo ""
	$(MAKE) dev-backend &
	$(MAKE) dev-frontend

# ─── Testing ─────────────────────────────────────────────────────────────────
test-backend:
	cd backend && python3 -m pytest tests/unit -v --tb=short

test-frontend:
	cd frontend && npx vitest run

test: test-backend test-frontend

test-integration:
	cd backend && python3 -m pytest tests/integration -v --tb=short

test-coverage:
	cd backend && python3 -m pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=xml
	cd frontend && npx vitest run --coverage

# ─── Linting ─────────────────────────────────────────────────────────────────
lint:
	cd backend && ruff check src/
	cd frontend && npm run lint

typecheck:
	cd backend && mypy src/ --ignore-missing-imports || true
	cd frontend && npx tsc --noEmit

# ─── Docker ──────────────────────────────────────────────────────────────────
docker-build:
	docker compose build

docker-up:
	docker compose up -d
	@echo "Proxy:    http://localhost (port 80)"
	@echo "Backend:  http://localhost:8000"
	@echo "Frontend: http://localhost:3000"

docker-down:
	docker compose down

docker-prod:
	docker compose up -d
	@echo "Application running at http://localhost"

docker-full:
	docker compose --profile full up -d
	@echo "Application running at http://localhost (with Redis)"

docker-dev:
	docker compose --profile dev up

# ─── Cleanup ─────────────────────────────────────────────────────────────────
clean:
	find backend -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find backend -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find backend -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf backend/.coverage backend/coverage.xml
	rm -rf frontend/dist frontend/coverage
	@echo "Cleaned build artifacts."

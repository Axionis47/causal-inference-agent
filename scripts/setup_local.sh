#!/bin/bash
# Local development setup script for Causal Orchestrator

set -e

echo "Setting up Causal Orchestrator local development environment..."
echo ""

# ─── Check prerequisites ────────────────────────────────────────────────────

# Check Python 3.11+
if command -v python3 >/dev/null 2>&1; then
    PY_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
    PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)
    if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 11 ]; }; then
        echo "ERROR: Python 3.11+ is required (found Python $PY_VERSION)."
        exit 1
    fi
    echo "Found Python $PY_VERSION"
else
    echo "ERROR: Python 3 is required but not installed."
    exit 1
fi

# Check Node 20+
if command -v node >/dev/null 2>&1; then
    NODE_VERSION=$(node -v | sed 's/v//' | cut -d. -f1)
    if [ "$NODE_VERSION" -lt 20 ]; then
        echo "ERROR: Node.js 20+ is required (found Node $(node -v))."
        exit 1
    fi
    echo "Found Node.js $(node -v)"
else
    echo "ERROR: Node.js is required but not installed."
    exit 1
fi

echo ""

# ─── Environment file ───────────────────────────────────────────────────────

if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp .env.example .env

    echo ""
    echo "Please enter your API keys (leave blank to skip and set later):"

    read -rp "  CLAUDE_API_KEY: " CLAUDE_KEY
    if [ -n "$CLAUDE_KEY" ]; then
        sed -i.bak "s/^CLAUDE_API_KEY=.*/CLAUDE_API_KEY=$CLAUDE_KEY/" .env && rm -f .env.bak
    fi

    read -rp "  KAGGLE_KEY: " KAG_KEY
    if [ -n "$KAG_KEY" ]; then
        sed -i.bak "s/^KAGGLE_KEY=.*/KAGGLE_KEY=$KAG_KEY/" .env && rm -f .env.bak
    fi

    read -rp "  KAGGLE_USERNAME: " KAG_USER
    if [ -n "$KAG_USER" ]; then
        sed -i.bak "s/^KAGGLE_USERNAME=.*/KAGGLE_USERNAME=$KAG_USER/" .env && rm -f .env.bak
    fi

    echo ""
    echo ".env created. You can edit it later to add or change values."
else
    echo ".env already exists — skipping."
fi

echo ""

# ─── Initialize data directory ──────────────────────────────────────────────

mkdir -p ./data
echo "Initialized ./data directory for local storage."

# ─── Install backend dependencies ───────────────────────────────────────────

echo ""
echo "Installing backend dependencies..."
cd backend
pip install -r requirements.txt
pip install pytest pytest-asyncio pytest-cov pytest-timeout httpx
cd ..

# ─── Install frontend dependencies ──────────────────────────────────────────

echo ""
echo "Installing frontend dependencies..."
cd frontend
npm ci
cd ..

# ─── Done ────────────────────────────────────────────────────────────────────

echo ""
echo "Setup complete! To start development:"
echo ""
echo "  Using Make:"
echo "    make dev            # Start both backend + frontend"
echo "    make dev-backend    # Start backend only"
echo "    make dev-frontend   # Start frontend only"
echo ""
echo "  Using Docker:"
echo "    docker compose --profile dev up"
echo ""
echo "  Manual:"
echo "    cd backend && uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload"
echo "    cd frontend && npm run dev"
echo ""

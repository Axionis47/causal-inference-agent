#!/bin/bash
# Causal Orchestrator — Local Startup
# Usage: ./start.sh

set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
BACKEND="$ROOT/backend"
FRONTEND="$ROOT/frontend"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
GRAY='\033[0;90m'
NC='\033[0m'

echo ""
echo -e "${GREEN}Causal Inference Orchestrator${NC}"
echo -e "${GRAY}────────────────────────────────────────${NC}"

# Check .env
if [ ! -f "$BACKEND/.env" ]; then
  echo "ERROR: backend/.env not found. Copy .env.example and add your API keys."
  exit 1
fi

# Kill any existing processes on our ports
lsof -ti:8000 2>/dev/null | xargs kill -9 2>/dev/null || true
lsof -ti:5173 2>/dev/null | xargs kill -9 2>/dev/null || true

# Start backend
echo -e "${BLUE}Starting backend...${NC}  http://localhost:8000"
cd "$BACKEND"
.venv/bin/uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload --log-level warning &
BACKEND_PID=$!

# Start frontend
echo -e "${BLUE}Starting frontend...${NC} http://localhost:5173"
cd "$FRONTEND"
npx vite --host 0.0.0.0 --port 5173 &
FRONTEND_PID=$!

echo -e "${GRAY}────────────────────────────────────────${NC}"
echo -e "${GREEN}Ready.${NC}"
echo -e "  Backend:  ${BLUE}http://localhost:8000${NC}  (API docs: http://localhost:8000/docs)"
echo -e "  Frontend: ${BLUE}http://localhost:5173${NC}"
echo ""
echo -e "${GRAY}Press Ctrl+C to stop both.${NC}"

# Cleanup on exit
cleanup() {
  echo ""
  echo -e "${GRAY}Shutting down...${NC}"
  kill $BACKEND_PID 2>/dev/null
  kill $FRONTEND_PID 2>/dev/null
  wait $BACKEND_PID 2>/dev/null
  wait $FRONTEND_PID 2>/dev/null
  echo -e "${GREEN}Done.${NC}"
}
trap cleanup EXIT INT TERM

# Wait for either to exit
wait

#!/bin/bash
# Local development setup script

set -e

echo "Setting up Causal Orchestrator local development environment..."

# Check prerequisites
command -v python3 >/dev/null 2>&1 || { echo "Python 3 is required but not installed."; exit 1; }
command -v node >/dev/null 2>&1 || { echo "Node.js is required but not installed."; exit 1; }

# Create virtual environment
echo "Creating Python virtual environment..."
cd backend
python3 -m venv venv
source venv/bin/activate

# Install backend dependencies
echo "Installing backend dependencies..."
pip install -r requirements.txt
pip install pytest pytest-asyncio pytest-cov pytest-timeout httpx

# Install frontend dependencies
echo "Installing frontend dependencies..."
cd ../frontend
npm install

# Setup environment file
cd ..
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo ""
    echo "IMPORTANT: Edit .env file with your API keys:"
    echo "  - GEMINI_API_KEY"
    echo "  - KAGGLE_KEY"
    echo "  - KAGGLE_USERNAME"
fi

echo ""
echo "Setup complete! To start development:"
echo ""
echo "  Backend:"
echo "    cd backend && source venv/bin/activate"
echo "    uvicorn src.api.main:app --reload"
echo ""
echo "  Frontend:"
echo "    cd frontend && npm run dev"
echo ""
echo "  Or use Docker:"
echo "    docker-compose --profile dev up"

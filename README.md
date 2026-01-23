# Causal Orchestrator

An agentic causal inference system that automatically analyzes datasets from Kaggle URLs and generates reproducible Jupyter notebooks with treatment effect estimates.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              React Frontend                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ URL Input   │  │ Job Status  │  │ Results     │  │ Agent Traces       │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           FastAPI Backend                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     Orchestrator Agent (Gemini)                      │   │
│  │  - Coordinates analysis flow via LLM reasoning                       │   │
│  │  - NO hardcoded if-else logic                                        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│       │              │                │               │                     │
│       ▼              ▼                ▼               ▼                     │
│  ┌─────────┐  ┌────────────┐  ┌─────────────┐  ┌───────────────┐           │
│  │ Data    │  │ Effect     │  │ Sensitivity │  │ Notebook      │           │
│  │ Profiler│  │ Estimator  │  │ Analyst     │  │ Generator     │           │
│  └─────────┘  └────────────┘  └─────────────┘  └───────────────┘           │
│                      │                                                      │
│                      ▼                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                       Critique Agent (Gemini)                        │   │
│  │  - Reviews all outputs for quality                                   │   │
│  │  - Can request iterations or reject                                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Key Features

- **Truly Agentic**: All method selection decisions made by Gemini LLM, no hardcoded rules
- **12 Causal Methods**: OLS, PSM, IPW, AIPW, DiD, IV/2SLS, RDD, Meta-learners, Causal Forest, Double ML
- **Critique Loop**: Quality control agent reviews and can iterate on analysis
- **8 Benchmark Datasets**: Stress-tested on diverse causal inference scenarios
- **Auto-scaling**: Deploys to GCP Cloud Run with automatic scaling
- **Full Observability**: Agent traces, structured logging, metrics

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 20+
- Docker (optional)
- GCP account (for production)

### Local Development

1. **Clone and setup environment**
   ```bash
   cd causal-orchestrator
   cp .env.example .env
   # Edit .env with your API keys
   ```

2. **Start backend**
   ```bash
   cd backend
   pip install -r requirements.txt
   uvicorn src.api.main:app --reload
   ```

3. **Start frontend**
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

4. **Access the app**
   - Frontend: http://localhost:5173
   - Backend API: http://localhost:8000
   - API Docs: http://localhost:8000/docs

### Using Docker

```bash
# Production mode
docker-compose up

# Development mode with hot reload
docker-compose --profile dev up
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `POST /api/v1/jobs` | Create | Submit Kaggle URL for analysis |
| `GET /api/v1/jobs/{id}` | Read | Get job details and status |
| `GET /api/v1/jobs/{id}/results` | Read | Get analysis results |
| `GET /api/v1/jobs/{id}/notebook` | Read | Download Jupyter notebook |
| `GET /api/v1/jobs/{id}/traces` | Read | View agent reasoning traces |
| `GET /health` | Read | Health check |

## Causal Methods

| Method | Estimand | Use Case |
|--------|----------|----------|
| OLS Regression | ATE | Baseline, experimental data |
| Propensity Score Matching | ATT | Observational data, selection bias |
| Inverse Probability Weighting | ATE | Covariate adjustment |
| Doubly Robust (AIPW) | ATE | Robust to model misspecification |
| Difference-in-Differences | ATT | Panel data, policy changes |
| Instrumental Variables (2SLS) | LATE | Endogeneity, compliance issues |
| Regression Discontinuity | LATE | Sharp/fuzzy cutoff designs |
| S-Learner | CATE | Heterogeneous effects |
| T-Learner | CATE | Heterogeneous effects |
| X-Learner | CATE | Imbalanced treatment groups |
| Causal Forest | CATE | Non-linear heterogeneity |
| Double ML | ATE | High-dimensional covariates |

## Benchmark Datasets

| Dataset | Type | Ground Truth |
|---------|------|--------------|
| IHDP | Heterogeneous Effects | Yes (simulated) |
| LaLonde/NSW | Propensity Score | Yes (RCT benchmark) |
| Twins | Counterfactual | Yes (twin pairs) |
| Card IV | Instrumental Variable | No (real-world) |
| Card-Krueger | Difference-in-Differences | No (natural experiment) |
| ACIC 2016 | High-dimensional | Yes (simulated) |
| Climate | Time Series | Yes (known physics) |
| News | Continuous Treatment | Yes (simulated) |

### Running Benchmarks

```bash
cd backend
python -m benchmarks.runners.benchmark_runner
```

## Testing

```bash
# Unit tests
cd backend
pytest tests/unit -v

# Integration tests
pytest tests/integration -v

# Benchmark tests
pytest benchmarks/runners/test_benchmark.py -v

# All tests with coverage
pytest --cov=src --cov-report=html
```

## Deployment

### GCP Cloud Run

1. **Authenticate**
   ```bash
   gcloud auth login
   gcloud config set project plotpointe
   ```

2. **Deploy**
   ```bash
   # Backend
   gcloud run deploy causal-backend \
     --source ./backend \
     --region us-central1 \
     --allow-unauthenticated \
     --set-secrets GEMINI_API_KEY=gemini-api-key:latest

   # Frontend
   gcloud run deploy causal-frontend \
     --source ./frontend \
     --region us-central1 \
     --allow-unauthenticated
   ```

### CI/CD

The project includes GitHub Actions workflows:

- **CI** (`ci.yml`): Lint, test, build on every push
- **CD** (`cd.yml`): Deploy staging → benchmark validation → production

## Project Structure

```
causal-orchestrator/
├── backend/
│   ├── src/
│   │   ├── agents/           # Agentic system
│   │   │   ├── orchestrator/ # Central coordinator
│   │   │   ├── specialists/  # Domain experts
│   │   │   └── critique/     # Quality reviewer
│   │   ├── api/              # FastAPI endpoints
│   │   ├── config/           # Configuration
│   │   ├── llm/              # Gemini client
│   │   ├── storage/          # Firestore client
│   │   └── jobs/             # Job management
│   ├── benchmarks/           # 8 benchmark datasets
│   └── tests/                # Unit & integration tests
├── frontend/
│   ├── src/
│   │   ├── components/       # React components
│   │   ├── pages/            # Page components
│   │   └── services/         # API client
│   └── public/
├── .github/workflows/        # CI/CD pipelines
├── docker-compose.yml        # Local development
└── README.md
```

## Configuration

| Variable | Description | Required |
|----------|-------------|----------|
| `GEMINI_API_KEY` | Google Gemini API key | Yes |
| `KAGGLE_KEY` | Kaggle API key | Yes |
| `KAGGLE_USERNAME` | Kaggle username | Yes |
| `GCP_PROJECT_ID` | GCP project for deployment | For prod |
| `ENVIRONMENT` | development/staging/production | No |
| `MAX_ITERATIONS` | Max critique iterations | No (default: 3) |

## License

MIT

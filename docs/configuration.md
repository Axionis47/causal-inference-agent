# Configuration Reference

All settings are defined in a single `Settings` class using Pydantic `BaseSettings`. Values are loaded from environment variables or a `.env` file in the project root.

Source: `backend/src/config/settings.py`

---

## Environment Setup

Create a `.env` file in `backend/`:

```bash
# Required for Claude (default provider)
CLAUDE_API_KEY=sk-ant-...
# Or use the alias:
ANTHROPIC_API_KEY=sk-ant-...

# Required for Kaggle dataset downloads
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_key
```

Settings are loaded with `case_sensitive=False` and `extra="ignore"`, so unknown variables are silently ignored. The settings instance is cached via `@lru_cache` and created once per process.

---

## Configuration Reference

### Application

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `APP_NAME` | string | `"Causal Inference Orchestrator"` | Application name |
| `APP_VERSION` | string | `"1.0.0"` | Version string |
| `ENVIRONMENT` | enum | `"development"` | `"development"`, `"staging"`, or `"production"` |
| `DEBUG` | bool | `false` | Enable debug mode |
| `LOG_LEVEL` | string | `"INFO"` | Logging level |

### Server

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `HOST` | string | `"0.0.0.0"` | Server bind address |
| `PORT` | int | `8000` | Server port |

### LLM

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `LLM_PROVIDER` | enum | `"claude"` | `"claude"`, `"gemini"`, or `"vertex"` |

**Claude (default provider):**

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `CLAUDE_API_KEY` | secret | (none) | Anthropic API key. Also accepts `ANTHROPIC_API_KEY`. |
| `CLAUDE_MODEL` | string | `"claude-sonnet-4-20250514"` | Model identifier |
| `CLAUDE_TEMPERATURE` | float | `0.1` | Sampling temperature |
| `CLAUDE_MAX_TOKENS` | int | `8192` | Maximum output tokens |

**Gemini (direct API):**

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `GEMINI_API_KEY` | secret | (none) | Google AI API key |
| `GEMINI_MODEL` | string | `"gemini-1.5-pro"` | Model identifier |
| `GEMINI_TEMPERATURE` | float | `0.1` | Sampling temperature |
| `GEMINI_MAX_TOKENS` | int | `8192` | Maximum output tokens |

**Vertex AI (GCP-managed):**

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `VERTEX_MODEL` | string | `"gemini-2.0-flash-exp"` | Model identifier |

Vertex AI uses Application Default Credentials (ADC) or a service account. No explicit API key is needed; set `GCP_PROJECT_ID` and `GCP_REGION` instead.

### Storage

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `USE_FIRESTORE` | bool | `false` | Use Firestore (true) or local JSON files (false) |
| `LOCAL_STORAGE_PATH` | string | `"./data"` | Path for local JSON storage when Firestore is off |
| `FIRESTORE_DATABASE` | string | `"(default)"` | Firestore database name |
| `GCS_BUCKET_DATASETS` | string | `"causal-datasets"` | GCS bucket for datasets |
| `GCS_BUCKET_NOTEBOOKS` | string | `"causal-notebooks"` | GCS bucket for notebooks |

### GCP

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `GCP_PROJECT_ID` | string | `""` | GCP project ID. Required in production with Firestore. |
| `GCP_REGION` | string | `"us-central1"` | GCP region for Vertex AI |

### Kaggle

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `KAGGLE_USERNAME` | string | `""` | Kaggle username for dataset downloads |
| `KAGGLE_KEY` | secret | (none) | Kaggle API key |

### Authentication

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `API_KEY` | secret | (none) | API key for `X-API-Key` header auth. When unset, auth is disabled (dev mode). Empty strings are treated as unset. |

### Agents

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `MAX_AGENT_ITERATIONS` | int | `3` | Maximum critique-iterate cycles per job |
| `AGENT_TIMEOUT_SECONDS` | int | `300` | Timeout per agent execution (seconds). Total job timeout is 10x this value. |

### SSE (Server-Sent Events)

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `SSE_ENABLED` | bool | `true` | Enable the SSE streaming endpoint |
| `SSE_HEARTBEAT_SECONDS` | int | `15` | Heartbeat interval for SSE keep-alive |

### Redis

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `REDIS_URL` | string | `"redis://localhost:6379"` | Redis connection URL |
| `REDIS_ENABLED` | bool | `false` | Enable Redis (reserved for future job queue scaling) |

### CORS

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `CORS_ORIGINS` | list[string] | `["http://localhost:3000", "http://localhost:5173", "http://localhost:80", "http://localhost"]` | Allowed CORS origins. Set via `CORS_ORIGINS` env var in production. |

### API Documentation

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `ENABLE_API_DOCS` | bool | `true` | Enable Swagger UI (`/docs`) and ReDoc (`/redoc`) |

---

## Provider Setup

### Claude (Anthropic)

1. Get an API key from [console.anthropic.com](https://console.anthropic.com/)
2. Set `CLAUDE_API_KEY` or `ANTHROPIC_API_KEY` in `.env`
3. (Optional) Change the model via `CLAUDE_MODEL`

This is the default provider. No other configuration is needed.

### Gemini (Google AI)

1. Get an API key from [aistudio.google.com](https://aistudio.google.com/)
2. Set `GEMINI_API_KEY` in `.env`
3. Set `LLM_PROVIDER=gemini` in `.env`

### Vertex AI (GCP)

1. Set up a GCP project with Vertex AI API enabled
2. Configure Application Default Credentials: `gcloud auth application-default login`
3. Set `GCP_PROJECT_ID` and `GCP_REGION` in `.env`
4. Set `LLM_PROVIDER=vertex` in `.env`

### Firestore (Production Storage)

1. Enable Firestore in your GCP project
2. Set `USE_FIRESTORE=true` in `.env`
3. Set `GCP_PROJECT_ID` in `.env`
4. Ensure ADC or service account credentials are available

---

## Development vs Production

| Aspect | Development | Production |
|--------|-------------|------------|
| Storage | Local JSON files (`./data`) | Firestore |
| Auth | Disabled (no `API_KEY`) | `API_KEY` required (warning logged if missing) |
| LLM | Claude with personal API key | Claude/Vertex with service credentials |
| CORS | Localhost origins | Cloud Run URLs |
| Hot reload | Enabled (`uvicorn --reload`) | Disabled |
| Logging | Console output | JSON structured logs |
| GCP validation | Skipped | `GCP_PROJECT_ID` required when Firestore is on |

---

## Validation Rules

The `Settings` class runs two validators after loading:

1. **API key alias resolution**: If `CLAUDE_API_KEY` is not set but `ANTHROPIC_API_KEY` is, the latter is used. Empty-string `API_KEY` values are treated as unset.

2. **Production validation**: In production (`ENVIRONMENT=production`):
   - `GCP_PROJECT_ID` is required when `USE_FIRESTORE=true` (raises `ValueError`)
   - Missing `API_KEY` logs a warning but does not block startup
   - Missing LLM API keys for the selected provider log a warning

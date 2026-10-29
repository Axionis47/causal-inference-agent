# syntax=docker/dockerfile:1
# One deployable application: React build + FastAPI + desk + all three lanes.
FROM node:20-bookworm-slim AS web-build
WORKDIR /build/web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/index.html web/tsconfig.json web/vite.config.ts ./
COPY web/src ./src
RUN npm run build

FROM ghcr.io/astral-sh/uv:0.12.0 AS uv
FROM python:3.12-slim-bookworm AS python-base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OPENSSL_armcap=0 \
    PATH="/opt/venv/bin:$PATH"
# Use baseline ARM instructions across Docker VM CPU implementations. This
# avoids SIGILL in cryptography's OpenSSL capability probe; x86 ignores it.

FROM python-base AS python-build
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential pkg-config libffi-dev \
    && rm -rf /var/lib/apt/lists/*
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_PROJECT_ENVIRONMENT=/opt/venv UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /build
COPY pyproject.toml uv.lock ./
# The application source stays at /app because current file addresses use that root.
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-install-project

FROM python-base AS app
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 causal \
    && useradd --uid 10001 --gid causal --create-home causal
COPY --from=python-build /opt/venv /opt/venv
RUN python -c "from langchain_google_vertexai import ChatVertexAI"
WORKDIR /app
COPY causal_agent ./causal_agent
COPY alembic.ini ./
COPY migrations ./migrations
COPY --from=web-build /build/web/dist ./web/dist
COPY docker/healthcheck.py /usr/local/bin/causal-healthcheck.py
COPY docker/sandbox_probe.py /usr/local/bin/causal-sandbox-probe.py
# Empty, owned storage is copied into new named volumes by Docker. No local
# datasets, transcripts, credentials or analysis outputs are baked into the image.
RUN mkdir -p data/raw data/profiles data/memory data/web .artifacts /home/causal/.cache \
    && printf '{}\n' > data/datasets.yaml \
    && chown -R causal:causal data .artifacts /home/causal/.cache
ENV HOME=/home/causal \
    PYTHONPATH=/app \
    MPLCONFIGDIR=/home/causal/.cache/matplotlib \
    LANGSMITH_TRACING=false
USER causal
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD ["python", "/usr/local/bin/causal-healthcheck.py"]
CMD ["python", "-m", "causal_agent.server", "--host", "0.0.0.0", "--port", "8000"]

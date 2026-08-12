FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir uv==0.12.0
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-cache
COPY backend ./backend
COPY migrations ./migrations
COPY alembic.ini ./
RUN useradd --uid 10001 --create-home app && mkdir /storage && chown app:app /storage
USER app
ENV STORAGE_DIR=/storage
EXPOSE 8008
CMD [".venv/bin/uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8008"]

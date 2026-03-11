# ── Build stage ─────────────────────────────────────────────────────────────
FROM python:3.13-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml ./

RUN uv venv /app/.venv && \
    uv sync --no-dev --no-install-project

COPY app/ ./app/
COPY templates/ ./templates/
COPY config/models.example.yaml ./config/models.example.yaml

# ── Runtime stage ───────────────────────────────────────────────────────────
FROM python:3.13-slim

# Minimal runtime deps — NO Slurm packages.
# Slurm binaries + libs are bind-mounted from the host.
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/app ./app
COPY --from=builder /app/templates ./templates
COPY --from=builder /app/config ./config

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

RUN mkdir -p /app/logs /app/data

# Always listen on 9000 inside the container.
# Users map this to any host port via docker-compose.
EXPOSE 9000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:9000/health || exit 1

# Internal port is always 9000 — the ROUTER_PORT env var controls
# what gets advertised to Slurm jobs and must match the HOST-side port mapping.
ENTRYPOINT ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9000"]

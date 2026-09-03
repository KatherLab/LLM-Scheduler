# ── Build stage ─────────────────────────────────────────────────────────────
FROM python:3.13-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock* ./

RUN uv venv /app/.venv && \
    uv sync --no-dev --no-install-project

COPY app/ ./app/
COPY templates/ ./templates/
COPY config/models.example.yaml config/cluster.example.yaml ./config/

# ── Runtime stage ───────────────────────────────────────────────────────────
FROM python:3.13-slim

# With CLUSTER_BACKEND=slurm_rest the image needs NO Slurm packages, no munge,
# no bind-mounted binaries and no host-matching UID — the router talks to
# slurmrestd over HTTP. That is the whole point of the REST backend.
#
#   curl           container HEALTHCHECK
#   openssh-client only for SLURM_TOKEN_MODE=command with the SSH recipe
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl openssh-client && \
    rm -rf /var/lib/apt/lists/*

# Run unprivileged. Nothing here needs root, and the SSH key this may mount is
# a credential that can mint Slurm tokens.
RUN groupadd --gid 1000 scheduler && \
    useradd --uid 1000 --gid 1000 --create-home --shell /usr/sbin/nologin scheduler

WORKDIR /app

COPY --from=builder --chown=scheduler:scheduler /app/.venv /app/.venv
COPY --from=builder --chown=scheduler:scheduler /app/app ./app
COPY --from=builder --chown=scheduler:scheduler /app/templates ./templates
COPY --from=builder --chown=scheduler:scheduler /app/config ./config

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    CLUSTER_BACKEND=slurm_rest \
    DATABASE_URL=sqlite:////app/data/router.db \
    VLLM_LOG_DIR=/app/logs \
    SBATCH_TEMPLATE_PATH=/app/templates/vllm_job.sh

RUN mkdir -p /app/logs /app/data && chown -R scheduler:scheduler /app/logs /app/data

USER scheduler

# Always listen on 9000 inside the container.
# Users map this to any host port via docker-compose.
EXPOSE 9000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:9000/health || exit 1

# Internal port is always 9000 — the ROUTER_PORT env var controls
# what gets advertised to Slurm jobs and must match the HOST-side port mapping.
ENTRYPOINT ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9000"]

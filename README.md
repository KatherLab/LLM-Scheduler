# vLLM Swapper + Model Router (Slurm, single HGX node)

This repo implements a **thin, fast OpenAI-compatible proxy** ("Model Router") plus a **Slurm-backed swapper/scheduler**
to start/stop/extend vLLM instances on a single HGX server with limited GPUs.

## What you get
- Stable endpoints (no client-visible ports):
  - `POST /v1/chat/completions`
  - `POST /v1/messages`
  - `GET /v1/models`
- Admin endpoints for scheduling and lifecycle:
  - `POST /admin/leases` (start now or schedule later)
  - `POST /admin/leases/{id}/extend`
  - `DELETE /admin/leases/{id}` (unload)
  - `GET /admin/leases`
  - `GET /admin/endpoints`
- Slurm integration:
  - vLLM runs as a Slurm job requesting `--gres=gpu:N`
  - Time limits (`--time`) and begin time (`--begin`) supported
  - Cancel / extend via `scancel` and `scontrol update`
- Endpoint registry + readiness checks:
  - vLLM jobs register back to the router with jobid/model/port
  - router health-checks `/health` and routes only to READY endpoints

## Assumptions
- One HGX server (Slurm node) that can run Slurm jobs locally.
- vLLM exposes OpenAI-compatible endpoints and `/health`.
- Router runs on the same node or reachable over the cluster network.

## Quick start (dev)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# configure
cp config/example.env .env
# edit .env for your node, Slurm partitions, and paths

# run
uvicorn app.main:app --host 0.0.0.0 --port 9000
```

## Model catalog
Edit `config/models.yaml` to define runnable models and defaults.

## How LiteLLM fits
Keep LiteLLM public-facing for auth/key management. Configure one backend for all "dynamic" models:
- `api_base` points to this router, and users set `model` to one of the catalog names.
This router will route to the currently-running vLLM instance for that model.

## Production notes
- Run with systemd (sample unit in `deploy/systemd/vllm-router.service`)
- Use Postgres if desired; default is SQLite for simplicity.
- Put the router behind your internal network / reverse proxy (e.g., nginx) if needed.

## Web UI
Open `http://<router-host>:9000/` for the timeline UI.

## Default model
Set `DEFAULT_MODEL` (and optionally `DEFAULT_MODEL_GPUS`, `DEFAULT_MODEL_TP`) in `.env` to keep a default model running when no other model is READY.
This does **not** implement request fallback chains; it only manages an idle default.

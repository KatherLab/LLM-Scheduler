#!/usr/bin/env bash
#SBATCH --output=./logs/%x-%j.out
#SBATCH --error=./logs/%x-%j.err

set -euo pipefail

# Env injected by router:
# MODEL_PATH, SERVED_MODEL_NAME, TP_SIZE, API_KEY, GPU_MEM_UTIL, EXTRA_ARGS, TOOL_ARGS, REASONING_PARSER,
# ROUTER_REGISTER_URL, VENV_ACTIVATE

mkdir -p ./logs

# --- 0. ACTIVATE VENV (optional) ---
if [[ -n "${VENV_ACTIVATE:-}" ]]; then
  echo "Activating venv: ${VENV_ACTIVATE}"
  # shellcheck disable=SC1090
  source "${VENV_ACTIVATE}"
fi

# --- 1. FIND FREE PORT ---
PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

echo "Starting vLLM job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}"
echo "Assigned Port: ${PORT}"

# --- 2. START vLLM ---
vllm serve "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --mm-encoder-tp-mode data \
  --mm-processor-cache-type shm \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --api-key "${API_KEY}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL:-0.95}" \
  ${REASONING_PARSER:+--reasoning-parser "${REASONING_PARSER}"} \
  ${TOOL_ARGS:-} \
  ${EXTRA_ARGS:-} &

VLLM_PID=$!

# --- 3. WAIT FOR HEALTH & REGISTER ---
MAX_RETRIES=300
for i in $(seq 1 $MAX_RETRIES); do
  if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    echo "vLLM is healthy, registering endpoint..."
    curl -fsS -X POST "${ROUTER_REGISTER_URL}" \
      -H "Content-Type: application/json" \
      -d "{\"slurm_job_id\":\"${SLURM_JOB_ID}\",\"model\":\"${SERVED_MODEL_NAME}\",\"host\":\"${SLURMD_NODENAME}\",\"port\":${PORT}}" >/dev/null || true
    break
  fi
  sleep 5
done

wait "${VLLM_PID}"

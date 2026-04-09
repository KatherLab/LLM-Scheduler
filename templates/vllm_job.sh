#!/usr/bin/env bash
set -euo pipefail

echo "Starting vLLM Slurm job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}"

# Activate venv
if [[ -n "${VENV_ACTIVATE:-}" ]]; then
  source "${VENV_ACTIVATE}"
fi

# Get a free port
PORT="$(python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')"
echo "Assigned Port: ${PORT}"

# Register with router (retry for up to 60s in case router is restarting)
REGISTERED=0
for i in $(seq 1 12); do
  if curl -fsS -X POST "${ROUTER_REGISTER_URL}" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer ${API_KEY}" \
    -d "{\"slurm_job_id\":\"${SLURM_JOB_ID}\",\"model\":\"${SERVED_MODEL_NAME}\",\"host\":\"${SLURMD_NODENAME}\",\"port\":${PORT}}"; then
    REGISTERED=1
    echo "Registered with router on attempt ${i}"
    break
  fi
  echo "Registration attempt ${i} failed, retrying in 5s..."
  sleep 5
done

if [ "${REGISTERED}" -eq 0 ]; then
  echo "Warning: failed to register endpoint after 12 attempts — continuing anyway"
fi

# Build argv safely
CMD=(
  vllm serve "${MODEL_PATH}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --tensor-parallel-size "${TP_SIZE}"
  --host 0.0.0.0
  --port "${PORT}"
  --api-key "${API_KEY}"
  --gpu-memory-utilization "${GPU_MEM_UTIL:-0.95}"
)

if [[ -n "${REASONING_PARSER:-}" ]]; then
  CMD+=(--reasoning-parser "${REASONING_PARSER}")
fi

# Parse string-style extra args/tool args with shell quoting preserved.
# These values must come from trusted config only.
if [[ -n "${TOOL_ARGS:-}" ]]; then
  eval "CMD+=( ${TOOL_ARGS} )"
fi

if [[ -n "${EXTRA_ARGS:-}" ]]; then
  eval "CMD+=( ${EXTRA_ARGS} )"
fi

echo "Launching command:"
printf '  %q' "${CMD[@]}"
echo

# Start vLLM — job lifetime == vLLM lifetime
exec "${CMD[@]}"
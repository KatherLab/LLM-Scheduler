#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/var/log/vllm/%x-%j.out
#SBATCH --error=/var/log/vllm/%x-%j.err

set -euo pipefail

# Env injected by router:
# MODEL_PATH, SERVED_MODEL_NAME, TP_SIZE, PORT, API_KEY, GPU_MEM_UTIL, EXTRA_ARGS, TOOL_ARGS, REASONING_PARSER, ROUTER_REGISTER_URL

mkdir -p /var/log/vllm

echo "Starting vLLM job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "Model: ${SERVED_MODEL_NAME} -> ${MODEL_PATH}"
echo "Port: ${PORT}"

# Start vLLM in background, then register endpoint once /health responds
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

# Wait for health
for i in $(seq 1 300); do
  if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    echo "vLLM is healthy, registering endpoint..."
    curl -fsS -X POST "${ROUTER_REGISTER_URL}" \
      -H "Content-Type: application/json" \
      -d "{\"slurm_job_id\":\"${SLURM_JOB_ID}\",\"model\":\"${SERVED_MODEL_NAME}\",\"host\":\"${SLURMD_NODENAME}\",\"port\":${PORT}}" \
      >/dev/null || true
    break
  fi
  sleep 1
done

wait "${VLLM_PID}"

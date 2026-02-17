#!/usr/bin/env bash
set -euo pipefail

# Env injected by router:
# MODEL_PATH, SERVED_MODEL_NAME, TP_SIZE, API_KEY, GPU_MEM_UTIL, EXTRA_ARGS, TOOL_ARGS, REASONING_PARSER,
# ROUTER_REGISTER_URL, VENV_ACTIVATE
#
# Optional env for retry behavior:
# VLLM_HEALTH_TIMEOUT_SECONDS (default 180)
# VLLM_MAX_RETRIES (default 2)
# VLLM_RETRY_DELAY_SECONDS (default 60)

HEALTH_TIMEOUT_SECONDS="${VLLM_HEALTH_TIMEOUT_SECONDS:-180}"
MAX_RETRIES="${VLLM_MAX_RETRIES:-2}"
RETRY_DELAY_SECONDS="${VLLM_RETRY_DELAY_SECONDS:-60}"

echo "Starting vLLM Slurm job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}"
echo "Health timeout: ${HEALTH_TIMEOUT_SECONDS}s | max retries: ${MAX_RETRIES} | retry delay: ${RETRY_DELAY_SECONDS}s"

# --- 0) ACTIVATE VENV (optional) ---
if [[ -n "${VENV_ACTIVATE:-}" ]]; then
  echo "Activating venv: ${VENV_ACTIVATE}"
  # shellcheck disable=SC1090
  source "${VENV_ACTIVATE}"
fi

# --- helper: get free port ---
alloc_port() {
  python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()'
}

attempt=1
while [[ "${attempt}" -le "${MAX_RETRIES}" ]]; do
  echo "---- vLLM attempt ${attempt}/${MAX_RETRIES} ----"

  PORT="$(alloc_port)"
  echo "Assigned Port: ${PORT}"

  # --- 1) START vLLM ---
  set +e
  vllm serve "${MODEL_PATH}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --api-key "${API_KEY}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL:-0.95}" \
    ${REASONING_PARSER:+--reasoning-parser "${REASONING_PARSER}"} \
    ${TOOL_ARGS:-} \
    ${EXTRA_ARGS:-} &
  VLLM_PID=$!
  set -e

  echo "vLLM PID: ${VLLM_PID}"

  # --- 2) WAIT FOR HEALTH OR EARLY EXIT ---
  start_ts="$(date +%s)"
  registered="0"

  while true; do
    # If process already died → fail attempt fast
    if ! kill -0 "${VLLM_PID}" >/dev/null 2>&1; then
      echo "vLLM process exited before becoming healthy."
      wait "${VLLM_PID}" || true
      break
    fi

    if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
      echo "vLLM is healthy."

      if [[ "${registered}" == "0" ]]; then
        echo "Registering endpoint..."
        curl -fsS -X POST "${ROUTER_REGISTER_URL}" \
          -H "Content-Type: application/json" \
          -d "{\"slurm_job_id\":\"${SLURM_JOB_ID}\",\"model\":\"${SERVED_MODEL_NAME}\",\"host\":\"${SLURMD_NODENAME}\",\"port\":${PORT}}" \
          >/dev/null || true
        registered="1"
      fi

      # Now: job lifetime == vLLM lifetime. If vLLM exits, job exits with same code.
      wait "${VLLM_PID}"
      exit_code=$?
      echo "vLLM exited with code ${exit_code}"
      exit "${exit_code}"
    fi

    now_ts="$(date +%s)"
    elapsed="$((now_ts - start_ts))"
    if [[ "${elapsed}" -ge "${HEALTH_TIMEOUT_SECONDS}" ]]; then
      echo "Timed out waiting for /health after ${HEALTH_TIMEOUT_SECONDS}s. Killing vLLM and failing attempt."
      kill "${VLLM_PID}" >/dev/null 2>&1 || true
      wait "${VLLM_PID}" >/dev/null 2>&1 || true
      break
    fi

    sleep 2
  done

  # If we got here, attempt failed (either early exit or timeout)
  if [[ "${attempt}" -lt "${MAX_RETRIES}" ]]; then
    echo "Attempt ${attempt} failed. Retrying after ${RETRY_DELAY_SECONDS}s..."
    sleep "${RETRY_DELAY_SECONDS}"
  fi

  attempt="$((attempt + 1))"
done

echo "All attempts failed. Exiting with failure."
exit 1

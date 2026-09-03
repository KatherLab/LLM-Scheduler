#!/usr/bin/env bash
set -euo pipefail

echo "Starting vLLM Slurm job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}"
echo "Runtime: ${RUNTIME_KIND:-venv} (${RUNTIME_NAME:-default})"

# ── Runtime selection ────────────────────────────────────────────────────────
# RUNTIME_KIND comes from the GPU class via cluster.yaml, so an aarch64 node
# gets an aarch64 image without the model entry knowing anything about it.
# The venv path is unchanged, so existing deployments keep working.

RUNTIME_KIND="${RUNTIME_KIND:-venv}"

# Prefix that turns the vLLM argv into a container invocation (empty for venv).
LAUNCH_PREFIX=()

case "${RUNTIME_KIND}" in
  apptainer)
    if [[ -z "${APPTAINER_IMAGE:-}" ]]; then
      echo "ERROR: RUNTIME_KIND=apptainer but APPTAINER_IMAGE is empty" >&2
      exit 2
    fi
    if [[ ! -f "${APPTAINER_IMAGE}" ]]; then
      echo "ERROR: Apptainer image not found: ${APPTAINER_IMAGE}" >&2
      echo "Note: Apptainer cannot cross-build — an aarch64 .sif must be built on an aarch64 node." >&2
      exit 2
    fi

    LAUNCH_PREFIX=(apptainer exec)

    # Note: `[[ ... ]] && cmd` would abort the script under `set -e` whenever
    # the test is false, so these are written as full if-blocks.
    if [[ "${APPTAINER_NV:-1}" == "1" ]]; then
      LAUNCH_PREFIX+=(--nv)
    fi

    # Comma-separated binds, each passed as its own --bind.
    if [[ -n "${APPTAINER_BINDS:-}" ]]; then
      IFS=',' read -ra _binds <<< "${APPTAINER_BINDS}"
      for b in "${_binds[@]}"; do
        if [[ -n "${b}" ]]; then
          LAUNCH_PREFIX+=(--bind "${b}")
        fi
      done
    fi

    LAUNCH_PREFIX+=("${APPTAINER_IMAGE}")
    echo "Apptainer image: ${APPTAINER_IMAGE}"
    ;;

  venv)
    if [[ -n "${VENV_ACTIVATE:-}" ]]; then
      # shellcheck disable=SC1090
      source "${VENV_ACTIVATE}"
    fi
    ;;

  *)
    echo "ERROR: unknown RUNTIME_KIND '${RUNTIME_KIND}' (expected apptainer or venv)" >&2
    exit 2
    ;;
esac

# Capture vLLM version from whichever runtime we ended up in.
# `${arr[@]+"${arr[@]}"}` expands to nothing when the array is empty, which a
# bare `"${arr[@]}"` does not under `set -u` on bash < 4.4.
VLLM_VERSION="$(${LAUNCH_PREFIX[@]+"${LAUNCH_PREFIX[@]}"} vllm --version 2>&1 | head -1 | tr -d '"' || true)"
echo "vLLM version: ${VLLM_VERSION}"

# ── Co-located group ─────────────────────────────────────────────────────────
# When COLOCATED_MODELS is set, this job is a GPU host: Slurm gave us one GPU
# and we run several vLLM servers on it, each with its own port and its own
# pre-allocated memory budget. Budgets were validated to fit before submission.
#
# This exists because the cluster gives us no way to subdivide a GPU — MPS is
# disabled, MIG is static and absent on GB10, and gres/shard would need a
# slurm.conf change.

if [[ -n "${COLOCATED_MODELS:-}" ]]; then
  echo "Co-location: launching ${COLOCATED_COUNT:-?} models on one GPU"

  free_port() {
    python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()'
  }

  register() {
    # $1=model  $2=port
    for i in $(seq 1 12); do
      if curl -fsS -X POST "${ROUTER_REGISTER_URL}" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer ${API_KEY}" \
        -d "{\"slurm_job_id\":\"${SLURM_JOB_ID}\",\"model\":\"$1\",\"host\":\"${SLURMD_NODENAME}\",\"port\":$2,\"vllm_version\":\"${VLLM_VERSION}\"}"; then
        return 0
      fi
      sleep 5
    done
    echo "Warning: failed to register $1 after 12 attempts"
    return 0
  }

  # Wait until an instance is actually serving before starting the next one.
  # vLLM sizes its KV cache by profiling *free* GPU memory at init, so two
  # instances initialising at once both measure the same free memory and both
  # size for it — they mis-size or OOM. Startup must therefore be serialised,
  # even though the total was validated to fit before submission.
  wait_until_ready() {
    local name="$1" port="$2" deadline=$(( SECONDS + ${COLOCATED_STARTUP_TIMEOUT:-900} ))
    while [[ ${SECONDS} -lt ${deadline} ]]; do
      if curl -fsS -m 3 "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
        echo "[${name}] ready on port ${port} after $(( SECONDS - deadline + ${COLOCATED_STARTUP_TIMEOUT:-900} ))s"
        return 0
      fi
      # A dead supervisor will never become ready; stop waiting for it.
      if ! kill -0 "$3" 2>/dev/null; then
        echo "[${name}] supervisor exited before becoming ready"
        return 1
      fi
      sleep 5
    done
    echo "[${name}] did not become ready within ${COLOCATED_STARTUP_TIMEOUT:-900}s"
    return 1
  }

  # One supervisor per co-tenant. A crashed model is restarted on its own port
  # and re-registered, so one bad model does not take down its neighbours —
  # they share a job, but not a failure.
  #
  # The port is assigned once, by the parent, rather than per attempt: the
  # parent needs it to health-check startup, and a restart is free to reuse it.
  supervise() {
    local name="$1" path="$2" util="$3" tp="$4" extra="$5" tools="$6" parser="$7" port="$8"
    local attempt=0
    while [[ ${attempt} -lt ${COLOCATED_MAX_RESTARTS:-3} ]]; do
      echo "[${name}] starting on port ${port} (gpu-mem-util ${util})"
      register "${name}" "${port}"

      local cmd=(
        ${LAUNCH_PREFIX[@]+"${LAUNCH_PREFIX[@]}"}
        vllm serve "${path}"
        --served-model-name "${name}"
        --tensor-parallel-size "${tp}"
        --host 0.0.0.0 --port "${port}"
        --api-key "${API_KEY}"
        --gpu-memory-utilization "${util}"
      )
      if [[ -n "${parser}" ]]; then cmd+=(--reasoning-parser "${parser}"); fi
      if [[ -n "${tools}" ]]; then eval "cmd+=( ${tools} )"; fi
      if [[ -n "${extra}" ]]; then eval "cmd+=( ${extra} )"; fi

      "${cmd[@]}" && break
      attempt=$((attempt + 1))
      echo "[${name}] exited (attempt ${attempt}); restarting in 10s"
      sleep 10
    done
    echo "[${name}] supervisor giving up after ${attempt} restarts"
  }

  # Start them one at a time, each fully up before the next begins.
  # A model that never becomes ready is left behind rather than blocking the
  # rest: partial service beats none, and its memory simply stays free.
  while IFS=$'\t' read -r name path util tp extra tools parser; do
    [[ -z "${name}" ]] && continue
    port="$(free_port)"
    supervise "${name}" "${path}" "${util}" "${tp}" "${extra}" "${tools}" "${parser}" "${port}" &
    pid=$!
    if ! wait_until_ready "${name}" "${port}" "${pid}"; then
      echo "[${name}] continuing without it"
    fi
  done < <(python3 -c '
import json, os, sys
for m in json.loads(os.environ["COLOCATED_MODELS"]):
    sys.stdout.write("\t".join([
        m["model"], m["model_path"], str(m["gpu_memory_utilization"]),
        str(m.get("tensor_parallel_size", 1)), m.get("extra_args", ""),
        m.get("tool_args", ""), m.get("reasoning_parser", ""),
    ]) + "\n")
')

  # The allocation lives as long as any co-tenant does.
  wait
  echo "All co-located models have exited"
  exit 0
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
    -d "{\"slurm_job_id\":\"${SLURM_JOB_ID}\",\"model\":\"${SERVED_MODEL_NAME}\",\"host\":\"${SLURMD_NODENAME}\",\"port\":${PORT},\"vllm_version\":\"${VLLM_VERSION}\"}"; then
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
  ${LAUNCH_PREFIX[@]+"${LAUNCH_PREFIX[@]}"}
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

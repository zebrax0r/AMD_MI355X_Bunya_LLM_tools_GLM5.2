#!/usr/bin/env bash
#
# serve-glm52.sh — one-click GLM-5.2 serving on Bunya's AMD MI355X nodes
# (bun159/160/161, 8x gfx950) via SGLang inside an Apptainer container.
#
# Usage:
#   ./serve-glm52.sh [serve]     start the server (default; runs until killed)
#   ./serve-glm52.sh serve --detach
#                                start the server, wait until healthy, then
#                                return the shell (server keeps running in the
#                                background for the life of the SLURM job) —
#                                use this to run opencode on the GPU node itself
#   ./serve-glm52.sh pull        build the .sif from the container image (once)
#   ./serve-glm52.sh download    prefetch model weights only (no GPU needed)
#   ./serve-glm52.sh stop        stop a running server
#   ./serve-glm52.sh status      show server state + health endpoint
#
# Apptainer lives ONLY on Bunya compute nodes, never the login nodes — so every
# mode except a bare 'stop'/'status' must run inside a salloc/sbatch allocation.
#
# Configuration comes from glm52.env next to this script (or $GLM52_ENV),
# see glm52-env.example. Environment variables you export beforehand win.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODE="serve"
DETACH=0
for arg in "$@"; do
    case "$arg" in
        --detach|-d) DETACH=1 ;;
        *)           MODE="$arg" ;;
    esac
done

log()  { printf '\033[1;34m[glm52]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[glm52 WARN]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[glm52 ERROR]\033[0m %s\n' "$*" >&2; exit 1; }

# ── Load config ─────────────────────────────────────────────────────────────

ENV_FILE="${GLM52_ENV:-$SCRIPT_DIR/glm52.env}"
if [[ -f "$ENV_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    log "Loaded config from $ENV_FILE"
else
    warn "No config file at $ENV_FILE (copy glm52-env.example to glm52.env); using environment only."
fi

MODEL_ID="${MODEL_ID:-amd/GLM-5.2-MXFP4}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-glm-5.2}"
SGLANG_IMAGE="${SGLANG_IMAGE:-docker://lmsysorg/sglang-rocm:v0.5.13.post1-rocm720-mi35x-20260618}"
PORT="${PORT:-30000}"
TP_SIZE="${TP_SIZE:-8}"
DP_SIZE="${DP_SIZE:-1}"
CONTEXT_LEN="${CONTEXT_LEN:-262144}"
MEM_FRACTION="${MEM_FRACTION:-0.85}"
ENABLE_AITER_ALLREDUCE_FUSION="${ENABLE_AITER_ALLREDUCE_FUSION:-1}"
# The sglang-rocm image sets SGLANG_SET_CPU_AFFINITY=1, but SGLang then pins
# workers to CPUs derived from the FULL node topology — which fail under a SLURM
# cgroup that only owns a subset of cores ("CPU number N is not eligible").
# Default it OFF; set to 1 only if you allocate the whole node's CPUs.
SET_CPU_AFFINITY="${SET_CPU_AFFINITY:-0}"
READY_TIMEOUT="${READY_TIMEOUT:-7200}"
EXTRA_SGLANG_ARGS="${EXTRA_SGLANG_ARGS:-}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-}"
SIF_PATH="${SIF_PATH:-}"

# Runtime state (PID + log) lives under MODEL_CACHE_DIR so it survives detach
# and is reachable by 'stop'/'status' from any shell in the allocation.
if [[ -n "$MODEL_CACHE_DIR" ]]; then
    PID_FILE="${PID_FILE:-$MODEL_CACHE_DIR/glm52-server.pid}"
    LOG_FILE="${LOG_FILE:-$MODEL_CACHE_DIR/glm52-server.log}"
else
    PID_FILE="${PID_FILE:-$SCRIPT_DIR/glm52-server.pid}"
    LOG_FILE="${LOG_FILE:-$SCRIPT_DIR/glm52-server.log}"
fi

server_running() { [[ -f "$PID_FILE" ]] && kill -0 "$(<"$PID_FILE")" 2>/dev/null; }

stop_server() {
    if server_running; then
        local pid; pid="$(<"$PID_FILE")"
        log "Stopping server (pid $pid) ..."
        kill "$pid" 2>/dev/null || true
        # Give SGLang's worker processes a moment, then make sure they're gone.
        for _ in $(seq 1 15); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
        kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
    else
        warn "No running server recorded in $PID_FILE."
    fi
    # Belt-and-braces: this node is a single-user 8-GPU grab, so clean up strays.
    pkill -f 'sglang.launch_server' 2>/dev/null || true
    rm -f "$PID_FILE"
}

# ── Simple modes first ──────────────────────────────────────────────────────

case "$MODE" in
    stop)
        stop_server
        log "Stopped."
        exit 0
        ;;
    status)
        if server_running; then
            log "Server process alive (pid $(<"$PID_FILE"))."
        else
            log "No server process recorded (pidfile $PID_FILE)."
        fi
        if curl -fsS -m 5 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
            log "Health check on port $PORT: OK"
        else
            warn "http://127.0.0.1:${PORT}/health not responding (not started, or still loading)."
            [[ -f "$LOG_FILE" ]] && log "Follow progress with: tail -f $LOG_FILE"
        fi
        exit 0
        ;;
    serve|pull|download) ;;
    *)
        die "Unknown mode '$MODE'. Use: serve | pull | download | stop | status"
        ;;
esac

# ── Preflight (shared by pull/download/serve) ───────────────────────────────

command -v apptainer >/dev/null 2>&1 \
    || die "apptainer not found — run this INSIDE a compute-node allocation (salloc/sbatch). Apptainer is not installed on Bunya login nodes."
command -v curl >/dev/null 2>&1 || die "curl not found on PATH."

[[ -n "$MODEL_CACHE_DIR" ]] \
    || die "MODEL_CACHE_DIR is not set. Point it at scratch (e.g. /scratch/user/\$USER/glm52/hf-cache). See glm52-env.example."
mkdir -p "$MODEL_CACHE_DIR" 2>/dev/null || true
[[ -d "$MODEL_CACHE_DIR" && -w "$MODEL_CACHE_DIR" ]] \
    || die "MODEL_CACHE_DIR '$MODEL_CACHE_DIR' does not exist or is not writable."

# Default the .sif next to the HF cache if not set explicitly.
SIF_PATH="${SIF_PATH:-$MODEL_CACHE_DIR/glm52-mi355x.sif}"
# SIF_PATH must name a .sif FILE, not a directory. If it points at a directory
# (or ends with '/'), treat it as a folder and drop the default filename in —
# 'apptainer pull' otherwise refuses ("Image file already exists").
if [[ "$SIF_PATH" == */ || -d "$SIF_PATH" ]]; then
    SIF_PATH="${SIF_PATH%/}/glm52-mi355x.sif"
    log "SIF_PATH was a directory — using $SIF_PATH"
fi

# Keep Apptainer's cache and scratch off /home (which has a tight quota) — point
# them at scratch. Set both APPTAINER_* and the SINGULARITY_* aliases.
: "${APPTAINER_CACHEDIR:=$MODEL_CACHE_DIR/../apptainer-cache}"
: "${APPTAINER_TMPDIR:=$MODEL_CACHE_DIR/../apptainer-tmp}"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" 2>/dev/null || true
export APPTAINER_CACHEDIR APPTAINER_TMPDIR
export SINGULARITY_CACHEDIR="$APPTAINER_CACHEDIR"
export SINGULARITY_TMPDIR="$APPTAINER_TMPDIR"

# ── Pull mode: build the .sif from the container image ──────────────────────

if [[ "$MODE" == "pull" ]]; then
    if [[ -f "$SIF_PATH" ]]; then
        log "Image already present: $SIF_PATH (delete it to re-pull)."
        exit 0
    fi
    log "Pulling $SGLANG_IMAGE -> $SIF_PATH (multi-GB; one-time) ..."
    apptainer pull "$SIF_PATH" "$SGLANG_IMAGE" \
        || die "apptainer pull failed. Does this node have outbound internet? Check APPTAINER_CACHEDIR space ($APPTAINER_CACHEDIR)."
    log "Image ready: $SIF_PATH"
    exit 0
fi

# For download/serve we need the .sif. Auto-build it if missing.
if [[ ! -f "$SIF_PATH" ]]; then
    log "No .sif at $SIF_PATH yet — pulling it now (one-time)."
    apptainer pull "$SIF_PATH" "$SGLANG_IMAGE" \
        || die "apptainer pull failed. Run './serve-glm52.sh pull' explicitly to debug."
    log "Image ready: $SIF_PATH"
fi

# Resolve HF token: env var, then token file.
if [[ -z "${HF_TOKEN:-}" && -n "${HF_TOKEN_FILE:-}" ]]; then
    [[ -r "$HF_TOKEN_FILE" ]] || die "HF_TOKEN_FILE '$HF_TOKEN_FILE' is not readable."
    HF_TOKEN="$(<"$HF_TOKEN_FILE")"
fi

# Weights already cached? (hub layout: models--org--name)
weights_dir="$MODEL_CACHE_DIR/hub/models--${MODEL_ID//\//--}"
weights_cached=0
[[ -d "$weights_dir/snapshots" ]] && weights_cached=1

if [[ -z "${HF_TOKEN:-}" && "$weights_cached" -eq 0 ]]; then
    die "No HF_TOKEN / HF_TOKEN_FILE set and weights for $MODEL_ID are not cached yet in $MODEL_CACHE_DIR."
fi

if [[ "$weights_cached" -eq 0 ]]; then
    free_gb="$(df -Pk "$MODEL_CACHE_DIR" | awk 'NR==2 {print int($4/1024/1024)}')"
    if [[ "${free_gb:-0}" -lt 500 ]]; then
        warn "Only ${free_gb} GB free in $MODEL_CACHE_DIR; $MODEL_ID (MXFP4) needs ~380 GB. Download may fail."
    fi
    log "Weights not cached yet — first start will download ~380 GB. Consider './serve-glm52.sh download' first."
else
    log "Found cached weights for $MODEL_ID."
fi

# ── Download mode (no GPU required) ─────────────────────────────────────────

if [[ "$MODE" == "download" ]]; then
    log "Prefetching $MODEL_ID into $MODEL_CACHE_DIR (no GPU required) ..."
    apptainer exec \
        --bind "$MODEL_CACHE_DIR":"$MODEL_CACHE_DIR" \
        --env HF_HOME="$MODEL_CACHE_DIR" \
        --env HF_TOKEN="${HF_TOKEN:-}" \
        --env HF_HUB_ENABLE_HF_TRANSFER=1 \
        "$SIF_PATH" \
        bash -c "hf download '$MODEL_ID' || huggingface-cli download '$MODEL_ID'" \
        || die "Weights download failed. Re-run to resume."
    log "Download complete."
    exit 0
fi

# ── Serve-mode preflight (GPU node checks) ──────────────────────────────────

[[ -e /dev/kfd ]] || die "/dev/kfd not found — is this a ROCm GPU node? (Apptainer + --rocm needs it.)"
[[ -e /dev/dri ]] || die "/dev/dri not found — is this a ROCm GPU node?"

if command -v rocminfo >/dev/null 2>&1; then
    gfx="$(rocminfo 2>/dev/null | grep -om1 'gfx[0-9a-f]*' || true)"
    case "$gfx" in
        gfx950) log "Detected gfx950 (MI350X/MI355X) — matches the configured MXFP4 image/model." ;;
        gfx942) warn "Detected gfx942 (MI300X/MI325X). This repo is tuned for MI355X/MXFP4. For gfx942 use the FP8 setup:
        MODEL_ID=zai-org/GLM-5.2-FP8  SGLANG_IMAGE=docker://lmsysorg/sglang-rocm:v0.5.13.post1-rocm700-mi30x-20260616  (TP8, no --dp)" ;;
        "")     warn "Could not detect GPU arch from rocminfo." ;;
        *)      warn "Detected $gfx — this recipe is tuned for gfx950 (MI355X)." ;;
    esac
fi

# In SGLang, --tp IS the total GPU count. --dp (with --enable-dp-attention) only
# subdivides those GPUs for attention — it does NOT multiply the GPU count. So an
# 8-GPU node means TP_SIZE=8 (optionally DP_SIZE=2 or 8 for dp-attention), and
# DP_SIZE must divide TP_SIZE.
GPUS_USED="$TP_SIZE"
if [[ "${DP_SIZE:-1}" -gt 1 && $(( TP_SIZE % DP_SIZE )) -ne 0 ]]; then
    die "DP_SIZE=$DP_SIZE must divide TP_SIZE=$TP_SIZE (dp-attention splits the $TP_SIZE GPUs into $DP_SIZE groups)."
fi
GPU_VIS="${ROCR_VISIBLE_DEVICES:-${HIP_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-}}}"
alloc_count=""
if [[ -n "$GPU_VIS" ]]; then
    alloc_count="$(awk -F, '{print NF}' <<<"$GPU_VIS")"
elif [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
    alloc_count="$SLURM_GPUS_ON_NODE"
fi
[[ -n "$GPU_VIS" ]] && log "Allocated GPUs: [$GPU_VIS]"
if [[ -n "$alloc_count" && "$alloc_count" -gt 0 && "$GPUS_USED" -ne "$alloc_count" ]]; then
    warn "TP_SIZE=$TP_SIZE uses $GPUS_USED GPU(s) but $alloc_count are allocated — you'd leave $((alloc_count - GPUS_USED)) idle (or over-subscribe). Set TP_SIZE=$alloc_count to use them all. (An MI355X node = 8 GPUs -> TP_SIZE=8.)"
fi

# Port free?
if (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; then
    exec 3>&- 3<&- || true
    die "Port $PORT is already in use on this node (another server running? try './serve-glm52.sh status')."
fi

if server_running; then
    die "A server is already recorded running (pid $(<"$PID_FILE")). Use './serve-glm52.sh status' or 'stop' first."
fi

# ── API key ─────────────────────────────────────────────────────────────────

API_KEY_FILE="$MODEL_CACHE_DIR/glm52-api-key"
if [[ -z "${SGLANG_API_KEY:-}" ]]; then
    if [[ -r "$API_KEY_FILE" ]]; then
        SGLANG_API_KEY="$(<"$API_KEY_FILE")"
        log "Using API key from $API_KEY_FILE"
    else
        SGLANG_API_KEY="$(openssl rand -hex 24 2>/dev/null || head -c 24 /dev/urandom | od -An -tx1 | tr -d ' \n')"
        (umask 077 && printf '%s' "$SGLANG_API_KEY" > "$API_KEY_FILE")
        log "Generated new API key and saved it to $API_KEY_FILE"
    fi
fi

# ── Launch ──────────────────────────────────────────────────────────────────

aiter_flag=()
[[ "$ENABLE_AITER_ALLREDUCE_FUSION" == "1" ]] && aiter_flag=(--enable-aiter-allreduce-fusion)

dp_flag=()
if [[ "${DP_SIZE:-1}" -gt 1 ]]; then
    dp_flag=(--dp "$DP_SIZE" --enable-dp-attention)
fi

# shellcheck disable=SC2206  # intentional word splitting of user-provided extra args
extra_args=($EXTRA_SGLANG_ARGS)

# Forward the SLURM GPU-visibility vars into the container (Apptainer inherits
# host env by default, but be explicit) so ROCm sees exactly the allocated GPUs.
gpu_env=()
[[ -n "${ROCR_VISIBLE_DEVICES:-}" ]] && gpu_env+=(--env "ROCR_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES")
[[ -n "${HIP_VISIBLE_DEVICES:-}"  ]] && gpu_env+=(--env "HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES")
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] && gpu_env+=(--env "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES")

# The MXFP4 MoE JIT-compiles FlyDSL kernels at runtime and tries to cache them
# INSIDE the image (/sgl-workspace/.../flydsl_cache) — but an Apptainer .sif is
# read-only, so that write fails ("Read-only file system"). Bind a writable
# scratch dir over that path (persists compiled kernels across runs, too).
FLYDSL_CACHE_DIR="${FLYDSL_CACHE_DIR:-$MODEL_CACHE_DIR/flydsl-cache}"
FLYDSL_CACHE_TARGET="${FLYDSL_CACHE_TARGET:-/sgl-workspace/aiter/aiter/jit/flydsl_cache}"
mkdir -p "$FLYDSL_CACHE_DIR" 2>/dev/null || true
cache_bind=(--bind "$FLYDSL_CACHE_DIR":"$FLYDSL_CACHE_TARGET")

log "Starting SGLang: $MODEL_ID  (TP=$TP_SIZE -> $GPUS_USED GPUs, DP=$DP_SIZE, ctx=$CONTEXT_LEN, port=$PORT)"
log "Image: $SIF_PATH"
log "Logs:  $LOG_FILE"

: > "$LOG_FILE"
apptainer exec --rocm \
    --bind "$MODEL_CACHE_DIR":"$MODEL_CACHE_DIR" \
    "${cache_bind[@]}" \
    --env HF_HOME="$MODEL_CACHE_DIR" \
    --env HF_TOKEN="${HF_TOKEN:-}" \
    --env HF_HUB_ENABLE_HF_TRANSFER=1 \
    --env SGLANG_SET_CPU_AFFINITY="$SET_CPU_AFFINITY" \
    ${gpu_env[@]+"${gpu_env[@]}"} \
    "$SIF_PATH" \
    python3 -m sglang.launch_server \
        --model-path "$MODEL_ID" \
        --served-model-name "$SERVED_MODEL_NAME" \
        --tp "$TP_SIZE" \
        "${dp_flag[@]}" \
        --host 0.0.0.0 --port "$PORT" \
        --tool-call-parser glm47 \
        --reasoning-parser glm45 \
        --kv-cache-dtype fp8_e4m3 \
        --mem-fraction-static "$MEM_FRACTION" \
        --context-length "$CONTEXT_LEN" \
        --api-key "$SGLANG_API_KEY" \
        --trust-remote-code \
        "${aiter_flag[@]}" \
        ${extra_args[@]+"${extra_args[@]}"} \
    >>"$LOG_FILE" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "$PID_FILE"

cleanup() {
    log "Shutting down server ..."
    kill "$SERVER_PID" 2>/dev/null || true
    pkill -f 'sglang.launch_server' 2>/dev/null || true
    rm -f "$PID_FILE"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# ── Wait for readiness ──────────────────────────────────────────────────────

log "Waiting for the server to become healthy (timeout ${READY_TIMEOUT}s; model load takes several minutes, first-run download much longer) ..."
log "Follow detailed progress in another shell with: tail -f $LOG_FILE"

start_ts="$(date +%s)"
while true; do
    if curl -fsS -m 5 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo
        tail -n 50 "$LOG_FILE" 2>&1 || true
        rm -f "$PID_FILE"
        die "Server process exited during startup. Last 50 log lines above ($LOG_FILE)."
    fi
    if (( $(date +%s) - start_ts > READY_TIMEOUT )); then
        die "Server did not become healthy within ${READY_TIMEOUT}s. Check: tail -f $LOG_FILE"
    fi
    sleep 10
done

# ── Connection banner ───────────────────────────────────────────────────────

NODE_HOST="$(hostname -s 2>/dev/null || hostname)"
JOBID="${SLURM_JOB_ID:-<jobid>}"
cat <<EOF

============================================================================
  GLM-5.2 is up and serving on Bunya's MI355X.

  Node:        $NODE_HOST     (job $JOBID)
  Endpoint:    http://$NODE_HOST:$PORT/v1   (OpenAI-compatible)
  Model name:  $SERVED_MODEL_NAME
  API key:     $API_KEY_FILE
               export SGLANG_API_KEY="\$(cat $API_KEY_FILE)"

  Smoke test (from this node):
    curl -s http://127.0.0.1:$PORT/v1/models \\
         -H "Authorization: Bearer \$SGLANG_API_KEY"

  Second shell into this job (to run opencode alongside the server):
    srun --overlap --jobid $JOBID --pty /bin/bash -l

  From your laptop (tunnel through the login node, then use localhost):
    ssh -N -L $PORT:$NODE_HOST:$PORT \${USER}@bunya1.rcc.uq.edu.au

  opencode: run ./opencode-setup.sh --host $NODE_HOST --port $PORT
            (or --host localhost when tunnelling), then pick
            '$SERVED_MODEL_NAME' via /models inside opencode.

  Stop with Ctrl-C, 'scancel $JOBID', or './serve-glm52.sh stop'.
============================================================================

EOF

if [[ "$DETACH" -eq 1 ]]; then
    # Disarm the cleanup traps: the server keeps running in the background
    # (until './serve-glm52.sh stop' or the SLURM job/allocation ends).
    trap - EXIT INT TERM
    disown "$SERVER_PID" 2>/dev/null || true
    log "Detached. You have your shell back — the server keeps running on this node."
    log "  Logs:  tail -f $LOG_FILE"
    log "  Stop:  ./serve-glm52.sh stop"
    exit 0
fi

# Stay attached: keeps the SLURM job alive and tears the server down on
# Ctrl-C / scancel via the traps above.
log "Attached. Ctrl-C (or scancel) stops the server. Streaming logs:"
tail -f "$LOG_FILE" &
TAIL_PID=$!
# Wait on the server; when it exits, stop tailing.
wait "$SERVER_PID"
kill "$TAIL_PID" 2>/dev/null || true

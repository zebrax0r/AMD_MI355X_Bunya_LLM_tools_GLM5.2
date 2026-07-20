# GLM-5.2 on Bunya's AMD MI355X — SGLang serving for opencode

Serve [GLM-5.2](https://huggingface.co/amd/GLM-5.2-MXFP4) from Bunya's 8× AMD
**MI355X** nodes (`bun159`/`bun160`/`bun161`), in an **Apptainer** container, via **SGLang**,
exposing an OpenAI-compatible endpoint you can drive with
[opencode](https://opencode.ai) — from the node itself, the login node, your
laptop over SSH, or (optionally) anyone over a public HTTPS tunnel.

Modeled on [wafer.ai's GLM-5.2-on-AMD writeup](https://www.wafer.ai/blog/glm52-amd)
and the [UQ-RCC Bunya docs](https://github.com/UQ-RCC/hpc-docs). Targets Bunya's
MI355X nodes (`bun159`/`bun160`/`bun161`): 8× MI355X each (gfx950, 288 GB HBM3),
EPYC5/Turin, Rocky Linux 9a, Apptainer 1.3.

This README is a complete walkthrough — with a Bunya account and access to an
MI355X node, you can go from clone to a working coding endpoint top to bottom.

> **Coming from the MI325X repo?** This is the Bunya sibling. The big changes:
> **Apptainer instead of podman**, the **MXFP4** model + `mi35x` image (gfx950),
> **TP8**, and Bunya's SLURM (`admin_test` partition, `a_rcc` account, `sdf`
> QoS). Apptainer only exists on compute nodes, so everything runs inside an
> allocation.

---

## What gets served

| | |
|---|---|
| Model | `amd/GLM-5.2-MXFP4` (753B MoE), served under the name `glm-5.2` |
| Engine | SGLang, pinned ROCm image `lmsysorg/sglang-rocm:v0.5.13.post1-rocm720-mi35x-20260618` |
| Parallelism | **TP8** across the node's 8 GPUs (optional dp-attention for throughput — see below) |
| Context | 262,144 tokens by default (`CONTEXT_LEN`; model supports up to 1M) |
| KV cache | FP8 (`fp8_e4m3`) |
| Tool calling / thinking | `--tool-call-parser glm47 --reasoning-parser glm45` — required for opencode's agentic loop |
| Auth | Bearer API key, auto-generated and persisted to `$MODEL_CACHE_DIR/glm52-api-key` |

**Why MXFP4 here (vs FP8 on MI325X)?** MI355X is gfx950 silicon with native
MXFP4 support, which is exactly what the wafer.ai post used. The MXFP4 quant is
~380 GB of weights — trivial across 8× 288 GB = 2.3 TB of HBM, leaving enormous
room for KV cache. The container ships its own ROCm 7.2 userspace; only the
kernel driver is shared with the host. (On gfx942 / MI300X / MI325X you'd use
the FP8 variant instead — see [Running on MI300X/MI325X](#running-on-mi300xmi325x-instead).)

---

## Bunya specifics you need to know

- **Apptainer, not podman**, and it lives **only on compute nodes** — never the
  login nodes. So `pull`, `download`, and `serve` all run inside a
  `salloc`/`sbatch` allocation.
- **The MI355X nodes are `bun159`, `bun160`, `bun161`**, and they currently sit in
  the **`admin_test`** partition (not `gpu_rocm`). Submit with
  `--partition=admin_test --account=a_rcc --qos=sdf --gres=gpu:mi355x:8` — the
  scheduler lands you on whichever is free; the serve banner prints the actual
  hostname to use for tunnels.
- **Run from `/scratch`** (`/scratch/user/$USER/...`), not `/home` (tight quota)
  and not `/QRISdata` (RDM — jobs can't be submitted from there). Check quota
  with `rquota`.
- **GPU passthrough is `--rocm`** — Apptainer binds `/dev/kfd`, `/dev/dri`, and
  the ROCm libraries automatically. No `--device`/`--group-add`/`--shm-size`
  needed; the container shares the host network and IPC (so the port and
  `/dev/shm` just work).

Confirm the live SLURM recipe any time with:

```bash
sinfo -o "%P %.10l %G %f" | grep mi355x       # partition + gres label
sacctmgr -np show assoc user=$USER format=account,qos   # your account + QoS
scontrol show node bun161 | grep -iE 'Gres|Partitions|State'
```

---

## Prerequisites

- **A Bunya account** with the `a_rcc` scheduling account and access to the
  `admin_test` partition / `sdf` QoS that reaches the MI355X nodes (these are a
  special allocation — confirm with `rcc-support@uq.edu.au` if `salloc` is
  rejected).
- **Scratch space** with **~450 GB free** for the MXFP4 weights (~380 GB) plus
  the container `.sif` (~30 GB). Use `/scratch/user/$USER` — check with `rquota`.
- **A HuggingFace account + access token** (read scope). Create one at
  <https://huggingface.co/settings/tokens>; make sure you can view
  <https://huggingface.co/amd/GLM-5.2-MXFP4> while logged in.
- **`opencode`** installed wherever you want to *use* the model (the node, the
  login node, or your laptop) — see <https://opencode.ai>.
- Optional, for a nicer opencode config merge: **`jq`** on the machine running
  `opencode-setup.sh`.

---

## Repository contents

| File | Purpose |
|---|---|
| `serve-glm52.sh` | The core one-click script: preflight, pull, download, serve, stop, status |
| `serve-glm52.sbatch` | SLURM batch wrapper around `serve-glm52.sh serve` (bun161 recipe) |
| `glm52-env.example` | Config template — copy to `glm52.env` and edit |
| `opencode-setup.sh` | Writes/merges the opencode provider config on any machine |
| `opencode.glm52.json` | The provider template `opencode-setup.sh` fills in |
| `share-glm52.sh` | Optional: public HTTPS tunnel via Cloudflare for users without SSH |
| `README.md` | This file |

Secrets never live in the repo: `glm52.env` (your HF token), the generated API
key, and the `.sif` are gitignored / stored under `$MODEL_CACHE_DIR` on scratch.

---

## Walkthrough

### Step 0 — Get the code onto Bunya

Clone (or `scp`) to a scratch location visible to the compute node:

```bash
cd /scratch/user/$USER
git clone <this-repo-url> glm52-bunya
cd glm52-bunya
```

### Step 1 — Configure

```bash
cp glm52-env.example glm52.env
$EDITOR glm52.env
```

At minimum set two values:

- `MODEL_CACHE_DIR` — an absolute scratch path, e.g.
  `/scratch/user/$USER/glm52/hf-cache`. This holds the ~380 GB of weights, the
  `.sif`, the API key, and the server log/PID. It must be writable from the
  compute node.
- `HF_TOKEN` — your HuggingFace token. (Or leave it blank and set `HF_TOKEN_FILE`
  to a path containing just the token.)

Everything else defaults to the correct MI355X recipe (see
[Configuration reference](#configuration-reference)).

### Step 2 — Allocate an MI355X node

Apptainer isn't on the login nodes, so grab a node first (the `--gres` lands you
on a free MI355X node — `bun159`, `bun160`, or `bun161`):

```bash
salloc --partition=admin_test --account=a_rcc --qos=sdf --nodes=1 \
  --gres=gpu:mi355x:8 --ntasks-per-node=1 --cpus-per-task=192 --mem=1800G \
  --time=08:00:00 --job-name=glm52 \
  srun --export=ALL --pty /bin/bash -l
```

You're now on an MI355X node (`hostname` tells you which). `cd` back to the repo
(`cd /scratch/user/$USER/glm52-bunya`).

### Step 3 — Build the image + fetch the weights (first time)

```bash
source glm52.env
./serve-glm52.sh pull        # build the .sif from the container image (one-time)
./serve-glm52.sh download    # prefetch the ~380 GB of MXFP4 weights (no GPU used)
```

Both are resumable — rerun if interrupted. `pull` and `download` don't touch the
GPUs, so you can also do them in a cheaper CPU allocation ahead of time if you
prefer; they just need a compute node (for Apptainer) with outbound internet.

### Step 4 — Serve

```bash
./serve-glm52.sh serve --detach
```

`--detach` starts the container, waits until the model is loaded and healthy
(watch progress with `tail -f $MODEL_CACHE_DIR/glm52-server.log`), prints a
connection banner, and **returns your shell** so you can run opencode right there
on the node. Drop `--detach` if you'd rather the script stay attached and tear
the server down on Ctrl-C.

First healthy startup with cached weights takes several minutes (loading across
8 GPUs + graph capture).

**Prefer batch?** Instead of `salloc`, submit the job (it runs `pull`+`serve`;
prefetch the weights first with an interactive `download` so the paid GPU job
isn't waiting on a 380 GB download):

```bash
mkdir -p logs
sbatch serve-glm52.sbatch
# once RUNNING, read node + endpoint + key from the job log:
grep -A 20 'GLM-5.2 is up' logs/glm52-<jobid>.out
```

The batch job serves until walltime or `scancel <jobid>` (SIGTERM is trapped and
the server stops cleanly).

### Step 5 — Verify it's serving

```bash
export SGLANG_API_KEY="$(cat $MODEL_CACHE_DIR/glm52-api-key)"
curl -s http://127.0.0.1:30000/v1/models -H "Authorization: Bearer $SGLANG_API_KEY"

curl -s http://127.0.0.1:30000/v1/chat/completions \
  -H "Authorization: Bearer $SGLANG_API_KEY" -H "Content-Type: application/json" \
  -d '{"model":"glm-5.2","messages":[{"role":"user","content":"Say hi in one word."}]}'
```

The first lists `glm-5.2`; the second returns a completion.

### Step 6 — Connect opencode

`opencode-setup.sh` writes/merges a `glm52-bunya` provider into
`~/.config/opencode/opencode.json` (safe `jq` merge if available; otherwise it
prints the block to paste). The config references `{env:SGLANG_API_KEY}`, so
export that variable in the shell that launches opencode. Then restart opencode
and pick **GLM 5.2 (Bunya MI355X)** via `/models`.

**A) On the GPU node itself** (simplest — pairs with `serve --detach`). Open a second
shell into the running job, then set up opencode there:

```bash
# from the login node, attach a second shell to your job:
srun --overlap --jobid <jobid> --pty /bin/bash -l
cd /scratch/user/$USER/glm52-bunya
./opencode-setup.sh --host localhost
export SGLANG_API_KEY="$(cat $MODEL_CACHE_DIR/glm52-api-key)"
opencode
```

**B) On your laptop** (tunnel through the login node — works everywhere):

```bash
# in one terminal, keep this open (<node> = the hostname the serve banner printed):
ssh -N -L 30000:<node>:30000 <user>@bunya1.rcc.uq.edu.au

# in another: (grab the key value from the cluster first)
./opencode-setup.sh --host localhost --api-key <key>
export SGLANG_API_KEY=<key>
opencode
```

Once selected, ask opencode to read/edit a file or run a command — tool calls and
reasoning are handled server-side by the `glm47`/`glm45` parsers, so the full
agentic loop just works.

### Step 7 — Shut down

```bash
./serve-glm52.sh stop        # stop the server (or just let the allocation end)
scancel <jobid>              # for a batch job
```

---

## Sharing with someone who has no SSH access (optional)

`share-glm52.sh` exposes the running endpoint over public HTTPS via a
**Cloudflare quick tunnel** — outbound-only, no root, no Cloudflare account. Run
it on the GPU node after the server is up:

```bash
./share-glm52.sh share --detach     # prints https://<random>.trycloudflare.com
```

On first use it downloads `cloudflared` into `$MODEL_CACHE_DIR/cloudflared/`,
checks the local server is healthy, opens the tunnel, and prints a ready-to-paste
opencode provider block (with the public URL and API key) to hand over. The
recipient drops it into their `~/.config/opencode/opencode.json`, restarts
opencode, and picks **GLM 5.2 (shared)** via `/models`. Manage with
`./share-glm52.sh status` / `stop`.

> ⚠️ Needs the compute node to have **outbound internet** (same path the weights
> download uses). If that's blocked, use the SSH tunnel in Step 6B instead.

> ⚠️ **The public URL + API key together grant full use of your model and your
> Bunya GPU-hours (billed to `a_rcc`).** Share the key over a private channel
> only, rotate it if it leaks (delete `$MODEL_CACHE_DIR/glm52-api-key` and
> restart the server), and **check RCC's acceptable-use policy before exposing
> HPC compute externally** — the API key is the only gate.

---

## Script reference

```
./serve-glm52.sh [serve]         start serving (default), stays attached
./serve-glm52.sh serve --detach  start serving, wait until healthy, return the shell
./serve-glm52.sh pull            build the .sif from the container image (one-time)
./serve-glm52.sh download        prefetch weights only (no GPU)
./serve-glm52.sh stop            stop the server
./serve-glm52.sh status          server state + health check

./opencode-setup.sh [--host H] [--port P] [--api-key K] [--embed-key] [--config PATH]
                                 write/merge the opencode provider config
                                 (--embed-key writes the key literally instead of {env:...})

./share-glm52.sh [share]         open a public HTTPS Cloudflare tunnel (add --detach to background)
./share-glm52.sh stop            take the tunnel offline
./share-glm52.sh status          tunnel state + current public URL
```

Handy extras:

```bash
tail -f $MODEL_CACHE_DIR/glm52-server.log           # follow server startup / requests
srun --overlap --jobid <jobid> --pty /bin/bash -l   # second shell on the serving node
```

---

## Configuration reference

All knobs live in `glm52.env` (copied from `glm52-env.example`). Anything you
`export` in your shell before running a script takes precedence over the file.

| Variable | Default | Meaning |
|---|---|---|
| `MODEL_CACHE_DIR` | *(required)* | Scratch path for HF cache + `.sif` + API key + server log (~450 GB free) |
| `HF_TOKEN` | *(required first run)* | HuggingFace token for the weights download |
| `HF_TOKEN_FILE` | — | Alternative to `HF_TOKEN`: path to a file containing just the token |
| `SIF_PATH` | `$MODEL_CACHE_DIR/glm52-mi355x.sif` | Where the Apptainer image is stored |
| `APPTAINER_CACHEDIR` | *(near `$MODEL_CACHE_DIR`)* | Apptainer layer cache (kept off `/home`) |
| `APPTAINER_TMPDIR` | *(near `$MODEL_CACHE_DIR`)* | Apptainer build scratch (kept off `/home`) |
| `SGLANG_API_KEY` | *(auto-generated)* | Endpoint bearer key; if unset, generated and saved to `$MODEL_CACHE_DIR/glm52-api-key` |
| `MODEL_ID` | `amd/GLM-5.2-MXFP4` | Model repo to serve |
| `SERVED_MODEL_NAME` | `glm-5.2` | Name clients use in the `model` field |
| `SGLANG_IMAGE` | `docker://lmsysorg/sglang-rocm:v0.5.13.post1-rocm720-mi35x-20260618` | Container image (pulled into `SIF_PATH`) |
| `PORT` | `30000` | Endpoint port on the node |
| `TP_SIZE` | `8` | Tensor-parallel degree = **total GPUs used** (8 = whole node) |
| `DP_SIZE` | `1` | dp-attention groups; `>1` adds `--dp N --enable-dp-attention` and must divide `TP_SIZE` (does *not* change the GPU count) |
| `CONTEXT_LEN` | `262144` | Max context length |
| `MEM_FRACTION` | `0.85` | SGLang `--mem-fraction-static` |
| `ENABLE_AITER_ALLREDUCE_FUSION` | `1` | Toggle `--enable-aiter-allreduce-fusion` (set `0` if allreduce crashes) |
| `READY_TIMEOUT` | `7200` | Seconds to wait for health before giving up |
| `EXTRA_SGLANG_ARGS` | — | Extra flags appended verbatim to `sglang.launch_server` |

**`TP_SIZE` is the total GPU count** — set it to the number of GPUs you allocate
(**8** on an MI355X node). `DP_SIZE` only matters for dp-attention: it splits
those same `TP_SIZE` GPUs into `DP_SIZE` attention groups (so it must divide
`TP_SIZE`) and does *not* multiply the GPU count. Pure **TP8** (`DP_SIZE=1`) is
the default and best for a single latency-sensitive opencode session; for
high-concurrency throughput try `TP_SIZE=8 DP_SIZE=2`. The script warns if
`TP_SIZE` doesn't match your allocation (e.g. `TP_SIZE=4` on an 8-GPU node leaves
4 idle — which is the trap the earlier `TP4×DP2` default fell into).

---

## Running on MI300X/MI325X instead

If you point this at a gfx942 node (Bunya's MI300x in `gpu_rocm`, or another
cluster's MI325X), switch to the FP8 variant + the `mi30x` image and drop
data-parallelism:

```bash
export MODEL_ID="zai-org/GLM-5.2-FP8"
export SGLANG_IMAGE="docker://lmsysorg/sglang-rocm:v0.5.13.post1-rocm700-mi30x-20260616"
export TP_SIZE=8
export DP_SIZE=1
```

FP8 weights are ~750 GB, so MI300X (192 GB) needs all 8 GPUs; MI325X (256 GB)
needs at least 4. `serve-glm52.sh` also detects gfx942 at runtime and reminds
you of this. (For a fuller gfx942 treatment see the MI325X sibling repo.)

---

## Notes & troubleshooting

- **"apptainer not found"**: you're on a login node. Apptainer is only on compute
  nodes — start an allocation (Step 2) first.
- **Startup time**: with cached weights, several minutes to healthy. Watch
  details with `tail -f $MODEL_CACHE_DIR/glm52-server.log`. The script
  health-polls and prints the banner only when `/health` returns 200.
- **Detach didn't return my shell**: make sure the copy of `serve-glm52.sh` on
  the cluster is current — `grep -c DETACH serve-glm52.sh` should be non-zero.
- **`--rocm` / GPUs not found**: confirm you actually got GPUs
  (`echo $ROCR_VISIBLE_DEVICES`, `rocminfo | grep gfx`). The script forwards the
  SLURM GPU-visibility vars into the container so ROCm sees exactly your
  allocation.
- **`salloc` rejected**: verify the recipe is still current (`sinfo`/`sacctmgr`
  commands above). `admin_test` is a restricted partition; if access was revoked,
  ask `rcc-support@uq.edu.au`.
- **Crash: "CPU number N is not eligible; choose between [...]"** (in
  `set_gpu_proc_affinity`): the image enables `SGLANG_SET_CPU_AFFINITY=1`, but
  SGLang pins workers to CPUs from the *full* node topology, which fail under a
  SLURM cgroup that only owns a subset of cores. The script forces
  `SGLANG_SET_CPU_AFFINITY=0` (`SET_CPU_AFFINITY` env) so this is handled by
  default. If you're on an older copy, either update or run with
  `export APPTAINERENV_SGLANG_SET_CPU_AFFINITY=0` before `serve`, or allocate the
  whole node's CPUs (`--cpus-per-task=384` / `--exclusive`).
- **Crash during CUDA-graph capture: `.aiter/jit/module_*.so: undefined symbol`**
  (e.g. `getPaddedM`): aiter JIT-compiles some kernels at runtime and caches the
  `.so` files in `$HOME/.aiter/jit` (your home is bind-mounted into the
  container, so the cache persists across runs). If a startup was **killed
  mid-compile** — which is easy to do while iterating — the cached module is
  truncated and every later run reloads the broken copy. Fix: clear it and
  restart —
  ```bash
  ./serve-glm52.sh stop
  rm -rf ~/.aiter        # (and ~/.triton if a triton kernel is the culprit)
  ./serve-glm52.sh serve --detach
  ```
  It recompiles cleanly on first use (adds ~a minute). Note this cache also eats
  your home file-quota — see `rquota`.
- **No speculative decoding**: MTP/EAGLE draft kernels aren't validated on ROCm
  for this model yet, so no `--speculative-*` flags are passed.
- **`--enable-aiter-allreduce-fusion`** comes from the wafer.ai post. If you hit
  allreduce/RCCL crashes, set `ENABLE_AITER_ALLREDUCE_FUSION=0`.
- **Out of space during pull/download**: `APPTAINER_CACHEDIR`/`APPTAINER_TMPDIR`
  default to scratch near `MODEL_CACHE_DIR` — make sure that has room and isn't
  on `/home`. Check with `rquota`.
- **opencode doesn't see the model**: it only reads config at startup — restart
  it after `opencode-setup.sh`, and make sure `SGLANG_API_KEY` is exported in
  that shell (or you used `--embed-key`).

---

## Sources

- [wafer.ai: GLM 5.2 on AMD](https://www.wafer.ai/blog/glm52-amd) — MXFP4/MI355X reference numbers (213 tok/s single stream, 2626 tok/s/node at TP4×DP2)
- [UQ-RCC Bunya docs](https://github.com/UQ-RCC/hpc-docs) — Apptainer, SLURM, GPU partitions, filesystems
- [SGLang cookbook: GLM-5.2](https://docs.sglang.io/cookbook/autoregressive/GLM/GLM-5.2) — pinned ROCm images, parser flags, AMD caveats
- [amd/GLM-5.2-MXFP4](https://huggingface.co/amd/GLM-5.2-MXFP4) · [zai-org/GLM-5.2-FP8](https://huggingface.co/zai-org/GLM-5.2-FP8)
- [opencode custom providers](https://opencode.ai/docs/providers/)

---

## License

The scripts in this repo are provided under the MIT License (see `LICENSE` —
fill in your name before publishing). The container images
(`lmsysorg/sglang-rocm`), the SGLang engine, and the model weights
(`amd/GLM-5.2-MXFP4` / `zai-org/GLM-5.2-FP8`) are covered by their own separate
licenses — review and comply with those independently.

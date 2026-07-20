#!/bin/bash --login
#SBATCH --job-name=glm52-mi355x
#SBATCH --partition=admin_test
#SBATCH --account=a_rcc
#SBATCH --qos=sdf
#SBATCH --nodes=1
#SBATCH --gres=gpu:mi355x:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=192
#SBATCH --mem=1800G
#SBATCH --time=24:00:00
#SBATCH --output=logs/glm52-%j.out
#SBATCH --error=logs/glm52-%j.out
#
# Batch wrapper around serve-glm52.sh for Bunya's MI355X nodes (bun159/160/161).
# Submit from /scratch (not /home, not RDM) with:
#
#   mkdir -p logs && sbatch serve-glm52.sbatch
#
# The partition/account/qos/gres above are the discovered bun161 recipe:
#   --partition=admin_test  (the MI355X nodes live here, not gpu_rocm)
#   --account=a_rcc  --qos=sdf  --gres=gpu:mi355x:8
# The MI355X nodes are bun159/bun160/bun161; the gres lands you on a free one.
# Add e.g. '#SBATCH --nodelist=bun161' if you want to pin a specific node.
# Confirm the current recipe with:  sinfo -o "%P %.10l %G %f" | grep mi355x
#
# The connection banner (node, endpoint URL, API key location, tunnel command)
# is printed into the job log. Once the job is RUNNING:
#
#   grep -A 20 'GLM-5.2 is up' logs/glm52-<jobid>.out
#
# The server runs until walltime or 'scancel <jobid>' (the script traps SIGTERM
# and stops the container cleanly).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "[glm52-sbatch] Job $SLURM_JOB_ID on $(hostname) at $(date)"
exec ./serve-glm52.sh serve

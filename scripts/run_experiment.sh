#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export BBNBP_CACHE_DIR="${BBNBP_CACHE_DIR:-$PWD/.cache/bbnbp}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PWD/.cache/matplotlib}"
export OMP_NUM_THREADS=1
export NUMBA_NUM_THREADS=1

# Each invocation gets a fresh bundle and preserves existing research evidence.
exec .venv/bin/python scripts/run_experiment.py "$@"

#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export BBNBP_CACHE_DIR="${BBNBP_CACHE_DIR:-$PWD/.cache/bbnbp}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PWD/.cache/matplotlib}"
export OMP_NUM_THREADS=1
export NUMBA_NUM_THREADS=1

# E0: separate validation tuning, then paired code-capacity development evaluation.
.venv/bin/python scripts/sweep_scaling_factor.py --config configs/cc_bb72_e0.yaml --output results/tuned_scaling.csv
.venv/bin/python -m bbnbp.evaluate --config configs/cc_bb72_e0.yaml --decoders bp,bp_tuned,bp_only,bp_tuned_only

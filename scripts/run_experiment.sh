#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export BBNBP_CACHE_DIR="${BBNBP_CACHE_DIR:-$PWD/.cache/bbnbp}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PWD/.cache/matplotlib}"
export OMP_NUM_THREADS=1
export NUMBA_NUM_THREADS=1

# M6 stages are explicit; E0 remains the default.
case "${1:-}" in
  m6-tune|m6-train)
    stage="${1#m6-}"
    shift
    exec .venv/bin/python scripts/m6.py "$stage" "$@"
    ;;
  e1|e2)
    exec .venv/bin/python scripts/m6.py "$@"
    ;;
esac

# Each E0 invocation gets a fresh bundle and preserves existing research evidence.
exec .venv/bin/python scripts/run_experiment.py "$@"

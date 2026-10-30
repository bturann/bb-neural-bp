"""Run E0 in a fresh timestamped bundle without replacing previous evidence."""

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/cc_bb72_e0.yaml")
    parser.add_argument("--output-root", default="results")
    args = parser.parse_args()
    bundle = Path(args.output_root).resolve()/datetime.now(timezone.utc).strftime("e0-%Y%m%dT%H%M%S%fZ")
    bundle.mkdir(parents=True, exist_ok=False)
    table = bundle/"tuned_scaling.csv"
    try:
        subprocess.run([sys.executable, str(ROOT/"scripts/sweep_scaling_factor.py"),
                        "--config", args.config, "--output", str(table)], cwd=ROOT, check=True)
        subprocess.run([sys.executable, "-m", "bbnbp.evaluate", "--config", args.config,
                        "--tuning-table", str(table), "--output-root", str(bundle),
                        "--decoders", "bp,bp_tuned,bp_only,bp_tuned_only"], cwd=ROOT, check=True)
    except subprocess.CalledProcessError as exc:
        (bundle/"FAILURE.txt").write_text(f"Command failed with exit status {exc.returncode}: {exc.cmd}\n")
        raise
    print(f"Experiment bundle: {bundle}", flush=True)


if __name__ == "__main__":
    main()

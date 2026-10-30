"""Explicit entry points for M6; no stage silently launches another experiment."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
os.environ.setdefault("BBNBP_CACHE_DIR", str(ROOT/".cache/bbnbp"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT/".cache/matplotlib"))
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="stage", required=True)
    commands.add_parser("tune", help="Full independent scaling-factor sweep")
    train = commands.add_parser("train", help="One seed, at most four wall-clock hours")
    train.add_argument("--device", choices=("cuda", "mps"), required=True)
    train.add_argument("--seed", type=int, choices=(1234, 1235, 1236), required=True)
    train.add_argument("--resume", type=Path)
    for name in ("e1", "e2"):
        evaluation = commands.add_parser(name)
        evaluation.add_argument("--checkpoint", action="append", required=True,
                                help="Repeat for all three selected best.pt files")
    args = parser.parse_args()
    python = sys.executable
    timeout = None
    if args.stage == "tune":
        command = [python, "scripts/sweep_scaling_factor.py", "--config", "configs/m6/tune.yaml",
                   "--output", "results/m6/tuned_scaling.csv"]
    elif args.stage == "train":
        command = [python, "-u", "-m", "bbnbp.train", "--config",
                   f"configs/m6/train_{args.device}_{args.seed}.yaml",
                   "--output-root", "results/m6/training", "--checkpoint-root", "checkpoints/m6"]
        if args.resume:
            from bbnbp.checkpoints import read_checkpoint
            prior_seconds = read_checkpoint(args.resume)["training_state"].get("gpu_elapsed_s", 0)
            timeout = 14400-prior_seconds
            if timeout <= 0:
                parser.error("This run exhausted its cumulative four-hour GPU budget")
            command += ["--resume", str(args.resume)]
        else:
            timeout = 14400
    else:
        from bbnbp.checkpoints import read_checkpoint
        if len(args.checkpoint) != 3:
            parser.error("E1/E2 require exactly three independently trained models")
        seeds = set()
        for path in args.checkpoint:
            checkpoint = Path(path)
            payload = read_checkpoint(checkpoint)
            runtime = payload["config"]["train"].get("runtime", {})
            if checkpoint.name != "best.pt" or runtime.get("purpose") != "m6_train":
                parser.error("E1/E2 require M6-selected best.pt checkpoints, never development weights")
            seeds.add(payload["config"]["seed"])
            # Locate the corresponding training report by the checkpoint run ID.
            report_path = ROOT/"results/m6/training"/checkpoint.parent.name/"training_summary.json"
            if not report_path.exists() or not json.loads(report_path.read_text())["training_complete"]:
                parser.error(f"Completed training report required: {report_path}")
        if seeds != {1234, 1235, 1236}:
            parser.error("The predeclared main training seeds are 1234, 1235, 1236")
        command = [python, "-u", "-m", "bbnbp.evaluate", "--config", f"configs/m6/{args.stage}.yaml",
                   "--decoders", "bp,bp_tuned,nbp", "--tuning-table", "results/m6/tuned_scaling.csv",
                   "--output-root", f"results/m6/{args.stage}", "--progress"]
        for checkpoint in args.checkpoint:
            command += ["--checkpoint", checkpoint]
    try:
        # An external deadline also bounds setup, validation, plotting, and shutdown.
        subprocess.run(command, check=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        parser.exit(2, "Four-hour wall-clock budget reached; use retained checkpoints and inspect the run before resuming.\n")


if __name__ == "__main__":
    main()

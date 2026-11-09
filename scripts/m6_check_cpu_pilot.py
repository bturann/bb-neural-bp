"""Tune all T=15 scaling factors, then run a bounded fresh-data CPU comparison."""

import argparse
import hashlib
import json
import os
import signal
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from bbnbp.config import load_config
from bbnbp.io import git_sha, source_manifest

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/m6/cpu_pilot_check.yaml")
    parser.add_argument("--budget", default="configs/m6/cpu_pilot_check_budget.yaml")
    parser.add_argument("--output-root", default="results/m6_cpu_pilot_check")
    parser.add_argument("--reuse-tuning", type=Path)
    args = parser.parse_args()
    os.chdir(ROOT)
    config = load_config(args.config)
    budget = yaml.safe_load(Path(args.budget).read_text())
    if config.eval.purpose != "development" or len(config.eval.checkpoints) != 1:
        parser.error("This check requires one frozen checkpoint and development samples")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")+"-"+config.config_hash[:8]
    directory = Path(args.output_root).resolve()/run_id
    directory.mkdir(parents=True, exist_ok=False)
    config.save(directory/"config.yaml")
    (directory/"budget.yaml").write_text(yaml.safe_dump(budget))
    (directory/"source_manifest.json").write_text(json.dumps(source_manifest(), indent=2)+"\n")
    checkpoint = Path(config.eval.checkpoints[0])
    sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    env = dict(os.environ, OMP_NUM_THREADS="1", NUMBA_NUM_THREADS="1",
               BBNBP_CACHE_DIR=str(ROOT/".cache/bbnbp"), MPLCONFIGDIR=str(ROOT/".cache/matplotlib"))
    commands = [
        ("tuning", [sys.executable, "-u", "scripts/sweep_scaling_factor.py", "--config",
                    str(directory/"config.yaml"), "--output", str(directory/"tuned_scaling.csv")]),
        ("comparison", [sys.executable, "-u", "scripts/m6_compare_stage1.py", "--config",
                        str(directory/"config.yaml"), "--tuning-table", str(directory/"tuned_scaling.csv"),
                        "--output", str(directory/"comparison")]),
    ]
    print(f"CPU pilot check bundle: {directory}", flush=True)
    started = time.perf_counter()
    stages, stopped = [], False
    if args.reuse_tuning:
        from m6_compare_stage1 import validate_tuning_table
        validate_tuning_table(config, args.reuse_tuning)
        shutil.copyfile(args.reuse_tuning, directory/"tuned_scaling.csv")
        shutil.copyfile(args.reuse_tuning.with_suffix(".config.yaml"), directory/"tuned_scaling.config.yaml")
        if args.reuse_tuning.with_suffix(".meta.json").exists():
            shutil.copyfile(args.reuse_tuning.with_suffix(".meta.json"), directory/"tuned_scaling.meta.json")
        (directory/"reused_tuning.json").write_text(json.dumps(dict(
            source=str(args.reuse_tuning), sha256=hashlib.sha256(args.reuse_tuning.read_bytes()).hexdigest()), indent=2)+"\n")
        commands = commands[1:]
        stages.append(dict(name="reuse_tuning", returncode=0, elapsed_s=time.perf_counter()-started))
    if budget.get("include_initial_reference", False):
        commands.append(("initial_reference", [sys.executable, "-u", "scripts/m6_initialization_reference.py", str(directory/"comparison")]))
    expected_stages = len(stages)+len(commands)
    try:
        for name, command in commands:
            remaining = budget["wall_seconds"]-(time.perf_counter()-started)
            if remaining <= 0:
                stopped = True
                break
            print(f"Starting {name}; remaining budget {remaining:.1f}s", flush=True)
            with (directory/f"{name}.log").open("w") as log:
                child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                try:
                    child.wait(timeout=remaining)
                except (subprocess.TimeoutExpired, KeyboardInterrupt):
                    stopped = True
                    child.send_signal(signal.SIGINT)
                    try:
                        child.wait(timeout=budget["interrupt_grace_seconds"])
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait(timeout=budget["kill_grace_seconds"])
            stages.append(dict(name=name, returncode=child.returncode,
                               elapsed_s=time.perf_counter()-started))
            print(f"{name} returned {child.returncode}; elapsed {time.perf_counter()-started:.1f}s", flush=True)
            if stopped or child.returncode:
                break
    finally:
        report = dict(purpose=budget["purpose"], git_sha=git_sha(), stages=stages,
                      elapsed_s=time.perf_counter()-started, wall_limit_reached_or_interrupted=stopped,
                      complete=len(stages) == expected_stages and all(row["returncode"] == 0 for row in stages) and not stopped,
                      checkpoint_sha256=sha, gpu_seconds=0, final_test_data_used=False, m6_complete=False)
        (directory/"status.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2), flush=True)
    if not report["complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

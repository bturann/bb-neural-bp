"""Run one bounded CPU learning pilot and retain measured runtime/profile evidence."""

import argparse
import json
import os
import pstats
import signal
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
    parser.add_argument("--config", default="configs/m6/cpu_pilot.yaml")
    parser.add_argument("--budget", default="configs/m6/cpu_pilot_budget.yaml")
    parser.add_argument("--output-root", default="results/m6_cpu_pilot")
    args = parser.parse_args()
    os.chdir(ROOT)
    config = load_config(args.config)
    budget = yaml.safe_load(Path(args.budget).read_text())
    if config.train.device != "cpu" or config.train.runtime.purpose != "smoke":
        parser.error("This launcher accepts CPU development configs only")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")+"-"+config.config_hash[:8]
    directory = Path(args.output_root).resolve()/run_id
    directory.mkdir(parents=True, exist_ok=False)
    config.save(directory/"config.yaml")
    (directory/"budget.yaml").write_text(yaml.safe_dump(budget))
    (directory/"source_manifest.json").write_text(json.dumps(source_manifest(), indent=2)+"\n")
    env = dict(os.environ, OMP_NUM_THREADS="1", NUMBA_NUM_THREADS="1",
               BBNBP_CACHE_DIR=str(ROOT/".cache/bbnbp"), MPLCONFIGDIR=str(ROOT/".cache/matplotlib"))
    profile = directory/"profile.pstats"
    command = [sys.executable, "-u", "-m", "cProfile", "-o", str(profile), "-m", "bbnbp.train",
               "--config", str(directory/"config.yaml"), "--output-root", str(directory/"training"),
               "--checkpoint-root", "checkpoints/m6_cpu_pilot"]
    print(f"CPU pilot bundle: {directory}", flush=True)
    started = time.perf_counter()
    timed_out = False
    with (directory/"console.log").open("w") as log:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            child.wait(timeout=budget["wall_seconds"])
        except subprocess.TimeoutExpired:
            timed_out = True
            child.send_signal(signal.SIGINT)
            try:
                child.wait(timeout=budget["interrupt_grace_seconds"])
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=budget["kill_grace_seconds"])
        except BaseException:
            child.send_signal(signal.SIGINT)
            try:
                child.wait(timeout=budget["interrupt_grace_seconds"])
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=budget["kill_grace_seconds"])
            raise
    receipt = dict(purpose=budget["purpose"], git_sha=git_sha(), device="cpu", gpu_seconds=0,
                   elapsed_s=time.perf_counter()-started, returncode=child.returncode,
                   stopping="wall_time_limit" if timed_out else "child_completed" if child.returncode == 0 else "child_failed",
                   final_test_data_used=False, m6_complete=False,
                   timing_scope="Profiled end-to-end CPU pilot; profiler overhead included")
    if profile.exists():
        stats = pstats.Stats(str(profile))
        ranked = sorted(stats.stats.items(), key=lambda pair: pair[1][3], reverse=True)
        functions = [dict(file=filename, line=line, function=function, primitive_calls=value[0],
                          total_calls=value[1], self_s=value[2], cumulative_s=value[3])
                     for (filename, line, function), value in ranked[:budget["profile_top_functions"]]]
        (directory/"profile_summary.json").write_text(json.dumps(dict(total_s=stats.total_tt, functions=functions), indent=2)+"\n")
    (directory/"pilot_status.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps(receipt, indent=2), flush=True)
    if child.returncode != 0 and not timed_out:
        raise SystemExit(child.returncode)


if __name__ == "__main__":
    main()

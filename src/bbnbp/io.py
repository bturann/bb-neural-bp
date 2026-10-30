"""Results are written by executed evaluations, with exact CSV column validation."""

import csv
import hashlib
import subprocess
from pathlib import Path

RESULT_COLUMNS = """run_id git_sha config_hash timestamp code noise_model rounds basis p
decoder stage1 T weight_sharing osd_method osd_order ms_scaling shots stage1_failures
osd_calls osd_rate logical_failures ler ler_lo ler_hi ler_stage1 lat_mean_us lat_p50_us
lat_p99_us lat_max_us lat_threads device seed""".split()


def git_sha():
    root = Path(__file__).resolve().parents[2]
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True,
                                      stderr=subprocess.DEVNULL).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--", ".",
                                         ":(exclude)results", ":(exclude)checkpoints", ":(exclude)figures"],
                                        cwd=root, text=True).strip()
        return sha + ("-dirty" if dirty else "")
    except subprocess.CalledProcessError:
        return "uncommitted"


def source_manifest():
    root = Path(__file__).resolve().parents[2]
    paths = []
    for directory in ("src", "scripts", "configs", "reference"):
        paths.extend(p for p in (root/directory).rglob("*")
                     if p.is_file() and "__pycache__" not in p.parts)
    paths.extend(root/name for name in ("requirements.txt", "pyproject.toml"))
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths)}


def append_results(csv_path, rows):
    path = Path(csv_path)
    rows = list(rows)
    for row in rows:
        if set(row) != set(RESULT_COLUMNS):
            raise ValueError(f"Result schema mismatch: {set(row) ^ set(RESULT_COLUMNS)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    if exists:
        with path.open(newline="") as f:
            if next(csv.reader(f)) != RESULT_COLUMNS:
                raise ValueError("Existing CSV has a different schema")
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)

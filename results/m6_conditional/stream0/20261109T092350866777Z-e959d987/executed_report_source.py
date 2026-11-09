"""Export diagnostic scores and plot fixed-training versus untouched unseen outcomes."""

import argparse
import json
import platform
from datetime import datetime,timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bbnbp.config import load_config
from bbnbp.evaluate import cpu_model_label,tuned_factor
from bbnbp.io import append_results
from bbnbp.metrics import wilson


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    args=parser.parse_args()
    d=args.directory
    if any((d/name).exists() for name in ("results.csv","learning_diagnostic.png","learning_diagnostic.pdf")):
        parser.error("Refusing to overwrite exported evidence")
    r=json.loads((d/"summary.json").read_text())
    c=load_config(d/"config.yaml")
    rows=[]
    timestamp=datetime.strptime(d.name.split("-")[0],"%Y%m%dT%H%M%S%fZ").replace(tzinfo=timezone.utc).isoformat()
    for name,filename,seed,sharing,scaling in (
        ("nbp_initial_fixed_train","scores-train-step0.npz",r["settings"]["training_pool_seed"],c.model.sharing,c.model.ms_scaling),
        ("nbp_final_fixed_train","scores-train-final.npz",r["settings"]["training_pool_seed"],c.model.sharing,c.model.ms_scaling),
        ("nbp_initial_unseen","scores-unseen-initial.npz",r["settings"]["unseen_seed"],c.model.sharing,c.model.ms_scaling),
        ("nbp_final_unseen","scores-unseen-final.npz",r["settings"]["unseen_seed"],c.model.sharing,c.model.ms_scaling),
        ("bp_tuned_unseen","scores-unseen-tuned.npz",r["settings"]["unseen_seed"],"scalar",tuned_factor(d/"tuned_scaling.csv",c,c.noise.p_grid[0]))):
        with np.load(d/filename) as values:
            shots=len(values["failed"])
            failed=int(values["failed"].sum())
            logical=int(values["logical_failed"].sum())
        lo,hi=wilson(logical,shots)
        rows.append(dict(run_id=d.name,git_sha=r["git_sha"],config_hash=c.config_hash,timestamp=timestamp,
            code=c.code.name,noise_model=c.noise.model,rounds=c.noise.rounds,basis=c.noise.basis,p=c.noise.p_grid[0],
            decoder=name+"_only",stage1=name,T=c.model.T,weight_sharing=sharing,osd_method="",osd_order="",
            ms_scaling=scaling,shots=shots,stage1_failures=failed,osd_calls=0,osd_rate=0,logical_failures=logical,
            ler=logical/shots,ler_lo=lo,ler_hi=hi,ler_stage1=logical/shots,lat_mean_us="",lat_p50_us="",lat_p99_us="",
            lat_max_us="",lat_threads=1,device=f"CPU {cpu_model_label(c.eval.cpu_model)}; {platform.platform()}",seed=seed))
    append_results(d/"results.csv",rows)
    with np.load(d/"scores-unseen-initial.npz") as initial,np.load(d/"scores-unseen-final.npz") as final:
        hard=initial["failed"]
        correct=~initial["logical_failed"]
        transitions=dict(scope="Descriptive decomposition of the fixed unseen outcomes; no new test or checkpoint selection.",
            initial_hard_cases=int(hard.sum()),initial_correct_cases=int(correct.sum()),
            initial_hard_now_correct=int((hard & ~final["logical_failed"]).sum()),
            initial_hard_now_correct_fraction=float((~final["logical_failed"][hard]).mean()),
            initial_correct_now_failed=int((correct & final["logical_failed"]).sum()),
            initial_correct_now_failed_fraction=float(final["logical_failed"][correct].mean()),
            correct_decodes_gained=int((initial["logical_failed"] & ~final["logical_failed"]).sum()),
            correct_decodes_lost=int((correct & final["logical_failed"]).sum()),
            final_converged_wrong_logical=int((~final["failed"] & final["logical_failed"]).sum()))
    (d/"unseen_transitions.json").write_text(json.dumps(transitions,indent=2)+"\n")
    training=[json.loads(line) for line in (d/"training_evaluation.jsonl").read_text().splitlines()]
    if training[-1]["step"]!=r["steps"]:
        training.append(dict(step=r["steps"],**r["final_train"]))
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    steps=[row["step"] for row in training]
    axes[0].plot(steps,[row["mean_loss"] for row in training],marker="o",color="#2166ac")
    axes[0].set(xlabel="Optimizer update",ylabel="Mean parity/logical multi-loss",title="Fixed training cases")
    axes[1].plot(steps,[100*(1-row["logical_failure_rate"]) for row in training],marker="o",color="#2166ac")
    axes[1].axhline(100*r["settings"]["training_min_correct_gain"],linestyle="--",color="black",label="Training gain screen")
    axes[1].set(xlabel="Optimizer update",ylabel="Logically correct Stage-1 decodes (%)",ylim=(0,100),title=f"{r['settings']['training_cases']:,} fixed training cases")
    axes[1].legend(fontsize=8)
    demand=[100*r[key]["fallback_demand"] for key in ("initial_unseen","final_unseen","tuned_unseen")]
    bars=axes[2].bar(["Initial","Final","Tuned BP"],demand,color=["#777777","#2166ac","#c65c15"])
    axes[2].bar_label(bars,fmt="%.2f%%",padding=3,fontsize=9)
    pool_name="reused validation cases" if r["settings"].get("validation_source") else "fixed unseen cases"
    axes[2].set(ylabel="Fallback demand (%)",ylim=(0,100),title=f"{r['settings']['unseen_cases']:,} {pool_name}")
    for ax in axes:ax.grid(axis="y",alpha=.2)
    fig.text(.5,.02,f"Single-stream diagnostic at p={c.noise.p_grid[0]:g}. Training is conditional on the declared mining decoder's failure. No OSD or latency claim.",ha="center",fontsize=8)
    fig.tight_layout(rect=(0,.06,1,1))
    for extension in ("png","pdf"):fig.savefig(d/f"learning_diagnostic.{extension}",dpi=180)
    plt.close(fig)
    (d/"executed_report_source.py").write_bytes(Path(__file__).read_bytes())
    print(d/"learning_diagnostic.png")


if __name__=="__main__":
    main()

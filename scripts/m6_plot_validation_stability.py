"""Render the validation-only ranking comparison from saved measured outcomes."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    args=parser.parse_args()
    d=args.directory
    if (d/"ranking_stability.png").exists() or (d/"ranking_stability.pdf").exists():
        parser.error("Refusing to overwrite rendered evidence")
    report=json.loads((d/"summary.json").read_text())
    original=json.loads((Path(report["settings"]["source"])/"summary.json").read_text())
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    colors={"parity":"#2166ac","bitwise":"#c65c15"}
    for i,arm in enumerate(("parity","bitwise")):
        ranking=report["rankings"][arm]
        root=Path(original["training_roots"][arm])
        run=list((root/"training").iterdir())[0]
        selected=json.loads((run/"best_validation.json").read_text())["mean_osd_rate"]
        final=json.loads((run/"validation.jsonl").read_text().splitlines()[-1])["mean_osd_rate"]
        axes[0].scatter(i-.12,100*(final-selected),marker="D",color="#777777",label="Original 256 shots/p" if i==0 else None)
        mean=100*ranking["last_minus_selected"]
        lo=100*ranking["paired_bootstrap"]["low"]
        hi=100*ranking["paired_bootstrap"]["high"]
        axes[0].errorbar(i+.12,mean,yerr=[[mean-lo],[hi-mean]],fmt="o",color=colors[arm],capsize=5,
                        label="New 2,048 shots/p + 95% interval" if i==0 else None)
        blocks=ranking["small_blocks"]
        axes[1].plot(np.arange(1,len(blocks)+1),[100*(r["last_rate"]-r["selected_rate"]) for r in blocks],
                     marker="o",label="Parity" if arm=="parity" else "BCE",color=colors[arm])
    for axis in axes:
        axis.axhline(0,color="black",linewidth=.8,linestyle="--")
        axis.set_ylabel("Last − original selected (percentage points)")
        axis.grid(axis="y",alpha=.2)
    axes[0].set_xticks([0,1],["Parity","BCE"])
    axes[0].set_title("Ranking on separate validation pools")
    axes[0].legend(fontsize=8)
    axes[1].set(xlabel="Disjoint 256-shot/p block",title="Small-pool rankings fluctuate")
    axes[1].set_xticks(range(1,9))
    axes[1].legend()
    fig.text(.5,.02,"Negative values favor the final checkpoint. Validation only; intervals are descriptive. Original model files are unchanged.",ha="center",fontsize=8)
    fig.tight_layout(rect=(0,.06,1,1))
    fig.savefig(d/"ranking_stability.png",dpi=180)
    fig.savefig(d/"ranking_stability.pdf")
    plt.close(fig)
    print(d/"ranking_stability.png")


if __name__=="__main__":
    main()

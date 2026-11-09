"""Render matched validation curves and observed fresh readout differences."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    parser.add_argument("--output-stem",default="readout_comparison")
    args=parser.parse_args()
    d=args.directory
    if (d/f"{args.output_stem}.png").exists() or (d/f"{args.output_stem}.pdf").exists():
        parser.error("Refusing to overwrite rendered evidence")
    report=json.loads((d/"summary.json").read_text())
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    tuned=None
    for name,label,color in (("full","Full-unroll control","#2166ac"),("frozen","First-converged readout","#c65c15")):
        root=Path(report["training_roots"][name])
        run=list((root/"training").iterdir())[0]
        rows=[json.loads(x) for x in (run/"validation.jsonl").read_text().splitlines()]
        axes[0].plot([r["step"] for r in rows],[100*r["mean_osd_rate"] for r in rows],marker="o",label=label,color=color)
        tuned=rows[0]["mean_tuned_rate"]
    axes[0].axhline(100*tuned,color="black",linestyle="--",label="Tuned BP")
    axes[0].set(xlabel="Optimizer update",ylabel="Fallback demand (%)",title="Shared larger validation pool")
    axes[0].legend(fontsize=8)
    values=[100*(p["models"]["nbp_frozen"]["fallback_demand"]-p["models"]["nbp_full"]["fallback_demand"]) for p in report["points"]]
    bars=axes[1].bar(range(len(values)),values,color=["#2166ac" if x<0 else "#c65c15" for x in values])
    axes[1].axhline(0,color="black",linewidth=.8)
    axes[1].set_xticks(range(len(values)),[f"{p['p']:.3f}" for p in report["points"]])
    axes[1].set(xlabel="Physical error probability p",ylabel="Frozen − full-unroll (percentage points)",title="Observed fresh-case difference")
    for bar,point in zip(bars,report["points"]):
        axes[1].annotate(f"pH={point['primary']['holm_pvalue']:.2g}",(bar.get_x()+bar.get_width()/2,bar.get_height()),
            xytext=(0,-14 if bar.get_height()<0 else 4),textcoords="offset points",ha="center",fontsize=8)
    lower,upper=axes[1].get_ylim()
    span=upper-lower
    axes[1].set_ylim(lower-.15*span,upper+.10*span)
    for ax in axes:ax.grid(axis="y",alpha=.2)
    fig.text(.5,.02,"Negative differences favor frozen readout. Single-seed development ablation; pH denotes five-point Holm-adjusted paired p-value.",ha="center",fontsize=8)
    fig.tight_layout(rect=(0,.06,1,1))
    fig.savefig(d/f"{args.output_stem}.png",dpi=180)
    fig.savefig(d/f"{args.output_stem}.pdf")
    plt.close(fig)
    print(d/f"{args.output_stem}.png")


if __name__=="__main__":
    main()

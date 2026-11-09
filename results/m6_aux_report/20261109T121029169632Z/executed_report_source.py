"""Create an auxiliary research report and standalone figures from saved evidence."""

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bbnbp.metrics import wilson


LABELS={"bp15":"BP15", "ordinary":"Tuned BP30", "bp30":"Tuned BP30",
        "matched_init":"Untrained matched30", "selected":"Learned30",
        "bp50":"Tuned BP50", "bp100":"Tuned BP100", "learned100":"Learned100", "native100":"Tuned native100"}


def save_figure(fig, root, name):
    fig.savefig(root/f"{name}.png",dpi=180,bbox_inches="tight")
    fig.savefig(root/f"{name}.pdf",bbox_inches="tight")
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("confirmation",type=Path)
    parser.add_argument("--accuracy",type=Path)
    parser.add_argument("--replication",type=Path)
    parser.add_argument("--latency",type=Path)
    parser.add_argument("--long-controls",nargs=2,type=Path)
    parser.add_argument("--long-latency",type=Path)
    parser.add_argument("--strong-latency",type=Path)
    parser.add_argument("--strong-study",type=Path)
    parser.add_argument("--strong-accuracy",type=Path)
    parser.add_argument("--extended-validation",type=Path)
    parser.add_argument("--cycle100-study",type=Path)
    parser.add_argument("--cycle100-accuracy",type=Path)
    args=parser.parse_args()
    d=Path("results/m6_aux_report")/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    d.mkdir(parents=True,exist_ok=False)
    inputs={}
    def read(root,name="summary.json"):
        path=root/name
        inputs[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())
    def link(root):return os.path.relpath(root,d)
    primary=read(args.confirmation)
    points=primary["points"]
    text=["# CPU auxiliary learned-decoder assessment", "",
        "This report separates fallback demand, logical accuracy and actual CPU latency. "
        "It is an auxiliary study; main M6/E1/E2 and M7 remain incomplete. Main final-test seeds are unused.","",
        "## Frozen method and independent Stage 1 result", "",
        "A structural 15-step neural min-sum model (1,455 trainable parameters) received "
        "4,000 updates on 32,000 unique BP15-rejected training cases at p=.001. "
        "Inference cycles the learned schedule for up to 30 iterations, with no extra training. "
        "The model and controls were frozen before drawing the 4,096 independent cases per noise point.","",
        "| Noise p | Tuned BP30 fallback | Matched initialization | Learned30 fallback | Primary Holm p |",
        "|---|---:|---:|---:|---:|"]
    fig,axes=plt.subplots(1,2,figsize=(9,3.8),sharey=True)
    for index,(point,axis) in enumerate(zip(points,axes)):
        n=point["shots"];counts=point["failures"]
        names=[point["ordinary_name"],"matched_init","selected"]
        text.append(f"| {point['p']:g} | {counts[names[0]]}/{n} ({100*counts[names[0]]/n:.3f}%) | "
            f"{counts['matched_init']}/{n} ({100*counts['matched_init']/n:.3f}%) | "
            f"{counts['selected']}/{n} ({100*counts['selected']/n:.3f}%) | {point['primary']['holm_pvalue']:.3g} |")
        values=np.array([counts[name]/n for name in names])*100
        bounds=np.array([wilson(counts[name],n) for name in names])*100
        axis.bar(np.arange(3),values,color=["#65758b","#a6b5c5","#2d8b77"],width=.65)
        axis.errorbar(np.arange(3),values,yerr=np.stack([values-bounds[:,0],bounds[:,1]-values]),fmt="none",color="black",capsize=4)
        axis.set_xticks(np.arange(3),["Tuned BP30","Untrained30","Learned30"],rotation=15)
        axis.set_title(f"p = {point['p']:g}; n = {n:,}")
        axis.set_ylim(0,100);axis.spines[["top","right"]].set_visible(False)
    axes[0].set_ylabel("Stage 1 fallback demand (%)")
    fig.suptitle("Independent auxiliary Stage 1 confirmation\nBars show nominal Wilson 95% intervals",fontsize=12)
    fig.tight_layout();save_figure(fig,d,"stage1")
    text.extend(["",f"The declared screen returned **{primary['clear_gain']}**. "
        "The primary gain is concentrated at p=.001; the p=.003 comparison is inconclusive. "
        "Separate two-point learning comparisons use matched initialization. "
        "This result alone establishes neither equal logical error rate nor lower latency.","",
        "![Stage 1 comparison](stage1.png)","",f"[Frozen confirmation evidence]({link(args.confirmation)}/summary.json)",""])
    if args.replication:
        r=read(args.replication)
        text.extend(["## Training-seed robustness","",
            "The two additional seeds and final-step checkpoint rule were specified before primary confirmation. "
            "All seeds use the same confirmation cases; their shot counts must not be added as independent data.","",
            "| Training seed | Pool seed | p=.001 fallback | p=.003 fallback | Equal-point mean |",
            "|---|---:|---:|---:|---:|"])
        text.append(f"| 1234 (primary) | 52234 | {100*points[0]['failures']['selected']/points[0]['shots']:.3f}% | "
            f"{100*points[1]['failures']['selected']/points[1]['shots']:.3f}% | {100*primary['mean_selected']:.3f}% |")
        means=[primary["mean_selected"]]
        for row in r["replications"]:
            rates=[p["failures"]/p["shots"] for p in row["points"]];means.append(row["mean_fallback_rate"])
            text.append(f"| {row['training_seed']} | {row['pool_seed']} | {100*rates[0]:.3f}% | {100*rates[1]:.3f}% | {100*row['mean_fallback_rate']:.3f}% |")
        text.extend(["",f"Across these three training seeds, equal-point mean fallback demand is "
            f"{100*np.mean(means):.3f}% ± {100*np.std(means,ddof=1):.3f} percentage points (sample standard deviation). "
            "This is auxiliary method robustness, not completion of the original main training matrix.",""])
    for title,root,strong in (("Full-pipeline logical accuracy",args.accuracy,False),
                              ("Strong-cascade full-pipeline logical accuracy",args.strong_accuracy,True),
                              ("Cycle100 full-pipeline logical accuracy",args.cycle100_accuracy,"cycle100")):
        if root is None:continue
        accuracy=read(root)
        labels=(dict(LABELS,ordinary="Best ordinary100",selected="Learned100",matched_init="Untrained100") if strong=="cycle100" else
                dict(LABELS,ordinary="Best ordinary up to 130",selected="BP100 then learned30",matched_init="BP100 then untrained30") if strong else LABELS)
        text.extend([f"## {title}","",
            "Each method retains its own Stage 1 decision. Identical deterministic fallback work is shared "
            "only for this functional assessment. Per-method requests and actual unique executions are separate; "
            "cached execution times are never performance evidence.","",
            "| p | Method | Fixed target complete | Cases | Logical failures | LER | Nominal Wilson 95% interval | Fallback requests |",
            "|---|---|---|---:|---:|---:|---|---:|"])
        for point in accuracy["points"]:
            for row in point["models"]:
                text.append(f"| {point['p']:g} | {labels[row['decoder']]} | {point['complete']} | {row['shots']} | "
                    f"{row['logical_failures']} | {100*row['logical_rate']:.4f}% | "
                    f"{100*row['nominal_wilson_lo']:.4f}–{100*row['nominal_wilson_hi']:.4f}% | {row['stage1_fallback_requests']} |")
        text.extend(["",f"Actual shared fallback executions: {accuracy['actual_unique_fallback_executions']:,}. "
            "No equivalence margin is asserted. Overlapping Wilson intervals do not prove equal LER.","",
            "Joint one-sided upper bounds on learned-only logical-failure probability use alpha/4 "
            "for four comparisons. They are emitted only for completed fixed samples:","",
            "| p | Baseline | Learned-only failures | Baseline-only failures | Upper probability bound |",
            "|---|---|---:|---:|---:|"])
        for point in accuracy["points"]:
            for name,pair in point["pairs"].items():
                upper=pair["extra_failure_probability_upper"]
                shown=f"{100*upper:.4f}%" if upper is not None else "Not available (budget-limited)"
                text.append(f"| {point['p']:g} | {labels[name]} | {pair['selected_only_failures']} | {pair['baseline_only_failures']} | {shown} |")
        text.append("")
    if args.strong_study:
        strong=read(args.strong_study)
        text.extend(["## Fresh strong-first-pass cascade comparison","",strong["scope"],"",
            "The learned checkpoint is the unchanged primary checkpoint. Ordinary factors and "
            "the comparison pipeline were frozen before these new confirmation cases.","",
            "| p | BP100 | Best ordinary up to 130 | BP100 then learned30 | BP100 then untrained30 | Primary Holm p | Learning Holm p |",
            "|---|---:|---:|---:|---:|---:|---:|"])
        for point in strong["points"]:
            n=point["shots"];counts=point["failures"];best=point["best_ordinary"]
            text.append(f"| {point['p']:g} | {counts['bp100']}/{n} | {best}: {counts[best]}/{n} | "
                f"{counts['strong_cascade']}/{n} | {counts['strong_init']}/{n} | "
                f"{point['primary']['holm_pvalue']:.3g} | {point['learning']['holm_pvalue']:.3g} |")
        text.append("")
    if args.extended_validation:
        extended=read(args.extended_validation)
        text.extend(["## Fixed-checkpoint longer-unroll validation","",extended["scope"],"",
            "| p | Iteration cap | Learned failures | Same-T ordinary failures |",
            "|---|---:|---:|---:|"])
        for row in extended["records"]:
            text.append(f"| {row['p']:g} | {row['T']} | {row['learned_failures']}/{row['shots']} | "
                f"{row['ordinary_failures']}/{row['shots']} |")
        text.append("")
    if args.cycle100_study:
        cycle=read(args.cycle100_study)
        text.extend(["## Independent equal-iteration cycle100 assessment","",cycle["scope"],"",
            "This study keeps the original learned parameters and cycles the schedule to 100 inference "
            "iterations. Native scalar100 factors, ordinary comparison choices and all three source "
            "checkpoints were frozen before the new cases. Source weights have no additional optimizer updates.","",
            "| p | Best ordinary100 | Matched initialization100 | Learned100 seed1234 | Learned100 seed1235 | Learned100 seed1236 | Primary Holm p |",
            "|---|---:|---:|---:|---:|---:|---:|"])
        fig,axes=plt.subplots(1,2,figsize=(9,3.8),sharey=True)
        for point,axis in zip(cycle["points"],axes):
            n=point["shots"];counts=point["failures"];ordinary=point["ordinary_name"]
            text.append(f"| {point['p']:g} | {ordinary}: {counts[ordinary]}/{n} | {counts['matched_init']}/{n} | "
                f"{counts['selected']}/{n} | {counts['seed1235']}/{n} | {counts['seed1236']}/{n} | "
                f"{point['comparisons']['selected']['primary']['holm_pvalue']:.3g} |")
            names=[ordinary,"matched_init","selected"]
            values=np.array([counts[name]/n for name in names])*100
            intervals=np.array([wilson(counts[name],n) for name in names])*100
            axis.bar(np.arange(3),values,color=["#65758b","#a6b5c5","#2d8b77"],width=.65)
            axis.errorbar(np.arange(3),values,yerr=np.stack([values-intervals[:,0],intervals[:,1]-values]),fmt="none",color="black",capsize=4)
            axis.set_xticks(np.arange(3),["Best ordinary100","Untrained100","Learned100"],rotation=15)
            axis.set_title(f"p = {point['p']:g}; n = {n:,}");axis.set_ylim(0,100)
            axis.spines[["top","right"]].set_visible(False)
        axes[0].set_ylabel("Stage 1 fallback demand (%)")
        fig.suptitle("Fresh equal-iteration cycle100 assessment\nNominal Wilson 95% intervals",fontsize=12)
        fig.tight_layout();save_figure(fig,d,"cycle100_stage1")
        text.extend(["",f"Declared clear-gain screen: **{cycle['clear_gain']}**. "
            f"Equal-point mean fallback demand is {100*cycle['mean_fallback']['selected']:.3f}% learned "
            f"versus {100*cycle['mean_ordinary']:.3f}% best ordinary. "
            "Replication checkpoints share the same test cases; their shot counts are not independent extra cases.","",
            "![Fresh cycle100 comparison](cycle100_stage1.png)",""])
    for title,root,filename in (("Primary CPU timing pilot",args.latency,"latency"),
                                ("Secondary longer-BP timing pilot",args.long_latency,"long_latency"),
                                ("Strong-first-pass cascade timing pilot",args.strong_latency,"strong_latency")):
        if root is None:continue
        timing=read(root)
        labels=dict(LABELS,**yaml_read(root/"settings.yaml").get("latency_labels",{}))
        text.extend([f"## {title}","",timing["scope"],"",
            "| p | Method | Measured paired shots | Mean (ms) | p50 (ms) | p99 (ms) | Maximum (ms) | Fallback calls |",
            "|---|---|---:|---:|---:|---:|---:|---:|"])
        fig,axes=plt.subplots(1,2,figsize=(9,3.6),sharey=True)
        for index,(point,axis) in enumerate(zip(timing["points"],axes)):
            raw=[json.loads(line) for line in (root/f"raw-p{index}.jsonl").read_text().splitlines()]
            settings=yaml_read(root/"settings.yaml")
            inputs[str(root/f"raw-p{index}.jsonl")]=hashlib.sha256((root/f"raw-p{index}.jsonl").read_bytes()).hexdigest()
            for row in point["models"]:
                text.append(f"| {point['p']:g} | {labels[row['method']]} | {row['shots']} | {row['lat_mean_us']/1000:.3f} | "
                    f"{row['lat_p50_us']/1000:.3f} | {row['lat_p99_us']/1000:.3f} | {row['lat_max_us']/1000:.3f} | {row['fallback_calls']} |")
                values=np.sort([r["total_us"]/1000 for r in raw if r["method"]==row["method"] and
                    settings["latency_warmup"]<=r["case"]<point["complete_cases_including_warmup"]])
                axis.step(values,np.arange(1,len(values)+1)/len(values),where="post",label=labels[row["method"]])
            axis.set_xscale("log");axis.set_xlabel("Full decode latency (ms)");axis.set_title(f"p = {point['p']:g}")
            axis.spines[["top","right"]].set_visible(False);axis.legend(fontsize=7)
        axes[0].set_ylabel("Empirical cumulative probability")
        fig.tight_layout();save_figure(fig,d,filename)
        text.extend(["",f"![Latency distributions]({filename}.png)","",
            "Paired selected-minus-baseline differences (negative favors the learned model); "
            "intervals are nominal, descriptive paired bootstrap intervals:","",
            "| p | Baseline | Statistic | Difference (ms) | Nominal 95% interval (ms) |",
            "|---|---|---|---:|---|"])
        for point in timing["points"]:
            for name,pair in point.get("paired_nominal_bootstrap",{}).items():
                for metric,value in pair.items():
                    text.append(f"| {point['p']:g} | {labels[name]} | {metric} | {value['difference_us']/1000:.3f} | "
                        f"{value['nominal_95_bootstrap_lo_us']/1000:.3f}–{value['nominal_95_bootstrap_hi_us']/1000:.3f} |")
        if any(point.get("additional_paired_contrasts") for point in timing["points"]):
            text.extend(["","Additional fixed contrasts (candidate minus baseline):","",
                "| p | Candidate | Baseline | Statistic | Difference (ms) | Nominal 95% interval (ms) |",
                "|---|---|---|---|---:|---|"])
            for point in timing["points"]:
                for pair in point.get("additional_paired_contrasts",{}).values():
                    for metric,value in pair["statistics"].items():
                        text.append(f"| {point['p']:g} | {labels[pair['candidate']]} | {labels[pair['baseline']]} | {metric} | "
                            f"{value['difference_us']/1000:.3f} | {value['nominal_95_bootstrap_lo_us']/1000:.3f}–{value['nominal_95_bootstrap_hi_us']/1000:.3f} |")
        text.append("")
    if args.long_controls:
        text.extend(["## Secondary stronger iteration-budget challenge","",
            "This challenge was added after the primary result. T50/T100 factors are tuned on existing "
            "validation and frozen before their confirmation decoding. It does not replace the primary comparison.","",
            "| p | BP iterations | Validation-selected factor | Confirmation fallback demand |",
            "|---|---:|---:|---:|"])
        for root in args.long_controls:
            challenge=read(root)
            for row in challenge["outcomes"]:
                text.append(f"| {challenge['p']:g} | {row['T']} | {row['factor']:g} | "
                    f"{row['failures']}/{row['shots']} ({100*row['failures']/row['shots']:.3f}%) |")
        text.append("")
    text.extend(["## Remaining work","",
        "The original three-seed main E1 training/evaluation, full p grid, main E2 sample regime, "
        "ablations and scaling experiments remain pending. An auxiliary Stage 1 screen or a 256-shot "
        "timing pilot cannot mark M6 complete. No GPU or paid compute was used for this assessment.",""])
    (d/"report.md").write_text("\n".join(text))
    (d/"input_hashes.json").write_text(json.dumps(inputs,indent=2)+"\n")
    shutil.copyfile(__file__,d/"executed_report_source.py")
    print(d)


def yaml_read(path):
    import yaml
    return yaml.safe_load(path.read_text())


if __name__=="__main__":main()

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
    parser.add_argument("--export-acceptance",type=Path)
    parser.add_argument("--launch-decision",type=Path)
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
    if args.cycle100_study:
        latest=read(args.cycle100_study)
        screen="passed" if latest["clear_gain"] else "did not pass"
        direction=all(point["failures"][name]<point["failures"][point["ordinary_name"]]
            for point in latest["points"] for name in ("selected","seed1235","seed1236"))
        seed_statement=("All three training seeds improved in direction; the per-seed corrected tests are reported below. "
            if direction else "Training-seed outcomes differ and are reported below. ")
        brief=["## Current finding","",
            f"The fixed learned100 inference rule {screen} its independent Stage1 screen, with "
            f"{latest['primary_benefits']} corrected primary benefits across {len(latest['points'])} noise points. "
            "It uses the original 1,455 learned source parameters and adds no optimizer updates. "
            +seed_statement,"",
            "| p | Validation-selected ordinary100 fallback | Learned100 fallback | Corrected primary p |",
            "|---|---:|---:|---:|"]
        for point in latest["points"]:
            n=point["shots"];counts=point["failures"];base=point["ordinary_name"]
            brief.append(f"| {point['p']:g} | {counts[base]}/{n} ({100*counts[base]/n:.2f}%) | "
                f"{counts['selected']}/{n} ({100*counts['selected']/n:.2f}%) | "
                f"{point['comparisons']['selected']['primary']['holm_pvalue']:.3g} |")
        cascade_statement="The strong-first-pass cascade is assessed separately below. "
        if args.strong_study:
            cascade=read(args.strong_study)
            if all(point["failures"]["strong_cascade"]>point["failures"][point["best_ordinary"]] for point in cascade["points"]):
                cascade_statement="The strong-first-pass cascade had more fallback requests than its ordinary control at both noise points. "
        brief.extend(["",cascade_statement+
            "The timing pilots do not establish p99 superiority, and this report asserts no LER equivalence margin. "
            "The detailed accuracy, timing and negative controls follow below.",""])
        if args.cycle100_accuracy:
            final_accuracy=read(args.cycle100_accuracy)
            brief.extend(["Cycle100 full-pipeline observed logical accuracy:","",
                "| p | Fixed target complete | Learned100 failures | Ordinary100 failures | Learned-only event upper bound |",
                "|---|---|---:|---:|---:|"])
            for point in final_accuracy["points"]:
                rows={row["decoder"]:row for row in point["models"]}
                upper=point["pairs"]["ordinary"]["extra_failure_probability_upper"]
                bound=f"{100*upper:.4f}%" if upper is not None else "Unavailable"
                brief.append(f"| {point['p']:g} | {point['complete']} | {rows['selected']['logical_failures']}/{point['shots']} | "
                    f"{rows['ordinary']['logical_failures']}/{point['shots']} | {bound} |")
            brief.extend(["","The event bounds are jointly one-sided at the declared family level; "
                "they do not assert a chosen equivalence margin.",""])
            for point in final_accuracy["points"]:
                rows={row["decoder"]:row for row in point["models"]}
                if rows["selected"]["stage1_fallback_requests"]>rows["ordinary"]["stage1_fallback_requests"]:
                    brief.extend([f"The smaller fixed accuracy sample at p={point['p']:g} recorded "
                        f"{rows['selected']['stage1_fallback_requests']} learned versus "
                        f"{rows['ordinary']['stage1_fallback_requests']} ordinary fallback requests. "
                        f"This fixed accuracy prefix contains the first {point['shots']:,} cases of "
                        "the full 4,096-case Stage1 pool.",""])
        if args.long_latency:
            observed_timing=read(args.long_latency)
            brief.extend(["Observed CPU means in the 256-case longer-decoder pilot:","",
                "| p | CPP100 mean (ms) | Tuned native100 mean (ms) | Learned100 mean (ms) |",
                "|---|---:|---:|---:|"])
            for point in observed_timing["points"]:
                rows={row["method"]:row for row in point["models"]}
                brief.append(f"| {point['p']:g} | {rows['bp100']['lat_mean_us']/1000:.3f} | "
                    f"{rows['native100']['lat_mean_us']/1000:.3f} | {rows['learned100']['lat_mean_us']/1000:.3f} |")
            brief.extend(["","These are pilot point estimates; paired intervals and the stronger BP130 comparison "
                "are reported below. No main E2/p99 acceptance is claimed.",""])
        text[4:4]=brief
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
    if args.export_acceptance:
        acceptance=read(args.export_acceptance)
        assert acceptance["checks_passed"] and acceptance["total_exact_cases"]==10000
        assert acceptance["checkpoint_sha256"]==primary["checkpoint_sha256"]
        text.extend(["## Numerical implementation acceptance","",
            "The frozen cycle100 PyTorch model and compiled CPU export matched exactly on 10,000 "
            "saved cases, including full corrections, convergence flags and first-convergence iterations. "
            "The two noise points used separate single-thread CPU workers. These are numerical "
            "acceptance cases, not extra independent performance samples.",""])
    if args.launch_decision:
        decision=read(args.launch_decision,"decision.json")
        assert decision["timing_data_collected"] is False
        text.extend(["## Optional timing extension","",
            f"The optional extra timing run was omitted for schedule: projected duration "
            f"{decision['projected_seconds']:.1f} s, versus {decision['remaining_to_cutoff_seconds']:.1f} s "
            "remaining before its declared cutoff. No extra timing observations or planned 1,024-case "
            "pool were collected. The three completed 256-case pilots are reported separately.",""])
    text.extend(["## Evidence audits","","| Evidence | Audit |","|---|---|"])
    audit_roots=[("Original Stage1 confirmation",args.confirmation),("Training-seed replications",args.replication),
        ("Original functional accuracy",args.accuracy),("Primary CPU timing",args.latency),
        ("Longer-BP CPU timing",args.long_latency),("Strong-cascade confirmation",args.strong_study),
        ("Strong-cascade CPU timing",args.strong_latency),("Strong-cascade functional accuracy",args.strong_accuracy),
        ("Cycle100 confirmation",args.cycle100_study),("Cycle100 functional accuracy",args.cycle100_accuracy)]
    for label,root in audit_roots:
        if root is None:continue
        if (root/"audit.json").exists():
            audit=read(root,"audit.json");assert audit["checks_passed"]
            text.append(f"| {label} | [Passed]({link(root)}/audit.json) |")
        else:text.append(f"| {label} | Pending |")
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

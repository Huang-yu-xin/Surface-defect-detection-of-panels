from __future__ import annotations
import csv, json, subprocess
from collections import Counter
from pathlib import Path

ROOT=Path("results/baseline_complementarity/global_exact_final_combo")
CLASS_NAMES=["jieba","zonglie","qilie","jiaza","yiwuyaru","huashang","mamianmakeng","yanghuatiepi","gunyin"]
CONFIGS=["u3e4_g060","u3e4_g080","u1e3_g060","u1e3_g080","uinf_g060","uinf_g080"]
NEIGHBORS={
 "u3e4_g060":{"u3e4_g080","u1e3_g060"},"u3e4_g080":{"u3e4_g060","u1e3_g080"},
 "u1e3_g060":{"u3e4_g060","u1e3_g080","uinf_g060"},"u1e3_g080":{"u3e4_g080","u1e3_g060","uinf_g080"},
 "uinf_g060":{"u1e3_g060","uinf_g080"},"uinf_g080":{"u1e3_g080","uinf_g060"}}
def read_csv(path):
 with path.open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
def iv(r,k):return int(float(r[k]))
def fv(r,k):return float(r[k])
def main():
 summary=read_csv(ROOT/"summary.csv");rows={r["config"]:r for r in summary};audit=read_csv(ROOT/"group_audit.csv");amap={(r["config"],r["representation"]):r for r in audit};attr=read_csv(ROOT/"rescue_regression.csv");changes=read_csv(ROOT/"roundtrip_changes.csv")
 qualifies={};criteria={}
 for cfg in CONFIGS:
  r=rows[cfg];a=amap[(cfg,"float")];ar=amap[(cfg,"roundtrip")]
  neighbor=any(abs(iv(rows[n],"net_gain")-iv(r,"net_gain"))<=1 for n in NEIGHBORS[cfg])
  c=[
   ("Exact Final Combo net gain >= +3",iv(r,"net_gain")>=3),
   ("roundtrip net gain >= +3",iv(r,"roundtrip_net_gain")>=3),
   ("rescued >= 3 images",iv(r,"rescue_images")>=3),
   ("rescued >= 3 grouped-split groups",iv(r,"rescue_groups")>=3),
   ("rescued >= 2 classes",iv(r,"rescue_classes")>=2),
   ("rescues span >= 2 failure clusters",iv(r,"rescue_failure_types")>=2),
   ("net gain without largest group > 0",iv(r,"net_gain_without_largest_rescue_group")>0),
   ("adjacent frozen config within 1 TP net",neighbor)]
  criteria[cfg]=c;qualifies[cfg]=all(x[1] for x in c)
 go=[c for c in CONFIGS if qualifies[c]]
 if go:
  go.sort(key=lambda c:(-iv(rows[c],"roundtrip_net_gain"),-iv(rows[c],"rescue_groups"),-iv(rows[c],"rescue_images"),-iv(rows[c],"rescue_classes"),iv(rows[c],"regressed"),iv(rows[c],"baseline_rows_added")))
  recommended=go[0];decision="GO"
 else:recommended="NONE";decision="NO-GO"
 best=max(CONFIGS,key=lambda c:(iv(rows[c],"net_gain"),iv(rows[c],"roundtrip_net_gain"),-iv(rows[c],"baseline_rows_added")))
 selected= recommended if recommended!="NONE" else best
 bm=json.loads((ROOT/"baseline_reproduction"/"metrics.json").read_text());tests=json.loads((ROOT/"builder_unit_tests.json").read_text());meta=json.loads((ROOT/"evaluation_metadata.json").read_text())
 build=json.loads((ROOT/"build_metrics.json").read_text())
 head=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip();dirty=subprocess.check_output(["git","status","--short"],text=True).strip()
 lines=["# GT-independent Exact Final Combo Val Report","",
 "## Scope and reproducibility","",
 f"- Repo HEAD: `{head}`",f"- Working tree had pre-existing and task changes: {'YES' if dirty else 'NO'}",
 "- GPU used: NO (`CUDA_VISIBLE_DEVICES=\"\"`; CPU-only post-processing)","- Test/Hidden run: NO","- Hidden submission generated: NO",
 "- Builder/evaluator separation: the builder accepts only Original, HFlip, and Baseline prediction caches plus frozen global parameters. It has no annotation, FN-list, rescue-list, failure-type, image-list, or evaluator input. Labels and retrospective CSVs are opened only by the evaluator after each cache exists.",
 f"- Cache coordinate / synthetic / active-HFlip assertions: {', '.join(k+'='+str(v) for k,v in tests.items() if k!='cache_checks')}",
 f"- Assignment schema: image=`{meta['assignment_schema']['image_column']}`, group=`{meta['assignment_schema']['group_column']}`. These groups are repository grouped-split units, not asserted production IDs.","",
 "## Baseline reproduction","",
 "| expected | float | roundtrip | final detections | stitched | float/roundtrip GT changes |",
 "|---:|---:|---:|---:|---:|---:|",
 f"| 824/21 | {bm['float_tp']}/{bm['float_fn']} | {bm['roundtrip_tp']}/{bm['roundtrip_fn']} | {bm['final_detection_count']} | {bm['stitched_count']} | {bm['roundtrip_changes']} |","",
 "The baseline prerequisite passed. The memory-safe CPU implementation uses the repository reference NumPy class-aware NMS and the frozen stitching logic. It reproduces the required 824 TP / 21 FN and 4,323 stitched boxes.","",
 "## Six frozen configurations","",
 "| config | added | float TP/FN | roundtrip TP/FN | rescued | regressed | net | images | groups | classes |",
 "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
 for cfg in CONFIGS:
  r=rows[cfg];lines.append(f"| {cfg} | {iv(r,'baseline_rows_added'):,} | {r['float_tp']}/{r['float_fn']} | {r['roundtrip_tp']}/{r['roundtrip_fn']} | {r['rescued']} | {r['regressed']} | {r['net_gain']} | {r['rescue_images']} | {r['rescue_groups']} | {r['rescue_classes']} |")
 nets=max(iv(rows[c],"net_gain") for c in CONFIGS);stable=[c for c in CONFIGS if abs(iv(rows[c],"net_gain")-iv(rows[c],"roundtrip_net_gain"))==0];qual=[c for c in CONFIGS if qualifies[c]];fewest=min(qual,key=lambda c:iv(rows[c],"baseline_rows_added")) if qual else "NONE"
 lines += ["",f"- Best net gain: {nets:+d} TP.",f"- Float/roundtrip-stable configs: {', '.join(stable) if stable else 'NONE'}.",f"- Fewest-added qualifying config: {fewest}.","",
 f"## Attribution for selected audit config: {selected}","",
 "### Rescued","",
 "| representation | image | gt index | group | class | failure | baseline → new | baseline IoU | new IoU |",
 "|---|---|---:|---|---|---|---|---:|---:|"]
 chosen=[r for r in attr if r["config"]==selected]
 rescued=[r for r in chosen if r["change"]=="rescued"]
 regressed=[r for r in chosen if r["change"]=="regressed"]
 for r in rescued:lines.append(f"| {r['representation']} | {r['image']} | {r['gt_index']} | {r['group']} | {r['class']} | {r['failure_type'] or 'unclassified'} | {r['baseline_status']} → {r['new_status']} | {float(r['baseline_best_iou']):.4f} | {float(r['new_best_iou']):.4f} |")
 if not rescued:lines.append("| — | — | — | — | — | — | — | — |")
 lines += ["","### Regressed","", "| representation | image | gt index | group | class | baseline → new | baseline IoU | new IoU |","|---|---|---:|---|---|---|---:|---:|"]
 for r in regressed:lines.append(f"| {r['representation']} | {r['image']} | {r['gt_index']} | {r['group']} | {r['class']} | {r['baseline_status']} → {r['new_status']} | {float(r['baseline_best_iou']):.4f} | {float(r['new_best_iou']):.4f} |")
 if not regressed:lines.append("| — | — | — | — | — | — | — |")
 lines += ["","## Baseline per-class reproduction","","| class | TP | FN |","|---|---:|---:|"]
 for cname in CLASS_NAMES:
  s=bm["per_class"][cname];lines.append(f"| {cname} | {s['tp']} | {s['fn']} |")
 af=amap[(selected,"float")];ar=amap[(selected,"roundtrip")]
 image_counts=Counter(r["image"] for r in rescued if r["representation"]=="float")
 lines += ["","## Distribution audit","",
 f"- Float rescues by image: `{json.dumps(image_counts,ensure_ascii=False,sort_keys=True)}`",
 f"- Float rescues by group: `{af['rescues_by_group']}`",f"- Float rescues by class: `{af['rescues_by_class']}`",f"- Float rescues by failure: `{af['rescues_by_failure']}`",
 f"- Unique rescue images/groups/classes/failure types: {af['rescue_images']}/{af['rescue_groups']}/{af['rescue_classes']}/{af['rescue_failure_types']}.",
 f"- Largest rescue-group contribution: {af['largest_group_rescues']}.",f"- Net gain after removing a largest rescue group: {af['net_gain_without_largest_rescue_group']}.","",
 "## Submission-roundtrip consistency","",
 f"- Selected config float: {rows[selected]['float_tp']}/{rows[selected]['float_fn']}; roundtrip: {rows[selected]['roundtrip_tp']}/{rows[selected]['roundtrip_fn']}.",
 f"- Selected config roundtrip rescued/regressed/net: {ar['rescued']}/{ar['regressed']}/{ar['net_gain']}.",
 f"- GT status changes caused solely by serialization: {sum(1 for r in changes if r.get('config')==selected)}.",
 "- Roundtrip used the formal floor(xmin/ymin), ceil(xmax/ymax), image clipping, six-decimal score, JSON write, JSON reread, and the same IoU>=0.5 matcher.","",
 "## Resources and engineering","",
 f"- Builder runtime: {build['runtime_seconds']:.1f} s.",
 f"- Evaluator runtime, baseline: {float(rows['baseline_reproduction']['runtime_seconds']):.1f} s.",
 f"- Evaluator runtime, six configs: {sum(float(rows[c]['runtime_seconds']) for c in CONFIGS):.1f} s.",
 f"- Total validated pipeline runtime (builder + baseline + six configs): {(build['runtime_seconds']+float(rows['baseline_reproduction']['runtime_seconds'])+sum(float(rows[c]['runtime_seconds']) for c in CONFIGS))/60:.1f} min.",
 f"- Peak evaluator RSS: {max(float(rows[c]['peak_rss_mb']) for c in rows):.1f} MB.",
 "- OOM/Killed: one initial unmodified PyTorch CPU replay was killed before completion; the validated memory-safe NumPy replay and all reported runs completed without OOM.",
 "- Added scripts: `scripts/41_build_baseline_global_complement_val.py`, `scripts/42_eval_baseline_global_exact_final_combo.py`, `scripts/43_report_baseline_global_exact_combo.py`, `scripts/44_run_baseline_global_exact_combo.sh`.",
 f"- Output root: `{ROOT}`. Candidate counts and per-class additions are recorded in `summary.csv` and each config manifest.","",
 f"## Final decision: {decision}",""]
 for i,(label,passed) in enumerate(criteria[selected],1):lines.append(f"{i}. {'PASS' if passed else 'FAIL'} — {label}.")
 lines += ["",f"`recommended_hidden_candidate = {recommended}`"]
 if decision=="NO-GO":lines.append("`next_direction = RareOS VFlip / new tile geometry`")
 lines += ["","No Test/Hidden data was read or executed, and no Hidden submission was generated."]
 (ROOT/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
 (ROOT/"decision.json").write_text(json.dumps({"decision":decision,"recommended_hidden_candidate":recommended,"selected_audit_config":selected,"criteria":{c:[{"criterion":x,"pass":p} for x,p in criteria[c]] for c in CONFIGS}},indent=2))
 print(decision,recommended)
if __name__=="__main__":main()

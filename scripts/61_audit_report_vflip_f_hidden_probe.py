"""Strict streaming integrity, BASE/F diff, reconciliation, and final report. No GT."""
from __future__ import annotations
import csv,hashlib,json,platform,resource,subprocess,time
from collections import Counter
from pathlib import Path
import numpy as np
import importlib.util,sys

ROOT=Path("results/vflip_factorial/hidden_probe_f")
BASE_ROOT=Path("results/baseline_complementarity/global_exact_hidden_probe/submission")
BASE=BASE_ROOT/"submission_u1e3_g060.json";NEW=ROOT/"submission/submission_vflip_f.json"
EXPECTED_SHA="dfaef6806e43773dbad00ac7c4fb9bc4d7d880716a0570f113906f6e89daeca6";EXPECTED_COUNT=3392951
CLASSES=["jieba","zonglie","qilie","jiaza","yiwuyaru","huashang","mamianmakeng","yanghuatiepi","gunyin"]
def module(path,name):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
A=module("scripts/47_audit_hidden_probe_submission.py","strict_stream_audit")
def read(path):return json.loads(Path(path).read_text())
def write(path,obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj,indent=2,allow_nan=False)+"\n")
def csvmap(path,key):
 with Path(path).open(encoding="utf-8-sig",newline="") as f:rows=list(csv.DictReader(f))
 out={r[key]:r for r in rows};assert len(out)==len(rows);return out
def stats(v):
 a=np.asarray(v,float);return {"mean":float(a.mean()),"p50":float(np.percentile(a,50)),"p90":float(np.percentile(a,90)),"p95":float(np.percentile(a,95)),"p99":float(np.percentile(a,99)),"max":float(a.max())}
def sha(path):return A.sha(Path(path))
def main():
 start=time.time();om=read("results/test_final_cache/original/manifest.json");dims={x["image_name"]:(x["width"],x["height"]) for x in om["items"]}
 bm=read(BASE_ROOT/"metrics.json");fm=read(ROOT/"f_cache/manifest.json");ca=read(ROOT/"audit/cache_audit.json");nm=read(ROOT/"submission/metrics.json");geo=read(ROOT/"audit/geometry_tests.json")
 print("START strict streaming integrity BASE",flush=True);bi=A.integrity(BASE,dims)
 print("START strict streaming integrity BASE+F",flush=True);ni=A.integrity(NEW,dims)
 write(ROOT/"audit/baseline_integrity.json",bi);write(ROOT/"audit/submission_integrity.json",ni)
 print("START streaming multiset diff",flush=True);diff,perdiff=A.diff_submissions(BASE,NEW)
 diff.update({"baseline_detections":bi["detections"],"new_detections":ni["detections"],"delta":ni["detections"]-bi["detections"],"baseline_only_fraction":diff["baseline_only_detections"]/bi["detections"]})
 write(ROOT/"submission/submission_diff_summary.json",diff)
 with (ROOT/"audit/submission_per_image_diff.csv").open("w",newline="",encoding="utf-8") as f:
  w=csv.DictWriter(f,fieldnames=list(perdiff[0]));w.writeheader();w.writerows(perdiff)
 basepi=csvmap(BASE_ROOT/"per_image.csv","image_id");newpi=csvmap(ROOT/"submission/per_image.csv","image_id")
 counts=[x["candidate_count"] for x in fm["items"]];deltas=[int(newpi[n]["final_detection_count"])-int(basepi[n]["final_detection_count"]) for n in dims]
 finaldist=stats(deltas);topdelta=sorted(zip(dims,deltas),key=lambda x:x[1],reverse=True)[:20]
 valf=read("results/vflip_factorial/full/cache/manifest.json");valo=read("results/fn_analysis/cache/manifest.json")
 valfa=read("results/vflip_factorial/audit/cache_full.json")
 valbase=read("results/vflip_factorial/exact_combo/baseline/metrics.json");valnew=read("results/vflip_factorial/exact_combo/full/metrics.json")
 val_final_delta=valnew["final_detection_count"]-valbase["final_detection_count"]
 compare={"candidate_per_image":{"val":valf["candidate_rows"]/474,"test":fm["candidate_rows"]/669},"f_over_o":{"val":valf["candidate_rows"]/valo["total_candidates"],"test":fm["candidate_rows"]/om["total_candidates"]},"final_delta_per_image":{"val":val_final_delta/474,"test":diff["delta"]/669},"stitched_change_per_image":{"val":(valnew["stitched_count"]-valbase["stitched_count"])/474,"test":(nm["stitched"]-bm["stitched"])/669}}
 for x in compare.values():x["test_over_val"]=x["test"]/x["val"] if x["val"] else None
 anomaly=[]
 if not .5<=ca["test_over_val_candidates_per_image"]<=2:anomaly.append("F candidate/image collapse or explosion")
 if diff["baseline_only_fraction"]>=.05:anomaly.append("baseline-only fraction >=5%")
 if not all([geo["status"]=="PASS",ca["status"]=="PASS",bi["json_valid"],ni["json_valid"]]):anomaly.append("structural audit failure")
 checks={
  "baseline_detection_count_exact":bi["detections"]==EXPECTED_COUNT==bm["detections"],"baseline_sha_exact":bi["sha256"]==EXPECTED_SHA==bm["sha256"],
  "geometry_pass":geo["status"]=="PASS","f_cache_669":fm["image_count"]==669==ca["image_count"],"f_cache_audit_pass":ca["status"]=="PASS",
  "f_parameters_frozen":all(fm[k]==v for k,v in {"tile_size":1280,"stride":768,"conf":1e-5,"tile_iou":.60,"max_det":1000,"half":True}.items()),
  "source_order_frozen":nm["source_order"]==["active_O_plus_Baseline","selected_HFlip","F_all_classes_all_candidates"],
  "submission_integrity":all(ni[k] for k in ["json_valid","schema_valid","category_valid","bbox_valid","score_valid","image_valid"]),
  "baseline_integrity":all(bi[k] for k in ["json_valid","schema_valid","category_valid","bbox_valid","score_valid","image_valid"]),
  "metrics_reconcile":ni["detections"]==nm["detections"] and ni["sha256"]==nm["sha256"] and ni["file_size_bytes"]==nm["file_size_bytes"] and sum(nm["class_counts"].values())==nm["detections"],
  "per_image_reconcile":set(basepi)==set(newpi)==set(dims) and sum(int(r["final_detection_count"]) for r in newpi.values())==ni["detections"],
  "diff_reconcile":diff["exactly_identical_detections"]+diff["baseline_only_detections"]==bi["detections"] and diff["exactly_identical_detections"]+diff["new_only_detections"]==ni["detections"],
  "baseline_only_below_5pct":diff["baseline_only_fraction"]<.05,"distribution_no_anomaly":not anomaly,"streaming_completed":nm["streaming"] and nm["exit_code"]==0 and not nm["oom_killed"],
  "no_gt_access_attested":not fm["ground_truth_accessed"] and not nm["ground_truth_accessed"],"no_platform_submission":not fm["competition_submission_performed"] and not nm["competition_submission_performed"]}
 decision="READY_FOR_MANUAL_HIDDEN_SUBMISSION" if all(checks.values()) else "STOP_AND_INVESTIGATE"
 distribution={"f_raw_per_image":ca["per_image"],"final_delta_per_image":finaldist,"val_test":compare,"f_class_counts":ca["class_counts"],"top20_f":ca["top20"],"top20_final_delta":[{"image":n,"delta":d} for n,d in topdelta],"top2_concentration":ca["top2_concentration"],"anomaly":bool(anomaly),"reasons":anomaly}
 write(ROOT/"audit/distribution_summary.json",distribution);write(ROOT/"audit/reconciliation_checks.json",checks)
 with (ROOT/"audit/distribution_audit.csv").open("w",newline="",encoding="utf-8") as f:
  w=csv.DictWriter(f,fieldnames=["metric","mean","p50","p90","p95","p99","max"]);w.writeheader();w.writerow({"metric":"f_raw_candidates",**ca["per_image"]});w.writerow({"metric":"final_detection_delta",**finaldist})
 perclass=[]
 for i,c in enumerate(CLASSES):
  vc=valfa["class_counts"][i];tc=ca["class_counts"][str(i)]
  perclass.append({"class":c,"val_f_candidates":vc,"test_f_candidates":tc,"val_share":vc/valf["candidate_rows"],"test_share":tc/fm["candidate_rows"],"test_val_share_ratio":(tc/fm["candidate_rows"])/(vc/valf["candidate_rows"])})
 with (ROOT/"audit/per_class_audit.csv").open("w",newline="",encoding="utf-8") as f:
  w=csv.DictWriter(f,fieldnames=list(perclass[0]));w.writeheader();w.writerows(perclass)
 outliers=[{"kind":"f_raw_candidates","rank":i+1,"image":x["image"],"value":x["count"]} for i,x in enumerate(ca["top20"])] + [{"kind":"final_detection_delta","rank":i+1,"image":n,"value":d} for i,(n,d) in enumerate(topdelta)]
 with (ROOT/"audit/outlier_images.csv").open("w",newline="",encoding="utf-8") as f:
  w=csv.DictWriter(f,fieldnames=list(outliers[0]));w.writeheader();w.writerows(outliers)
 manifest={k:v for k,v in ni.items() if k!="per_image_counts"};manifest.update({"config":"vflip_f","decision":decision,"source_order":nm["source_order"],"frozen_parameters":{"tile_size":1280,"stride":768,"conf":1e-5,"tile_iou":.60,"max_det":1000,"batch":6,"half":True},"competition_submission_performed":False})
 write(ROOT/"submission/submission_manifest.json",manifest)
 context={"repo_head":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),"git_status_short":subprocess.check_output(["git","status","--short"],text=True).splitlines(),"python":platform.python_version(),"torch":None,"torchvision":None,"ultralytics":None,"gpu":subprocess.check_output(["nvidia-smi","--query-gpu=name,driver_version,memory.total","--format=csv,noheader"],text=True).strip(),"test_inference_run":"F only","test_hidden_gt_accessed":False,"competition_submission_performed":False}
 import torch,torchvision,ultralytics
 context.update(torch=torch.__version__,torchvision=torchvision.__version__,ultralytics=ultralytics.__version__,weights=fm["weights"],weights_sha256=fm["weights_sha256"]);write(ROOT/"audit/execution_context.json",context)
 total=time.time()-start;engineering={"baseline_reproduction_runtime_seconds":bm["runtime_seconds"],"geometry_runtime_seconds":geo["runtime_seconds"],"f_inference_runtime_seconds":fm["runtime_seconds"],"cache_audit_runtime_seconds":ca["runtime_seconds"],"final_combo_runtime_seconds":nm["runtime_seconds"],"submission_audit_runtime_seconds":total,"total_runtime_seconds":bm["runtime_seconds"]+geo["runtime_seconds"]+fm["runtime_seconds"]+ca["runtime_seconds"]+nm["runtime_seconds"]+total,"peak_rss_mb":max(bm["peak_rss_mb"],nm["peak_rss_mb"],resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024),"gpu_peak_allocated_mb":fm["peak_gpu_allocated_mb"],"gpu_peak_reserved_mb":fm["peak_gpu_reserved_mb"],"oom_events":fm["oom_events"],"killed":False};write(ROOT/"audit/engineering_metrics.json",engineering)
 lines=["# RareOS VFlip-F Frozen Test-side Audit","","## A. Execution status","",f"- Repo HEAD: `{context['repo_head']}`",f"- Workspace dirty: {'YES' if context['git_status_short'] else 'NO'}",f"- GPU: {context['gpu']}",f"- Python/PyTorch/Torchvision/Ultralytics: {context['python']} / {context['torch']} / {context['torchvision']} / {context['ultralytics']}",f"- RareOS weights: `{context['weights']}`; SHA256 `{context['weights_sha256']}`","- Test inference run = F only","- Hidden/Test GT accessed = NO","- competition submission performed = NO","","## B. Current 98.19 baseline reproduction","",f"- Expected / actual detections: {EXPECTED_COUNT:,} / {bi['detections']:,}",f"- Expected / actual SHA256: `{EXPECTED_SHA}` / `{bi['sha256']}`",f"- Result: {'PASS' if checks['baseline_detection_count_exact'] and checks['baseline_sha_exact'] else 'FAIL'}","","## C. F geometry","",*([f"- {k}: {v}" for k,v in geo['checks'].items()]),"","## D. F Test cache","",f"- Images: {fm['image_count']}",f"- Candidate rows: {fm['candidate_rows']:,}",f"- Coordinate residual: {ca['coord_residual']}",f"- NaN/Inf / invalid bbox: {ca['nan_inf_count']} / {ca['invalid_bbox_count']}",f"- Runtime: {fm['runtime_seconds']:.2f} s; GPU peak allocated/reserved: {fm['peak_gpu_allocated_mb']:.2f}/{fm['peak_gpu_reserved_mb']:.2f} MiB","","## E. F candidate distribution","","| metric | Val | Test | Test/Val |","|---|---:|---:|---:|"]
 for k,x in compare.items():lines.append(f"| {k} | {x['val']:.6f} | {x['test']:.6f} | {x['test_over_val']:.6f} |")
 lines += ["",f"Test per-image: mean={ca['per_image']['mean']:.2f}, p50={ca['per_image']['p50']:.2f}, p90={ca['per_image']['p90']:.2f}, p95={ca['per_image']['p95']:.2f}, p99={ca['per_image']['p99']:.2f}, max={ca['per_image']['max']:.0f}; top2 concentration={ca['top2_concentration']:.4%}.","","| class | Val F | Test F | Val share | Test share | Test/Val share |","|---|---:|---:|---:|---:|---:|"]
 for x in perclass:lines.append(f"| {x['class']} | {x['val_f_candidates']:,} | {x['test_f_candidates']:,} | {x['val_share']:.6f} | {x['test_share']:.6f} | {x['test_val_share_ratio']:.6f} |")
 lines += ["","## F. Exact BASE+F Final Combo","",f"- BASE detections / BASE+F / delta: {bi['detections']:,} / {ni['detections']:,} / {diff['delta']:+,}",f"- BASE stitched / BASE+F / delta: {bm['stitched']:,} / {nm['stitched']:,} / {nm['stitched']-bm['stitched']:+,}",f"- F participation / mixed-source stitched: {nm['f_participating_stitched']:,} / {nm['mixed_source_stitched']:,}",f"- Per-image final delta: mean={finaldist['mean']:.2f}, p50={finaldist['p50']:.2f}, p90={finaldist['p90']:.2f}, p95={finaldist['p95']:.2f}, p99={finaldist['p99']:.2f}, max={finaldist['max']:.0f}","","## G. Submission diff","",f"- Exactly identical: {diff['exactly_identical_detections']:,}",f"- New-only: {diff['new_only_detections']:,}",f"- Baseline-only: {diff['baseline_only_detections']:,} ({diff['baseline_only_fraction']:.4%})",f"- Same bbox/category with score replacement: {diff['same_bbox_score_replacements']:,}","","## H. Submission integrity","",f"- Path: `{ni['path']}`",f"- SHA256: `{ni['sha256']}`",f"- File size: {ni['file_size_bytes']:,} bytes",f"- Detections/images: {ni['detections']:,} / {ni['image_count']}",f"- JSON/schema/bbox/score/image: {ni['json_valid']}/{ni['schema_valid']}/{ni['bbox_valid']}/{ni['score_valid']}/{ni['image_valid']}","","## I. Engineering","",f"- Total runtime: {engineering['total_runtime_seconds']:.2f} s; F inference: {fm['runtime_seconds']:.2f} s",f"- Peak RSS: {engineering['peak_rss_mb']:.2f} MiB; GPU allocated/reserved: {engineering['gpu_peak_allocated_mb']:.2f}/{engineering['gpu_peak_reserved_mb']:.2f} MiB",f"- OOM/Killed: {bool(fm['oom_events'])}/NO","","## J. Test-side final decision","",f"**{decision}**","",f"recommended_submission = `{ni['path'] if decision.startswith('READY') else None}`",f"sha256 = `{ni['sha256']}`",f"detections = {ni['detections']}",f"file_size = {ni['file_size_bytes']}","","competition submission performed = NO","","Hidden result interpretation remains frozen exactly as specified: +3 or more promotes F; +1/+2 retains it without D/G; zero freezes VFlip; regression first triggers checksum/pipeline verification. No automatic upload occurred."]
 (ROOT/"report.md").write_text("\n".join(lines)+"\n");write(ROOT/"decision.json",{"decision":decision,"checks":checks,"recommended_submission":ni["path"] if decision.startswith("READY") else None})
 with (ROOT/"summary.csv").open("w",newline="",encoding="utf-8") as f:
  row={"decision":decision,"baseline_detections":bi["detections"],"final_detections":ni["detections"],"delta":diff["delta"],"stitched":nm["stitched"],"sha256":ni["sha256"],"file_size_bytes":ni["file_size_bytes"]};w=csv.DictWriter(f,fieldnames=list(row));w.writeheader();w.writerow(row)
 print(json.dumps({"decision":decision,"submission":ni["path"],"sha256":ni["sha256"],"detections":ni["detections"],"diff":diff,"engineering":engineering},indent=2))
if __name__=="__main__":main()
